#!/usr/bin/env python3
"""使用 DistilGPT2 运行可重复的 TruthfulQA 解码实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
REFERENCE_COLUMNS = [
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
    "Correct Answers",
    "Incorrect Answers",
    "Source",
]


def parse_args() -> argparse.Namespace:
    """读取数据选择、解码参数和输出位置等命令行选项。

    返回：
        完整实验配置。默认值分别覆盖纯 top-p、纯 top-k 和二者联合模式，
        并让每个实验单元独立生成25次。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/generations_modes_antirepeat.csv")
    parser.add_argument("--sample-output", default="outputs/sampled_questions.csv")
    parser.add_argument("--data-path", default="data/TruthfulQA.csv")
    parser.add_argument("--model-name", default="distilbert/distilgpt2")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "category_all"],
        default="random",
        help="random=从全部问题随机抽样；category_all=纳入指定类别的全部问题。",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="sampling-mode=category_all 时必填；类别名称不区分大小写。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--limit-settings", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--warmup-generations", type=int, default=3)
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用 PyTorch deterministic algorithms；某些设备/算子可能不支持。",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过输出文件中已经完成的问题-参数-重复编号组合。",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.7, 1.0, 1.5],
    )
    parser.add_argument(
        "--top-ps",
        nargs="+",
        type=float,
        default=[0.6, 0.80, 0.90, 0.95],
    )
    parser.add_argument(
        "--top-ks",
        nargs="+",
        type=int,
        default=[0, 10, 15, 20, 25, 30],
        help="取值为 0 时关闭 top-k 筛选。",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--eos-boost-start",
        type=int,
        default=0,
        help="生成该数量的新 token 后开始逐步提高 EOS 概率；0 表示关闭。",
    )
    parser.add_argument("--eos-boost-factor", type=float, default=1.03)
    return parser.parse_args()


def build_decoding_settings(args: argparse.Namespace) -> list[tuple[str, float, float, int]]:
    """构造三类解码参数组合。

    三类模式：
    1. pure_top_p：
       只启用 top-p，令 top_k=0。

    2. pure_top_k：
       只启用 top-k，令 top_p=1.0。

    3. top_p_x_top_k：
       同时启用 top-p 和 top-k。

    每一种模式都会分别搭配 args.temperatures 中的所有 temperature。

    返回：
        每个元素都是：
        (
            decoding_mode,
            temperature,
            top_p,
            top_k,
        )
    """
    # 只保留小于 1.0 的 top-p。top-p == 1则相当于没有使用top-p选择
    top_ps = [value for value in args.top_ps if value < 1.0] 

    # 只保留正的 top-k。
    # top_k=0 表示关闭 top-k，因此不属于真正启用的 top-k 设置。
    positive_top_ks = [value for value in args.top_ks if value > 0]


    # 用于保存所有解码组合。
    #
    # 每个 tuple 的格式：
    # (
    #     模式名称,
    #     temperature,
    #     top_p,
    #     top_k,
    # )

    # 对每个 temperature，分别构造三类设置。
    settings: list[tuple[str, float, float, int]] = []
    for temperature in args.temperatures:
        settings.extend(("pure_top_p", temperature, top_p, 0) for top_p in top_ps)
        settings.extend(("pure_top_k", temperature, 1.0, top_k) for top_k in positive_top_ks)
        settings.extend(
            ("top_p_x_top_k", temperature, top_p, top_k)
            for top_p in top_ps
            for top_k in positive_top_ks
        )
    if not top_ps:
        raise ValueError("--top-ps must contain at least one value below 1.0")
    if not positive_top_ks:
        raise ValueError("--top-ks must contain at least one positive value")
    return settings


def stable_seed(*parts: object) -> int:
    """为一次模拟重复生成不同但可复现的局部随机种子。
    stable_seed does not remove randomness across repetitions. 
    It assigns a different but reproducible random seed to each repetition,
    making every individual random draw reproducible under the same computational environment

    同一次重复在重新运行时是deterministic的；
    不同重复之间仍然是不同的random draws；
    100次重复不会全部变成相同答案。

    可以把它理解成“可以回放的随机抽取”。每次抽取仍然随机，
    但stable seed记录了每次应该使用哪一套随机数，
    因此以后可以准确回放这次抽取。

    参数：
        *parts: 主种子、问题编号、解码参数和重复编号。

    返回：
        由 SHA-256 稳定映射得到的整数。重复编号改变时随机抽样也会改变；
        重新运行同一条实验记录时则可复现原种子。

    说明：
        “稳定”表示种子生成规则可复现，不表示100次重复使用同一个种子。
    """
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def load_questions(
    path: Path,
    n: int,
    seed: int,
    sampling_mode: str = "random",
    category: str | None = None,
) -> pd.DataFrame:
    """下载并验证 TruthfulQA 数据，再进行不放回问题抽样。
    默认是50个问题。

    参数：
        path: 官方 TruthfulQA CSV 的本地缓存路径。
        n: random 模式下抽取的问题数；类别模式中忽略。
        seed: 使不放回简单随机抽样可以复现的主种子。
        sampling_mode: ``random`` 或 ``category_all``。
        category: 类别模式下需要完整纳入的 TruthfulQA 类别。

    返回：
        抽取后的问题表，保留所有参考答案列和稳定原始行号。抽样前先删除
        问题文本完全重复的记录。
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(TRUTHFULQA_URL, context=context) as response:
            path.write_bytes(response.read())

    data = pd.read_csv(path)
    missing = [column for column in REFERENCE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing TruthfulQA columns: {missing}")

    data = data[REFERENCE_COLUMNS].dropna(subset=REFERENCE_COLUMNS[:-1]).copy()
    data = data.drop_duplicates("Question").reset_index(names="source_row_id")
    if sampling_mode == "random":
        if category is not None:
            raise ValueError("--category is only valid with --sampling-mode category_all")
        if n and n < len(data):
            data = data.sample(n=n, replace=False, random_state=seed).sort_values("source_row_id")
    elif sampling_mode == "category_all":
        if not category or not category.strip():
            available = ", ".join(sorted(data["Category"].unique()))
            raise ValueError(
                "--category is required with --sampling-mode category_all. "
                f"Available categories: {available}"
            )
        matches = data["Category"].str.casefold() == category.strip().casefold()
        if not matches.any():
            available = ", ".join(sorted(data["Category"].unique()))
            raise ValueError(f"Unknown category {category!r}. Available categories: {available}")
        data = data.loc[matches].sort_values("source_row_id")
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    data = data.reset_index(drop=True)
    data.insert(0, "prompt_id", np.arange(len(data), dtype=int))
    return data


def choose_device(requested: str | None) -> str:
    """选择用户指定的计算设备；未指定时优先 CUDA，否则使用 CPU。

    参数：
        requested: 用户可选的设备名称。

    返回：
        PyTorch 使用的设备字符串。
    """
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: Path) -> str:
    """计算输入数据文件的 SHA-256。
    确认两次实验使用的数据文件完全相同。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_config(args: argparse.Namespace, data_path: Path) -> dict[str, object]:
    """构造决定生成结果含义的配置；用于安全续跑和审计。
    这个函数把“决定实验结果含义的所有重要设置”集中保存成一个配置字典，主要用于：

    记录这次实验到底是怎样运行的；
    重新运行时复现实验；
    断点续跑时防止新旧实验结果混在一起；
    后期检查结果时提供审计记录。
    """
    return {
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "truthfulqa_sha256": sha256_file(data_path),
        "num_prompts": args.num_prompts,
        "sampling_mode": args.sampling_mode,
        "category": args.category,
        "max_new_tokens": args.max_new_tokens,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "temperatures": args.temperatures,
        "top_ps": args.top_ps,
        "top_ks": args.top_ks,
        "decoding_modes": ["pure_top_p", "pure_top_k", "top_p_x_top_k"],
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "eos_boost_start": args.eos_boost_start,
        "eos_boost_factor": args.eos_boost_factor,
        "limit_settings": args.limit_settings,
    }


def experiment_id(config: dict[str, object]) -> str:
    '''
    这个函数给完整实验配置生成一个简短、稳定的实验编号，也就是configuration fingerprint。
    '''
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def runtime_metadata(device: str) -> dict[str, object]:
    """记录软件、硬件和确定性相关运行环境。"""
    metadata: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": device,
    }
    if device == "cuda":
        metadata["device_name"] = torch.cuda.get_device_name(0)
    elif device == "mps":
        metadata["device_name"] = "Apple Metal Performance Shaders"
    else:
        metadata["device_name"] = platform.processor() or platform.machine()
    return metadata


def split_answers(value: object) -> list[str]:
    """把 TruthfulQA 中以分号分隔的参考答案拆成列表。

    参数：
        value: 包含一个或多个答案的 CSV 单元格。

    返回：
        去除首尾空白后的非空参考答案列表。
    """
    return [part.strip() for part in str(value).split(";") if part.strip()]


def normalize_words(text: str) -> list[str]:
    """把英文文本标准化为小写字母数字 token。

    参数：
        text: 模型生成答案或参考答案。

    返回：
        用于精确匹配和 token-F1 诊断的单词列表。

    说明：
        标准化会忽略标点，后续词袋 F1 也不考虑词序。
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """计算一个生成答案与一个参考答案之间的词袋 token-F1。
    Token-F1衡量生成答案和参考答案之间有多少单词重合，同时考虑precision和recall。
   注意：每次只与一个标准答案比较

    参数：
        prediction: 模型实际生成的答案。
        reference: 一条 TruthfulQA 参考答案。

    返回：
        取值范围为 [0, 1] 的 token 重合 F1。它是词汇相似度诊断，
        不是人工事实正确性判断。
    """
    pred = normalize_words(prediction)
    ref = normalize_words(reference)
    if not pred or not ref:
        return float(pred == ref)
    pred_counts = pd.Series(pred).value_counts()
    ref_counts = pd.Series(ref).value_counts()
    overlap = sum(min(pred_counts.get(token, 0), ref_counts.get(token, 0)) for token in pred_counts.index)
    #overlap直接看生成答案与参考答案有多少重合的词
    precision = overlap / len(pred) #生成答案中的token，有多大比例也出现在参考答案中？
    recall = overlap / len(ref) #参考答案中的token，有多大比例被生成答案覆盖？
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def reference_metrics(generated: str, correct_value: object, incorrect_value: object) -> dict[str, float | int]:
    """把一次生成分别与正确、错误参考答案集合比较。
     参考的是
    question_row["Correct Answers"]
    question_row["Incorrect Answers"]
    故而有一系列正确的答案与错误答案

    参数：
        generated: 模型实际生成的答案。
        correct_value: 用分号分隔的正确参考答案。
        incorrect_value: 用分号分隔的错误参考答案。

    返回：
        正确答案精确匹配标记、最大正确参考 F1、最大错误参考 F1、两者差值，
        以及生成答案是否更接近正确参考的二元代理指标。

    说明：
        margin 大于0只表示词汇上更接近某条正确参考，不能直接称为经过
        人工核验的事实正确率。
    """
    correct = split_answers(correct_value)
    incorrect = split_answers(incorrect_value)
    normalized = " ".join(normalize_words(generated))
    exact = int(any(normalized == " ".join(normalize_words(answer)) for answer in correct))
    best_correct = max((token_f1(generated, answer) for answer in correct), default=0.0)
    best_incorrect = max((token_f1(generated, answer) for answer in incorrect), default=0.0)
    return {
        "exact_match_correct": exact, #是否与某条正确参考答案完全匹配
        "max_correct_token_f1": best_correct, #与最接近的正确答案之间的token-F1
        "max_incorrect_token_f1": best_incorrect, #与最接近的错误答案之间的token-F1
        "token_f1_margin": best_correct - best_incorrect, #正确相似度减去错误相似度
        "reference_accuracy": int(best_correct > best_incorrect), #是否在词汇上更接近正确答案
    }


def distinct_n(words: list[str], n: int) -> float:
    """计算单个答案内部“唯一 n-gram 数/全部 n-gram 数”。

    参数：
        words: 按空白切分后的生成答案。
        n: n-gram 阶数。

    返回：
        取值范围为 [0, 1] 的 distinct-n；答案过短时返回0。

    说明：
        该指标衡量一个答案内部的重复，不衡量100次答案之间的多样性。
    """
    if len(words) < n:
        return 0.0
    ngrams = list(zip(*(words[index:] for index in range(n))))
    return len(set(ngrams)) / len(ngrams)


def trigram_repetition_rate(words: list[str]) -> float:
    """计算1减去唯一 trigram 比例，衡量答案内部三元组重复。

    参数：
        words: 按空白切分后的生成答案。

    返回：
        取值范围为 [0, 1] 的 trigram 重复率，越大表示重复越严重。
    """
    if len(words) < 3:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    return 1.0 - len(set(trigrams)) / len(trigrams)


def build_prompt(question: str) -> str:
    """把 TruthfulQA 问题格式化为因果语言模型续写提示。

    参数：
        question: 原始 TruthfulQA 问题。

    返回：
        以 ``Answer:`` 结尾的问题-答案提示文本。
    """
    return f"Question: {question}\nAnswer:"


def load_completed_keys(
    output_path: Path, resume: bool, current_experiment_id: str
) -> set[tuple[int, float, float, int, int]]:
    """读取已完成实验键，使中断后的任务可以安全续跑。

    参数：
        output_path: 已存在的逐次生成结果 CSV。
        resume: 是否保留并跳过已有记录。

    返回：
        由问题、temperature、top-p、top-k 和重复编号组成的已完成键集合。
    """
    if not resume or not output_path.exists():
        return set()
    key_columns = ["prompt_id", "temperature", "top_p", "top_k", "repetition_id"]
    existing = pd.read_csv(output_path)
    if "experiment_id" not in existing.columns:
        raise ValueError(
            f"Cannot safely resume legacy output {output_path}: it has no experiment_id. "
            "Use --no-resume with a new output path."
        )
    ids = set(existing["experiment_id"].dropna().astype(str))
    if ids != {current_experiment_id}:
        raise ValueError(
            f"Resume configuration mismatch for {output_path}: found experiment_id={sorted(ids)}, "
            f"current={current_experiment_id}. Use a new output path or --no-resume."
        )
    duplicate_count = int(existing.duplicated(key_columns).sum())
    if duplicate_count:
        raise ValueError(f"Existing output contains {duplicate_count} duplicate experiment keys")
    # 显式重排列顺序，避免 CSV 列顺序变化导致实验键字段错位。
    return set(existing[key_columns].itertuples(index=False, name=None))


def flush_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    """把缓冲区结果追加到 CSV，并清空内存缓冲区。

    参数：
        rows: 尚未写盘的生成记录。
        output_path: 目标 CSV 路径。
    """
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
    rows.clear()


def main() -> None:
    """运行全部问题、解码设置和重复实验，并定期保存断点结果。"""
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_path = root / args.output
    sample_path = root / args.sample_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and output_path.exists():
        output_path.unlink()

    data_path = root / args.data_path
    questions = load_questions(
        data_path,
        args.num_prompts,
        args.seed,
        sampling_mode=args.sampling_mode,
        category=args.category,
    )
    selected_category = (
        str(questions["Category"].iloc[0]) if args.sampling_mode == "category_all" else None
    )
    questions["sampling_mode"] = args.sampling_mode
    questions["selected_category"] = selected_category
    questions.to_csv(sample_path, index=False)
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be at least 1.0")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size cannot be negative")
    if args.eos_boost_start < 0:
        raise ValueError("--eos-boost-start cannot be negative")
    if args.eos_boost_start > 0 and args.eos_boost_factor <= 1.0:
        raise ValueError("Enabled EOS boost requires --eos-boost-factor > 1.0")
    settings = build_decoding_settings(args)
    if args.limit_settings:
        settings = settings[: args.limit_settings]

    device = choose_device(args.device)
    torch.use_deterministic_algorithms(args.deterministic_algorithms)
    config = experiment_config(args, data_path)
    current_experiment_id = experiment_id(config)
    completed = load_completed_keys(output_path, args.resume, current_experiment_id)
    planned = len(questions) * len(settings) * args.repetitions
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, revision=args.model_revision
    ).to(device).eval()
    resolved_model_commit = getattr(model.config, "_commit_hash", None)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "experiment_id": current_experiment_id,
                "experiment_config": config,
                "resolved_model_commit": resolved_model_commit,
                "runtime": runtime_metadata(device),
                "deterministic_algorithms": args.deterministic_algorithms,
                "timing_scope": "model.generate only; excludes tokenization and text decoding",
                "cuda_synchronized": device == "cuda",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.warmup_generations > 0 and not questions.empty:
        warmup = tokenizer(build_prompt(questions.iloc[0]["Question"]), return_tensors="pt").to(device)
        with torch.inference_mode():
            for _ in range(args.warmup_generations):
                model.generate(
                    **warmup,
                    do_sample=False,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        if device == "cuda":
            torch.cuda.synchronize()

    rows: list[dict[str, object]] = []
    progress = tqdm(total=planned, initial=len(completed), desc="Generations")
    for _, question_row in questions.iterrows():
        prompt = build_prompt(question_row["Question"])
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_tokens = int(encoded.input_ids.shape[1])
        prompt_settings = settings.copy()
        np.random.default_rng(stable_seed(args.seed, "setting-order", question_row["prompt_id"])).shuffle(
            prompt_settings
        )
        for decoding_mode, temperature, top_p, top_k in prompt_settings:
            for repetition_id in range(args.repetitions):
                key = (int(question_row["prompt_id"]), temperature, top_p, top_k, repetition_id)
                if key in completed:
                    continue
                local_seed = stable_seed(args.seed, *key)
                torch.manual_seed(local_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(local_seed)

                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        exponential_decay_length_penalty=(
                            args.eos_boost_start,
                            args.eos_boost_factor,
                        ) if args.eos_boost_start > 0 else None,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        forced_eos_token_id=tokenizer.eos_token_id,
                    )
                if device == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                generated_ids = output[0, prompt_tokens:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                words = generated_text.split()
                new_tokens = int(generated_ids.shape[0])
                ended_with_eos = bool(
                    new_tokens > 0 and int(generated_ids[-1]) == tokenizer.eos_token_id
                )
                stopped_early_on_eos = bool(
                    ended_with_eos and new_tokens < args.max_new_tokens
                )
                metrics = reference_metrics(
                    generated_text,
                    question_row["Correct Answers"],
                    question_row["Incorrect Answers"],
                )
                rows.append(
                    {
                        "experiment_id": current_experiment_id,
                        "model_name": args.model_name,
                        "requested_model_revision": args.model_revision,
                        "resolved_model_commit": resolved_model_commit,
                        "main_seed": args.seed,
                        "max_new_tokens": args.max_new_tokens,
                        "repetition_penalty": args.repetition_penalty,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        "eos_boost_start": args.eos_boost_start,
                        "eos_boost_factor": args.eos_boost_factor,
                        "prompt_id": key[0],
                        "source_row_id": int(question_row["source_row_id"]),
                        "repetition_id": repetition_id,
                        "random_seed": local_seed,
                        "type": question_row["Type"],
                        "category": question_row["Category"],
                        "question": question_row["Question"],
                        "best_answer": question_row["Best Answer"],
                        "best_incorrect_answer": question_row["Best Incorrect Answer"],
                        "correct_answers": question_row["Correct Answers"],
                        "incorrect_answers": question_row["Incorrect Answers"],
                        "source": question_row["Source"],
                        "sampling_mode": args.sampling_mode,
                        "selected_category": selected_category,
                        "decoding_mode": decoding_mode,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "word_length": len(words),
                        "distinct_1": distinct_n(words, 1),
                        "distinct_2": distinct_n(words, 2),
                        "trigram_repetition_rate": trigram_repetition_rate(words),
                        **metrics,
                        "model_generation_latency_seconds": latency,
                        "latency_seconds": latency,
                        "new_tokens": new_tokens,
                        "ended_with_eos": ended_with_eos,
                        "stopped_early_on_eos": stopped_early_on_eos,
                        "tokens_per_second": new_tokens / latency if latency > 0 else np.nan,
                        "generated_text": generated_text,
                    }
                )
                progress.update(1)
                if len(rows) >= args.checkpoint_every:
                    flush_rows(rows, output_path)
    flush_rows(rows, output_path)
    progress.close()
    print(f"Saved/resumed {planned:,} planned generations in {output_path}")


if __name__ == "__main__":
    main()
