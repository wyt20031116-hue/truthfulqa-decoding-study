# k=1 forensic audit plan

## Question

Why did the post-hoc `top_k=1` extension have a mean exact unique-answer rate
slightly above its five-repetition minimum of 0.20?

## Evidence already available

- The run used the same 50-question frame as the main experiment.
- The model was loaded in evaluation mode and each generation received a stable
  per-row seed.
- Of 250 prompt-temperature cells, exact-text variation occurred in 20 cells.
- All 20 cells came from four prompts (`prompt_id` 1, 7, 26, and 47), and each
  of these prompts varied at all five temperatures. The other 46 prompts were
  invariant across repetitions.
- Hugging Face Transformers 4.40.2 removes tokens with scores strictly below
  the kth score. At `top_k=1`, exact ties for the maximum score therefore leave
  multiple tokens available to multinomial sampling.

The token-level replay completed on the original A100 BF16 configuration. All
100 sampled replay rows encountered a step with two finite candidates after
the `top_k=1` filter, and the same four prompts again produced multiple texts.
This directly confirms tie-preserving top-k filtering as the source of the
residual variation.

## Direct test

`scripts/audit_k1_determinism.py` will:

1. audit all saved `k=1` rows and list every varying cell and answer;
2. replay the four anomalous prompts with the original seeds and settings;
3. save generated token IDs;
4. record the post-warp support size at every generation step; and
5. compare each prompt with ordinary greedy decoding.

The companion true-greedy control used `do_sample=False` for five calls on each
of the same 50 questions. All 250 calls were identical within prompt at both
the token and text levels. The original sampled `k=1` rows matched the greedy
answer 1,190/1,250 times (95.2%), and every prompt's greedy answer occurred
among its sampled `k=1` variants.

## Reporting rule

Describe the existing arm as "top-k sampling with k=1," not as greedy
decoding. Its residual variation is a documented consequence of exact-score
ties under the tie-preserving Transformers filter, not greedy nondeterminism.
