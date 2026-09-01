# Semantic diversity results

## Method

- Embedded all 56,250 generated answers with `BAAI/bge-small-en-v1.5` (384 dimensions).
- For each of the 2,250 prompt-setting cells, computed the mean cosine distance over all 300 pairs among the 25 repeated answers.
- This metric is judge-independent. It measures between-answer semantic variation rather than correctness or within-answer lexical repetition.
- Interaction models match the main factorial analysis: categorical temperature and sampling parameter, prompt fixed effects, CR1 covariance clustered by the 50 prompts, and joint F tests with 49 denominator degrees of freedom.

## Main findings

1. Temperature strongly increases semantic diversity. Marginal mean cosine distance rose from 0.0127 at temperature 0.1 to 0.2030 at 1.5 in pure top-p, and from 0.0210 to 0.1973 in pure top-k.
2. Top-p amplifies the temperature effect. The temperature-by-top-p interaction was significant, F(12,49)=7.76, BH-adjusted p=1.63e-7, partial R-squared=0.0363.
3. Relative to p=0.6, the semantic-diversity increase from temperature 0.1 to 1.5 was amplified by 0.0401 at p=0.8, 0.0651 at p=0.9, and 0.0742 at p=0.95; all 95% intervals excluded zero.
4. Top-k differences were much smaller. Marginal semantic distance ranged only from 0.1043 at k=10 to 0.1084 at k=30. Its interaction was statistically detectable, F(16,49)=2.07, BH-adjusted p=0.0319, but had small partial R-squared=0.0086.
5. Semantic distance agreed strongly with the lexical between-generation metrics: Spearman rho=0.919 with pairwise token Jaccard distance, 0.888 with corpus Distinct-1, and 0.835 with unique-answer rate.

## Interpretation

The temperature result is not merely a change in wording: higher temperature produces answers that differ in semantic content. Top-p meaningfully controls how strongly temperature expands this semantic range. Top-k has, at most, a small modifying role within the tested range of 10 to 30.

These results do not establish that higher semantic diversity is better. They should be combined with latency, repetition, and human-audited quality evidence when selecting settings.
