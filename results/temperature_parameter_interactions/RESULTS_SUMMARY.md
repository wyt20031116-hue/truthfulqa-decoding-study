# Temperature-by-parameter interaction analysis

## Method

The analysis uses the 2,250 prompt-setting means. Each observation first
averages 25 repeated generations, so repetitions are not treated as independent.
Separate models were fit for pure top-p and pure top-k:

`outcome ~ prompt fixed effects + categorical temperature * categorical parameter`

Joint interaction tests use CR1 standard errors clustered by the 50 prompts and
an F reference distribution with 49 denominator degrees of freedom. P-values
are Benjamini-Hochberg adjusted across 13 outcomes within each interaction
family. Partial R-squared measures the incremental fit from the interaction.

## Main result

Temperature and top-p interact strongly for between-generation diversity.

- Unique-answer rate: joint F(12,49) = 18.31, BH-adjusted p < 0.000001,
  partial R-squared = 0.0415.
- Pairwise token Jaccard distance: F(12,49) = 5.77, adjusted p = 0.000011,
  partial R-squared = 0.0213.
- Corpus Distinct-1: adjusted p < 0.000001, partial R-squared = 0.0953.

Relative to top-p = 0.6, the diversity effect of moving from temperature 0.1
to 1.5 was amplified as follows:

| top-p | Unique-answer amplification | Pairwise-distance amplification |
|---:|---:|---:|
| 0.8 | 0.095 (95% CI 0.049--0.141) | 0.069 (0.019--0.119) |
| 0.9 | 0.140 (0.083--0.197) | 0.123 (0.065--0.182) |
| 0.95 | 0.146 (0.073--0.218) | 0.127 (0.058--0.197) |

Thus, top-p is not merely an additive diversity control: its effect becomes
larger at high temperature.

## Top-k result

The top-k interaction was much smaller. After multiplicity adjustment:

- Unique-answer rate: adjusted p = 0.102, partial R-squared = 0.0024.
- Pairwise token distance: adjusted p = 0.060, partial R-squared = 0.0046.

Some individual extreme-temperature contrasts were positive, but the joint
evidence does not support a stable top-k interaction over the full factorial
surface. Corpus Distinct-1 was statistically significant, but its interaction
partial R-squared was only 0.0117 and the two primary between-generation
metrics did not pass the family-wise analysis rule.

## Quality outcomes

Neither correctness interaction passed the adjusted test:

- temperature x top-p correctness: adjusted p = 0.236;
- temperature x top-k correctness: adjusted p = 0.102.

The top-p informativeness interaction was statistically detectable, but
informativeness is a provisional Qwen3-NF4 measure whose direction changed in
the human-audit sensitivity analysis. It should not be interpreted as a robust
quality interaction without additional human labels.

## Interpretation

The formal interaction analysis strengthens the existing visual conclusion:
temperature is the primary diversity control, top-p amplifies temperature's
effect, and top-k contributes little within 10--30. The result is strongest for
direct generation metrics and does not depend on the provisional judge.

`corpus_distinct_2` was excluded because it is undefined in 283 prompt-settings
with too few bigrams. Exact uniqueness, pairwise token distance, and corpus
Distinct-1 remain available and agree on the top-p interaction.
