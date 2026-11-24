# Addendum to the analysis plan: exploratory reanalysis

Written on 2026-11-24, after the test outputs had been generated (2026-10-02), analysed (2026-10-03) and inspected.
It does not change `PROTOCOL.md`, which remains the plan fixed before the test run, or the October summary
`results/summary.json`, which contains the planned analyses together with the descriptive and post-hoc analyses
listed in `CHANGELOG.md`.

## Questions

1. Does the advantage of the field score over disagreement remain against simple controls that need no token
   probabilities and against finer measures of disagreement?
2. Which token positions and which aggregation of token log-probabilities determine the ranking, and how does the
   text that some models write around the JSON record affect it?
3. How does a fitted combination of scores depend on its parameterisation and on the features it is given?

The analyses use only the stored outputs in `results/raw_test.jsonl`; no model was called again. All of them were
chosen after the test results were known and are exploratory. `xreview reanalyze` writes them to
`results/reanalysis.json` and draws `figures/token_sets.png` and `figures/errors_caught.png`.

## Definitions

- JSON span: the characters from the first `{` to the last `}` of the reply, the text the parser reads. Text
  outside the JSON span: any non-whitespace character outside it. Strict parsing: `json.loads` of the whole reply
  gives an object. A record is well-formed when the JSON span parses and the object has the fields `intent` (a string)
  and `slots` (a list of objects with string `type` and `value`).
- Token sets of the greedy reply (`tokens.masks`), a token belonging to a set when its characters overlap the stated
  characters: `all`, every token; `json`, the JSON span (every token if the reply has no span); `json-open`, the JSON
  span without the one token that contains the opening brace; `field-span`, from the first character of the first
  intent, slot-type or slot-value string to the last character of the last one, punctuation between them included;
  `field`, the intent, slot-type and slot-value strings (`L-field` of the October analysis).
- Aggregations of negative log-probability over a token set: `worst` (largest), `mean`, `sum`; an empty set scores 0,
  as `L-field` does. `field worst` equals `L-field`, `all sum` equals `L-seq`. Token scores use the emitted text
  whether or not the reply parses.
- Position of the least probable token of the reply: before the JSON span, in the token containing the opening brace,
  in the rest of the span, or after the span; and whether the token containing the opening brace is the least
  probable token inside the span.
- Scores without token probabilities: output tokens, the number of returned token/log-probability pairs (the API's
  output-token count is larger by one on every greedy reply); predicted slots (0 for an unparsable reply); request
  words; nearest similarity, 1 minus the largest TF-IDF cosine similarity of the request to a training request;
  retrieved intents and retrieved slot types, the share of the eight retrieved examples whose intent, or multiset of
  slot types, differs from the record (1 for an unparsable reply).
- Disagreement: `D` as planned; `D jaccard`, mean over the four samples of 1 − |A ∩ B| / |A ∪ B| for the multisets of
  intent and (slot type, normalised value), 1 when either record is unparsable, computed exactly as a fraction and
  converted to a float once, so that equal values tie; `D entropy`, entropy of the distinct records among the four
  samples, unparsable samples forming one class. A request is unanimous when `D` = 0.
- Combined scores: logistic regression on the features in `reanalysis.COMBINED` (`C ranks`: R, D, L-field;
  `no probabilities`; `one call`; `one call + samples`). Within each of five folds, every feature is replaced by its
  mid-rank empirical distribution in the training folds and standardised with training-fold moments; the folds are the
  ones used for `C` in the October analysis (seed 2026) and are the same for every feature set. The spread of AUROC
  over 20 fold assignments is reported. Intervals for combined scores are conditional on the out-of-fold predictions
  and do not include the variability of refitting.
- Review budget: as in the October analysis, the share of records with the highest scores, ties ordered by the same
  seeded permutation; errors caught are reported per 1,000 requests, and the error among accepted records with a
  bootstrap interval that keeps each request's tie priority fixed. The rules-first policy (`R+L`) ranks flagged
  records by `L-field` among themselves: when more records are flagged than the budget allows (Qwen2.5-3B at 10%,
  123 flagged, 100 reviewed), the flagged records with the highest `L-field` are reviewed.

## Main exploratory contrasts

Paired AUROC differences (`reanalysis.CONTRASTS`): field sum − field worst; json-open worst − field worst;
field worst − all worst; one call − field worst; one call + samples − one call; field worst − retrieved slot types;
field worst − no probabilities; one call − no probabilities. Paired differences in errors caught and in error among
accepted records at 10, 20 and 30% (`reanalysis.BUDGET_CONTRASTS`): L-field − D, field sum − D, one call − D.
Intervals are percentile bootstrap intervals (2,000 resamples of requests, seed 2026), pointwise, without adjustment
for multiplicity. Everything else in `reanalysis.json` is descriptive: single-score AUROC for every token set and
aggregation, budgets for the other scores, AUROC within strata of predicted slot count and of output format, and
AUROC without the eight test requests whose normalised text also occurs in the training split.
