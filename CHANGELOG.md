# Changes after the analysis plan was fixed

The plan (`PROTOCOL.md`) was written after a 40-request pilot and before the test run on 2026-10-02. The test
outputs were not changed afterwards. Changes to the analysis code:

- JSON parsing takes the text between the first `{` and the last `}`; the pilot showed that models often wrap the
  record in a label or code fence. Made before the test run.
- Additional descriptive analyses, not in the plan: kinds of error (`errors.py`), AUROC of disagreement for one to
  four samples, error rate by the probability of the least certain field, and the policy that reviews records
  flagged by the rules first and then ranks the rest by `L_field` (`R+L` in `summary.json`).
- Ties in the review ranking are broken by the same random permutation for every budget, so that the sets of
  reviewed records are nested as the budget grows.
- Figures redrawn in greyscale.
- The Ollama client raises an error when requested token log-probabilities are missing or do not match the output
  text (no stored record was affected); `xreview run` accepts `--out`.
- The pilot with fixed examples only (`results/pilot_fixed_examples_dev.jsonl`) was produced before retrieval of
  similar examples was added; its outputs are kept for the comparison reported in the paper.
- The majority vote used in the paper is taken over the parsable outputs among the greedy and four sampled records
  (ties: the earliest output); this is now stated in `analysis.majority`.
- Code review before release: the rules-then-field ranking uses ranks instead of a large constant; analysis refuses
  duplicate model/request pairs and missing token log-probabilities instead of silently using fallbacks; the runner
  remembers records written in the same run and ignores a last line cut off by an interruption; models are taken
  from the output file; figure files have descriptive names. Published numbers are unchanged.
- `metrics.py` lost an unused helper after the freeze. The repository has no history from before the freeze, so
  `protocol_freeze.sha256` documents the hashes at that time but the frozen files themselves are not included.
