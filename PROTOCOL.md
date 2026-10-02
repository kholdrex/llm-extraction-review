# Analysis plan

Fixed on 2026-10-02 after a 40-utterance pilot on the development split and before any test-split output was generated.

## Question
Which signal available without a gold annotation best identifies the records, extracted by a small language model,
that should be sent to manual review: (R) rule-based validation, (D) disagreement among sampled extractions, or
(L) token log-probabilities of the greedy extraction?

## Data and models
- MASSIVE 1.1, locale en-US (CC BY 4.0). Ontology (60 intents, 55 slot types, intent–slot compatibility) from the
  training split. 1000 test utterances drawn with seed 2026; 100 development utterances for the pilot only.
- qwen2.5:3b-instruct, gemma3:4b, qwen2.5:7b-instruct (Ollama, Q4_K_M) on one Apple M1 Pro.
- Prompt: ontology, output rules and one fixed example per scenario (system message), plus the 8 nearest training
  utterances by TF-IDF cosine similarity (user message).
- Per utterance: one greedy extraction with token log-probabilities and 4 samples at temperature 0.7 (seeds 1–4).

## Outcome
A record is correct when the intent and the multiset of (slot type, normalised value) equal the gold annotation.
Unparsable output counts as an incorrect record.

## Error scores
- R: 1 if any check fails (no JSON object, wrong structure, unknown intent or slot type, slot type never seen with
  the intent in training data, value absent from the request, duplicate slot), else 0.
- D: share of the 4 samples whose record differs from the greedy record.
- L-seq: negative log-probability of the whole greedy output.
- L-field: negative of the lowest token log-probability among tokens of the intent, slot-type and slot-value strings.
- C: logistic regression on (R, D, L-field), cross-fitted with 5 folds within each model.

## Analyses
1. Primary: AUROC of each score for detecting incorrect records, per model; paired bootstrap (2000 resamples of
   utterances) for the differences L-field − D and L-field − R.
2. Error rate among automatically accepted records when the 10%, 20% and 30% highest-scoring records are reviewed
   (ties broken at random with seed 2026); random review as reference.
3. Cost of each signal: model calls and measured generation time per record.
4. Secondary: accuracy of the majority-vote record over greedy + 4 samples versus the greedy record.
