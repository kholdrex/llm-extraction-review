# Replication protocol

This protocol becomes binding with the commit that adds it. From then on it is not edited; changes go into a dated
addendum and are exploratory. It is committed, with the manifest, the code named below and the gate result, and
pushed to the public repository before any development, test or format-arm request is sent.

## 1. Purpose

Test whether the main result of the stored-output study (`PROTOCOL.md`, `ADDENDUM.md`) holds on new data: for small
models run locally, a token score from one greedy call ranks wrong intent-and-slot records better than disagreement
among four sampled records. Secondary aims: estimate the gain in errors caught at a 30% review budget, test whether
the effect of text around the JSON record on token scores depends on the `JSON:` labels of the prompt's worked
examples, and describe how a `p_min` threshold chosen on development data transfers to test data.

## 2. Data

### 2.1. Sources

MASSIVE 1.1, locale en-US, as in the original study (archive checked by SHA-256 in `xreview.data`).

SNIPS: the seven-intent benchmark of June 2017 released by Snips (Coucke et al., 2018) in `github.com/sonos/nlu-benchmark`
under CC0 1.0, with a request that publications using it cite Coucke et al. (2018); used here in the BIO-tagged
train/valid/test split of Goo et al. (2018), `github.com/MiuLab/SlotGated-SLU`, commit
`941bc27b6312d110345bedc036c6b533f4b217a7`, files `data/snips/{train,valid,test}/{seq.in,seq.out,label}` with 13,084,
700 and 700 lines. The SlotGated-SLU repository states no licence of its own. The files are not redistributed:
`xreview.snips` downloads them at the pinned commit and checks the SHA-256 of each (`xreview.snips.SHA256`, also in
the manifest). The utterances are the space-joined tokens of `seq.in` (lower-cased in the source); slots are obtained
from the BIO tags by joining the words of each B-/I- run, an I- tag that does not continue a slot of its type
starting a new one (`xreview.snips.slots_from_bio`). Utterance ids are the split and the 0-based line number
(`test-17`). Publications of this replication cite Coucke et al. (2018) and Goo et al. (2018).

### 2.2. Sets

Normalised text means lower-cased with whitespace collapsed (`xreview.validate.normalize`). The sets are drawn in this
order, each from a pool ordered by numeric id (SNIPS: by line) with `random.Random(2027).sample`, the selected ids
then sorted the same way:

| Set | Items | Pool |
|---|---|---|
| Fresh MASSIVE test | 500 | MASSIVE test split without the 1,000 test and 100 development ids of `data/sample.json`, and without any request whose normalised text occurs in the MASSIVE training split, among those 1,100 October requests or among the 20 gate requests: 1,958 |
| MASSIVE development | 100 | MASSIVE development split without the October development ids and the gate ids, and without any request whose normalised text occurs in the training split, among the October requests, among the gate requests or in the fresh MASSIVE test set: 1,898 |
| SNIPS test | 700 | all test utterances, in file order |
| SNIPS development | 100 | SNIPS validation split without any utterance whose normalised text occurs in the SNIPS training split or the SNIPS test set: 684 |
| Format arm | 200 | the fresh MASSIVE test ids in numeric order, shuffled with `random.Random(2028).shuffle`; the first 200, sorted by numeric id |

The fresh MASSIVE test and development sets are text-disjoint from each other, from the October samples, from the
gate requests and from the training split. The SNIPS test set is the full published set: 26 of its utterances have a
normalised text that also occurs in the SNIPS training split (listed in the manifest), and two of them, `test-296`
and `test-395`, share one text. The primary contrast is repeated without the 26 as a sensitivity analysis.

`python -m xreview.replication manifest` writes the sets, the pool sizes, the gate ids, the fixed examples, the
SNIPS duplicates and the SNIPS source to `data/replication_manifest.json`, SHA-256
`5195cec4c044ec915af49707bb26868472f2163b2ac8d15c03bec7aee513322a` (`xreview.replication.MANIFEST_SHA256`);
regenerating it gives the same bytes. The runner and the analysis refuse a manifest with another hash.

### 2.3. Prompts

The ontology (intents, slot types, slot types observed with each intent) of each data set comes from its own
training split, and retrieval of the eight most similar examples uses that training split. Fixed examples: one
training utterance with at least one slot per scenario, drawn with seed 2026 (`xreview.data.examples`); for MASSIVE
these are the 18 examples of the original study, for SNIPS, where the intent serves as the scenario, `train-1564`
(AddToPlaylist), `train-4629` (BookRestaurant), `train-7159` (GetWeather), `train-6974` (PlayMusic), `train-9186`
(RateBook), `train-12504` (SearchCreativeWork) and `train-1474` (SearchScreeningEvent).

Labelled arm: the prompt of the original study (`xreview.prompts`); for MASSIVE the system message is identical to
`data/system_prompt.txt`. Bare arm: the same messages with the label `JSON: ` removed from the line of every worked
example, nothing else changed (`xreview.replication.messages`).

## 3. Models and environment

Primary models (development, test, and the format arm): `qwen2.5:7b-instruct` and `ibm/granite4.1:3b` (IBM Granite
4.1, 3.4B parameters, Q4_K_M, Apache 2.0, released April 2026).

Format-only models (format arm, both prompt arms): `qwen2.5:3b-instruct` and `gemma3:4b`, the two models that wrapped
the record in the original study (517 and 1,000 of 1,000 greedy replies with text outside the JSON span, against 0 for
Qwen2.5-7B). They are added because the primary models wrote bare JSON in all available replies (Qwen2.5-7B 1,000 of
1,000 in October; Granite 20 of 20 gate replies and the 12 greedy replies of a pipeline check on training utterances),
so their format contrasts are expected to be not informative.

Full digests, checked against the server before every invocation (`xreview.replication.DIGESTS`):
`qwen2.5:7b-instruct` `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, `ibm/granite4.1:3b`
`6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb`, `qwen2.5:3b-instruct`
`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`, `gemma3:4b`
`a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`. Ollama 0.34.3 on an Apple M1 Pro with 16 GB, as
in the original study. Context 4,096 tokens, at most 256 output tokens, `keep_alive` 30 min; the request is that of
`xreview.llm.chat`.

Per test item and model: call 0 is greedy (temperature 0, seed 0) with log-probabilities; calls 1–4 are samples at
temperature 0.7 with seeds 1–4. Development and format-arm items: call 0 only.

## 4. Gate

Greedy calls with the labelled prompt on the 20 MASSIVE development requests `251, 657, 1197, 1225, 1241, 1327,
1430, 1525, 1545, 1771, 1934, 2118, 2246, 2421, 2846, 2963, 3194, 3203, 3248, 3296` (the gate requests; excluded from
every set of Section 2). A model passes if every reply has a non-empty list of tokens with log-probabilities that
concatenate to the reply text and are all finite, and at least 18 of 20 replies are well-formed.

`ibm/granite4.1:3b` passed on 2026-11-24: 20 of 20 replies with aligned, finite, non-empty log-probabilities, 20 of 20
well-formed, none with text outside the record. The replies are stored in `results/replication/gate.jsonl` with the
server's version and digest in `results/replication/gate.meta.jsonl`, one row per call with the fields `dataset`,
`arm`, `model`, `id`, `call`, `status`, the reply (`text`, `tokens`, token counts, `seconds`) and `well_formed`. These
two files predate the ledger format of Section 5, are kept unchanged and are part of the freeze.
`python -m xreview.replication_analysis gate` recomputes the verdict from them, checking that they hold the 20 gate
requests in order, all from Granite with call 0, and one metadata row with the version and digest of Section 3.

## 5. Running

The checks below protect an honest run from accidents (an interrupted write, a resumed run in the wrong state, a
malformed server reply, a forgotten commit) and leave a record that can be compared with the public git history;
they do not prevent deliberate tampering by whoever runs the code.

### 5.1. Order

```bash
# 1. Freeze: commit this protocol, the manifest, the code and the gate files; push.
python -m xreview.replication run massive_dev
python -m xreview.replication run snips_dev
python -m xreview.replication_analysis thresholds
# 2. Second commitment: commit results/replication/dev.jsonl, results/replication/dev.jsonl.meta.jsonl and
#    results/replication/thresholds.json; push.
python -m xreview.replication run massive_test
python -m xreview.replication run snips_test
python -m xreview.replication run format_arm --arm labelled
python -m xreview.replication run format_arm --arm bare
python -m xreview.replication_analysis analyze --out results/replication/analysis.json
```

Each phase writes to one fixed ledger and its metadata file: `results/replication/dev.jsonl`, `test.jsonl` and
`format.jsonl`, each with `<ledger>.meta.jsonl`. The models of each run are fixed by the set and the arm
(`xreview.replication.roster`); within a run they are processed one after the other, items in manifest order. The
`thresholds` command writes `results/replication/thresholds.json`: the four thresholds, the manifest hash and the
SHA-256 of the development ledger and of its metadata file. It refuses to write unless the development ledger is fully
admitted (Section 5.4) and each of the four model × data set cells has every development call and a finite
threshold. The code checks that the three files of the second commitment are committed; pushing is what makes the
commitment public, and the code cannot check that.

### 5.2. Checks before an invocation

The runner refuses to start unless: the manifest has the hash of Section 2.2; the protocol, the manifest, the gate
files and the modules of `src/xreview` listed in `xreview.replication.FROZEN_CODE` have exactly the bytes of their
blobs at HEAD (other files, tracked or not, are not checked); every git command it calls succeeds; and the server
reports Ollama 0.34.3 and the full digest of Section 3 for every model of the run. For test and format runs it also
requires the thresholds of Section 5.1 to be usable: the threshold file, the development ledger and its metadata file
have the bytes of their blobs at HEAD, the threshold file names this manifest and lists the SHA-256 of both
development files, these match, the file holds four finite thresholds, and the admitted development replies
reproduce them exactly.

The header of an invocation records its id, phase, set, arm, models, manifest hash, the HEAD commit id, the Ollama
version, the full digests and the start time, and, for test and format runs, the SHA-256 of the threshold file.

### 5.3. Attempt rows, retries and resumption

Every attempt is one row with phase, data set, prompt arm, model, item, call index, invocation id, attempt number and
status. Before a reply can count, its shape is checked: the server's payload must be an object with a message, the
content a string, and, for call 0, the token list non-empty, every entry a pair of a string and a finite number, the
tokens concatenating to the content. `ok` means the reply passed; `invalid` means a reply was received and failed the
check (the reply is kept); `error` means the request failed or raised any exception, including a server payload that
is not an object with a message: the attempt is recorded with the exception message, and the payload itself is not
stored. Token metadata is read only when log-probabilities were requested, that is for call 0; on a sampled call any
token metadata the server adds is ignored. The stop reason reported by the server is stored. A well-formed reply that
extracts the wrong record is `ok`; it is not retried.

A call with an `ok` row is never sent again. A failed attempt is retried at once with the same prompt and seed. A
call with three failed attempts is exhausted: it is not sent again, the rest of that model's cell in that run is
skipped, the other models continue, and the run ends with a non-zero status naming the exhausted calls. Because each
phase has one ledger, the count of attempts cannot restart in another file.

One lock covers a ledger and its metadata file. Under it the runner repairs both files (a complete last row gets its
newline; an incomplete last line is moved, with its byte offset, to `<file>.torn` and the file is cut at the last
newline), checks the ledger as in Section 5.4, and appends the invocation header only before its first call, so an
invocation that makes no call writes nothing. If the ledger holds any header or row that Section 5.4 does not admit,
the runner stops before any call and names the problem; it does not work around it. What to do with such a ledger is
the author's decision, recorded in a dated addendum.

### 5.4. Which rows count

Headers are collected from all supplied ledgers first. A header is admitted when its set and phase match, its arm is
one of the set's, its models are exactly the roster of the set and arm, each with the pinned digest, the Ollama version
and manifest hash are those of the protocol, it records a commit, and, for test and format, it records the SHA-256 of
the committed threshold file. An invocation with any header that is not admitted, or with two different headers, is
invalid in every ledger. A row counts only if its invocation is valid and its phase, data set and arm equal those of
the header and its model is in the header's roster.

For every expected call (phase, data set, arm, model, item, call index) exactly one `ok` row must count. A cell is
incomplete if any expected call is missing, exhausted, has more than one `ok` row, or has an `ok` row after three
failed attempts. `ok` rows for calls the manifest does not expect are reported and ignored, in the development, test
and format results alike; for the labelled format replies of the primary models, which are test greedy replies, the
whole test manifest decides what is expected. Every output is published, including errors, invalid replies, torn
fragments and the gate.

## 6. Outcome, parser and scores

- Parser: the permissive parser of the original study, the text from the first `{` to the last `}`. A reply is
  well-formed when that text parses as a JSON object with exactly the keys `intent` (a string) and `slots` (a list of
  objects with exactly the string keys `type` and `value`); extra keys fail. A reply without such a span is not
  well-formed. Strict parsing of the whole reply is counted per cell and arm but not used for scoring.
- A record is correct when the intent and the multiset of (slot type, normalised value) equal the annotation; a reply
  that is not well-formed is a wrong record. `R` (Section 4 of the original plan; slot values are checked as
  substrings of the normalised request) is a score, not the outcome.
- Scores, with the definitions of the original study and its addendum: `R`, `D` (a sample that is not well-formed
  never agrees, also with a greedy reply that is not well-formed), `L_field` (worst field-string token; 0 when the
  reply has no field string), the field sum, the worst token over the JSON span without the opening token, and the
  worst token over all tokens (every token when the reply has no JSON span). Token scores use the emitted text whether
  or not the reply is well-formed.
- Text outside the JSON span (`W`): the reply contains a non-whitespace character outside the span; for a reply
  without a span, anywhere in the reply.

## 7. Analyses (`xreview.replication_analysis`)

### 7.1. Primary contrast and verdict

For each of the four named cells (SNIPS and fresh MASSIVE × Qwen2.5-7B and Granite 4.1 3B): the paired difference
AUROC(`L_field`) − AUROC(`D`) against greedy-record error, with a 95% interval. A cell is complete when every expected
test call has exactly one eligible `ok` row (Section 5.4) and estimable when it has both wrong and correct records and
a finite difference and interval. A complete, estimable cell replicates when the unrounded lower bound of the interval
is above 0; a lower bound of exactly 0 does not.

Verdict over the four named cells: replicated when at least three replicate, not replicated when at most two do, and
incomplete, without a verdict, when any named cell is missing, incomplete or not estimable; the reason is reported for
each cell. The AUROC contrast is the headline; no secondary or sensitivity result changes it.

Intervals: 2,000 resamples of the items with replacement, drawn as one index vector over the items in manifest order
from a generator freshly seeded with 2026 for each interval and applied to both scores; the interval is the 2.5th and
97.5th percentile (linear interpolation) of the finite resampled differences; resamples with an undefined AUROC are
left out and their number is reported. Intervals are pointwise, without adjustment for multiplicity. Undefined
quantities are written as JSON `null`.

### 7.2. Secondary (estimates)

1. Review yield: errors caught per 1,000 items, and error among accepted records, at 10%, 20% and 30% budgets,
   `L_field` − `D`, with paired intervals; reported as an estimate without a separate success rule. Budget rule:
   `k = round(β·n)` with Python's `round`; items ordered by descending score, ties broken by ascending priority
   `u = numpy.random.default_rng(2026).random(n)` drawn once over the items in manifest order; bootstrap replicates keep
   each item's `u` (`xreview.analysis.accepted_error`, `xreview.reanalysis.caught`).
2. AUROC(field sum) − AUROC(`L_field`) and AUROC(worst over the JSON span without the opening token) − AUROC(`L_field`),
   paired intervals as in 7.1.
3. Format arm, for each of the four format models, on the 200 format items. Labelled replies: the test greedy replies
   for the primary models, the labelled format replies for the format-only models; bare replies: the bare format
   replies. Each arm is scored against its own outcomes. `W` is the share of replies containing a non-whitespace
   character outside the JSON span;
   `G = AUROC(worst over the JSON span without the opening token) − AUROC(worst over all tokens)`. Reported per arm:
   `n`, wrong, wrapped, replies that are not well-formed, strict parses and `G`; contrasts `W_bare − W_labelled`
   (point estimate, with the counts of items wrapped under one arm only) and `G_bare − G_labelled` with a paired
   interval as in 7.1. A model's contrast is informative when its labelled arm has at least one wrapped reply, each
   arm has at least 20 wrong and at least 20 correct records, and the `G` difference is estimable; otherwise it is
   reported as not informative, with the reasons. An informative contrast supports the dependence on the label when
   `W_bare − W_labelled < 0` and the upper bound of `G_bare − G_labelled` is below 0, and does not support it
   otherwise. The dependence is called shared by the two models that wrapped in October only when both are informative
   and both support it; otherwise the two results are reported individually. No conclusion is drawn from a single
   supporting model, and all four results are reported. A format result needs only its own calls, not a complete
   primary cell.
4. Threshold transport (descriptive, no success tolerance): the committed `t` of each primary cell
   (`numpy.quantile(p_min_dev, 0.30)`, linear interpolation, over the 100 development replies, `p_min = exp(−L_field)`,
   1 for a reply without field strings). A test record goes to review when `p_min < t`; `p_min = t` is accepted.
   Reported: realised review share and error among accepted records, against the 30% budget ranking. The analysis
   reads the committed values and the development inputs they name, and exits with an error, writing nothing, if the
   recomputed values differ.
5. Per cell: unanimous items (`D = 0`) and wrong among them; records flagged by `R` and wrong among them; replies that
   are not well-formed; strict parses.

### 7.3. Sensitivity

Each a separate analysis with its own completion and estimability status, none producing another headline: the
primary contrast without replies that are not well-formed, in every cell; for SNIPS, without the 26 training-text
matches.

## 8. Planning

Resampling the October items (2,000 resamples with the October tie priorities) gives these 95% ranges for samples of
500 and 700 items. AUROC(`L_field`) − AUROC(`D`): Qwen2.5-3B 0.109–0.180 and 0.112–0.175, Gemma 3 4B 0.152–0.235 and
0.158–0.228, Qwen2.5-7B 0.150–0.233 and 0.157–0.225. Extra errors caught per 1,000 at 30%: −8 to 36 and −6 to 33,
26 to 72 and 30 to 69, 34 to 82 and 37 to 79. Both sizes resolve AUROC differences of the October size and yield
gains near 50 per 1,000; neither resolves a yield gain near 12. These ranges come from MASSIVE and the October
models; they say nothing about Granite or SNIPS.

Machine time: about 7 hours at the October latencies (test about 5.9 h, development 0.3 h, format arm 0.6 h); the
median latencies of the latest pipeline check on training utterances (three items per cell) imply up to about 13 hours.

## 9. Registration and amendments

The protocol, manifest, code and gate files are committed and pushed before the first development call, and the
development ledger, its metadata file and the threshold file before the first test call; both commit ids are cited in
the paper, and every invocation header records the commit it ran from. The scores, the primary contrast, the rosters,
the item sets and the decision rules are those above. Every output is published. A change after the freeze is
described in a dated addendum and its results are exploratory.
