# Cochrane text simplification baselines

This project implements reproducible baselines for CLEF SimpleText Task 1:
automatically rewrite complex biomedical text into language intended for
patients and other non-specialists. There are two separate sequence-to-sequence
experiments. Both use `facebook/bart-base` as the pretrained model, but have
separate training data, checkpoints, and evaluation runs.

## Experiments and data

| Experiment | Input → target | CSV files used | Train / validation / test |
|---|---|---|---:|
| Sentence simplification | One complex sentence → its aligned lay sentence(s) | `data/cochraneauto_sents_{train,val,test}.csv` | 11,510 / 1,697 / 1,512 |
| Document simplification | Complex abstract → lay summary | `data/cochrane_docs_{train,val,test}.csv` | 3,967 / 500 / 502 |

The sentence-level CSV stores the target as a Python-list string in `simple`.
The loader parses that value and joins multiple target sentences with spaces,
so a split operation becomes one generated target containing both sentences.
An empty list becomes an empty target and is retained as a deletion example.
The sentence operation label (such as `rephrase`, `split`, or `delete`) is not
given to the model; it learns the input-to-target mapping directly.

The document experiment uses the direct complex/lay abstract pairs in
`cochrane_docs_*.csv`. The repository also contains other corpus files,
including `cochraneauto_docs_*.csv` and paragraph alignments; they are not
used by these two baseline commands.

## Why these baseline choices?

### Model: `facebook/bart-base`

BART is an encoder-decoder Transformer pretrained with a denoising objective:
it learns to reconstruct text that has been deliberately corrupted. This is a
natural starting point for simplification, where the encoder reads complex
text and the decoder generates a rewritten version. Fine-tuning a pretrained
model is also a more meaningful baseline than training a language model from
scratch on these comparatively small aligned datasets. BART has been widely
used for text generation and is directly relevant to document simplification
work.

The **base** checkpoint is a resource-conscious choice: it is smaller and
cheaper to fine-tune than BART-large, making both experiments more feasible on
typical research hardware. It is not claimed to be the strongest available
model. A larger checkpoint or a biomedical-domain model could be a follow-up
comparison, but changing the model family would make this initial baseline
less straightforward to reproduce.

Sentence and document tasks get separate fine-tuned checkpoints because their
inputs, target lengths, and context requirements differ. Sentence training
uses the aligned sentence examples, so the model can learn local rewriting,
compression, and splitting. Document training sees each abstract as one input
and the corresponding lay summary as one target, giving it access to the
document context. This is a direct baseline, not an explicit discourse planner:
it does not predict copy/rephrase/delete operations, enforce paragraph
structure, or guarantee consistency across generated sentences. The
document-level source limit is 1,024 tokens, so longer abstracts are truncated.

The **copy baseline** returns the input unchanged. It is deliberately simple:
it establishes what the metrics score when no simplification is attempted
and checks that the evaluation path works. It is not expected to be a strong
lay-text system.

### Metrics: complementary views, not a single quality score

Text simplification has competing goals: preserve the important medical
meaning, make the wording accessible, and avoid unsupported changes. No one
automatic metric captures all three. The selected metrics provide different,
imperfect views of similarity to the available reference:

| Metric | Why include it? | What it does not establish |
|---|---|---|
| **SARI** | Designed for text simplification; looks at n-grams the system adds, keeps, and deletes relative to the source and reference. This makes it more relevant to simplification edits than a pure overlap score (Xu et al., 2016). | It does not verify medical facts, clinical safety, readability, or whether a deletion removed a critical qualification. |
| **BLEU** | A standard, reproducible corpus-level lexical-overlap measure. It makes it easier to compare with text-generation baselines and penalizes outputs that diverge substantially from the reference (Papineni et al., 2002). | Many valid rewrites use different words from the reference, so low overlap need not mean poor simplification; high overlap does not prove the meaning is preserved. |
| **BERTScore F1** | Uses contextual token embeddings (`roberta-large`) to capture semantic similarity when equivalent wording differs, complementing surface n-gram matching (Zhang et al., 2020). | General-purpose embeddings are not a medical fact checker and may miss changed numbers, negation, uncertainty, or patient-relevant details. |
| **LENS** (optional) | A learned metric developed specifically for text simplification, providing a task-focused complement to BLEU and BERTScore (Maddela et al., 2023). | A learned score is still an estimate, depends on its checkpoint and implementation, and does not replace expert review or establish clinical safety. |

For this reason, scores should be reported together rather than collapsed into
one ranking without explanation. Inspect example predictions, especially
negation, dosage, numbers, uncertainty, adverse effects, and deleted content.
Automatic metrics compare with the reference wording; they are not a substitute
for expert or patient-centered evaluation.

**SARI implementation note:** the current code uses a small, local
single-reference variant for a dependency-light, transparent baseline. It
lowercases and whitespace-tokenizes, uses *sets* of 1–4-grams (not
frequency-weighted n-gram counts), computes add/keep F1 and delete precision,
then averages over examples. SARI implementations differ in tokenization,
reference handling, and aggregation, so this value is not guaranteed to match
the standard SARI implementations used in published work or official
leaderboards. Treat it as a provisional within-project comparison only; use a
validated standard SARI implementation with its exact version and settings
recorded before making formal cross-paper or leaderboard comparisons.

Use the training split to fit model parameters and validation split to select
the checkpoint (lowest validation loss). Do not train, tune hyperparameters,
or select a checkpoint using the test split. The supplied test sets may be
used for the project's final local evaluation.

## Environment setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
```

If PowerShell blocks activation because script execution is disabled, either
allow activation for the current PowerShell process with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate
again, or call the environment's Python executable directly as
`.venv\Scripts\python.exe` in each command. If `py -3.12` is unavailable,
install Python 3.12 and reopen PowerShell, or replace `3.12` with an installed
Python version between 3.10 and 3.12.

Install PyTorch and the pinned project dependencies:

```powershell
python -m pip install torch
python -m pip install -r requirements.txt
```

The default `torch` command installs the wheel offered by the configured
package index. For a CUDA-enabled run, select the PyTorch install command
appropriate for the installed CUDA driver using the official PyTorch
installation selector before installing the remaining requirements.

LENS is an optional learned simplification metric with its own model
checkpoint. Install it only if you intend to request that metric:

```powershell
python -m pip install -r requirements-lens.txt
```

The first model training/evaluation run downloads the BART checkpoint from
Hugging Face. BERTScore and LENS also download their metric checkpoints when
first run. Those downloads require network access and enough disk space.

## Train the models

From the project root, train the sentence and document models independently:

```powershell
python baseline.py train --task sentence --output-dir models/sentence
python baseline.py train --task document --output-dir models/document
```

The default settings are:

| Setting | Default |
|---|---:|
| Random seed | 42 |
| Epochs | 3 |
| Learning rate | `5e-5` |
| Per-device batch size | 4 |
| Weight decay | `0.01` |
| Sentence input / target token limit | 512 / 128 |
| Document input / target token limit | 1024 / 256 |
| Checkpoint selection | Lowest validation loss |

`bart-base` has a model context limit; the configured document source limit is
1024 tokens. Inputs and targets longer than their configured limits are
truncated. Batch size and precision can be adjusted for available memory, for
example:

```powershell
python baseline.py train --task document --output-dir models/document --batch-size 2 --fp16
```

`--fp16` should only be used on hardware that supports half-precision training.
To pin the exact pretrained model revision for repeatability, pass its
immutable Hugging Face commit identifier:

```powershell
python baseline.py train --task sentence --output-dir models/sentence --revision <commit>
```

The output directory contains the model/tokenizer checkpoint, Trainer
checkpoints, and `run_config.json`. The run config records training settings,
Python/PyTorch/Transformers versions, the resolved model revision when
available, and SHA-256 hashes of the training and validation CSVs. Use the
same data files, model revision, seed, arguments, library versions, and
hardware/software setup for the closest reproducibility. GPU kernels and
different hardware or software builds can cause small numeric differences.

## Experiments with sentence simplification

This section brings together experiments that simplify individual sentences.
Each experiment describes its approach, training setup, and evaluation so that
the results can be considered alongside other sentence simplification methods.

### Experiment 1: BART sentence simplification

This experiment fine-tuned `facebook/bart-base` to map complex biomedical
sentences to their aligned Cochrane lay-text targets. It used the sentence
training and validation splits described above and the following settings:
seed 42, three epochs, learning rate `5e-5`, batch size 4, weight decay
`0.01`, FP16, and maximum source/target lengths of 512/128 tokens.
The best checkpoint was selected by lowest validation loss.

Validation loss decreased over the three recorded epochs:

| Epoch | Training step | Validation loss |
|---:|---:|---:|
| 1 | 1,439 | 3.3118 |
| 2 | 2,878 | 3.2018 |
| 3 | 4,317 | 3.1959 |

The lowest recorded validation loss was 3.1959 at epoch 3. This is the model's
validation token-prediction loss; it is not a SARI, BLEU, BERTScore,
readability, or human-quality score.

#### Held-out test evaluation

The evaluation covers 1,512 held-out sentence examples and reports:

| Metric | Score | Interpretation |
|---|---:|---|
| SARI | 25.25 | Local single-reference SARI variant; see the implementation caveat below. |
| BLEU | 26.48 | Corpus BLEU, SacreBLEU 2.4.3; measures n-gram overlap with the reference. |
| BERTScore F1 | 92.66 | BERTScore F1 on a 0–100 scale, using `roberta-large` (bert-score 0.3.13). |

These are complementary similarity measures, not a single overall quality
score. In particular, SARI is computed by this project's local implementation
with lowercase whitespace tokens, one reference, 1–4-gram orders, and a
sentence-level mean. It should not be treated as a canonical SARI result or
compared directly with scores from a different implementation. BLEU and
BERTScore also measure similarity to the reference, not factual correctness,
clinical safety, or readability. The evaluation used deterministic beam
search with 4 beams, batch size 8, and source/target limits of 512/128 tokens.

Some corpus-level observations from `sentence_test_predictions.csv` help
explain these scores:

- 580 of 1,512 references (38.4%) are empty, representing aligned deletion
  targets. The model produced an empty output for 114 of those cases; it
  produced a non-empty output for the other 466.
- 214 outputs (14.2%) are empty overall, including 100 examples with a
  non-empty reference.
- 647 outputs (42.8%) exactly match their source after trimming whitespace.
  The average source length is 25.1 whitespace-separated words and the
  average prediction length is 16.8, about one-third shorter overall. This
  reduction is not uniformly a simplification: shortening can omit details,
  and copying leaves difficult wording unchanged.

These counts show both why edit-based metrics matter for this dataset and why
the aggregate scores need careful interpretation. In the predictions, some
outputs retain technical terms or numerical detail while references use
broader lay wording; other outputs shorten a claim. Check the original source,
reference, and prediction together, especially for numbers, negation,
uncertainty, outcomes, and adverse effects. Neither high semantic similarity
nor shorter output alone establishes that the medical meaning was preserved.

The evaluation artifacts are `results/sentence_test_metrics.json` and
`results/sentence_test_predictions.csv`. For a validation-set evaluation,
run:

```powershell
python baseline.py evaluate --task sentence --split val --model-path bart-cochrane
```

The recorded test run can be reproduced with:

```powershell
python baseline.py evaluate --task sentence --split test --model-path bart-cochrane --batch-size 8
```

The test command writes its metric JSON and prediction CSV under `results/`;
by default it overwrites the existing test artifacts. Use the `--metrics-output`
and `--predictions` options to preserve them when running another evaluation.

## Evaluate

### Start with the copy baseline

The `copy` baseline returns each complex input unchanged. It needs no trained
model and is useful to confirm that the data loader, prediction writing, and
SARI implementation work before training:

```powershell
python baseline.py evaluate --task sentence --split val --model-path copy --metrics sari
python baseline.py evaluate --task document --split val --model-path copy --metrics sari
```

### Evaluate trained checkpoints

After training, run validation evaluation and use those results for model or
hyperparameter decisions:

```powershell
python baseline.py evaluate --task sentence --split val --model-path models/sentence
python baseline.py evaluate --task document --split val --model-path models/document
```

Once choices are final, evaluate on the held-out test sets:

```powershell
python baseline.py evaluate --task sentence --split test --model-path models/sentence
python baseline.py evaluate --task document --split test --model-path models/document
```

Greedy/random sampling is disabled; generation uses deterministic beam search
with 4 beams by default. Evaluation uses the same input token limits as
training. To change generation or batch settings, use `--num-beams`,
`--batch-size`, `--max-source-length`, or `--max-target-length`.

### Metrics and output files

By default, evaluation computes:

- **SARI:** simplification edits (additions, retained content, and deletions).
  See the SARI implementation note above: the current code reports a local
  single-reference variant, not a validated canonical SARI score.
- **BLEU:** corpus BLEU from SacreBLEU, comparing predictions to the reference.
- **BERTScore F1:** semantic similarity from `roberta-large`, reported on a
  0–100 scale.

Add the optional LENS metric after installing `requirements-lens.txt`:

```powershell
python baseline.py evaluate --task sentence --split val --model-path models/sentence --metrics sari bleu bertscore lens
```

Sentence metrics compare each output with the aligned reference sentence(s)
joined into one string; document metrics compare with the lay abstract.

Each run writes two files by default:

- `results/<task>_<split>_predictions.csv`: pair ID, source, reference, and
  generated prediction (also retaining sentence/paragraph identifiers when
  available).
- `results/<task>_<split>_metrics.json`: task, split, model, input CSV hash,
  example count, metric scores and versions, and evaluation settings.

For example, `--predictions path\outputs.csv` and
`--metrics-output path\scores.json` change these destinations. To evaluate
only a subset of available metrics, pass a list such as `--metrics sari bleu`.
BLEU requires `sacrebleu`; BERTScore requires `bert-score`; LENS requires the
optional `lens-metric` package. A missing requested metric dependency is
reported as an error rather than silently skipped.

## Tests

Run the lightweight loader/SARI tests without training models:

```powershell
python -m unittest discover -s tests -v
```

These tests cover sentence target parsing, empty deletion targets, document
references, and basic SARI behavior. A passing unit test does not mean the
pretrained model has been trained; training and model-based evaluation are
separate commands described above.
