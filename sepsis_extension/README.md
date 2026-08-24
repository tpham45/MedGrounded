# sepsis_extension: Uncertainty-Aware RAG for SEP-1 Compliance

An extension of MedGrounded's faithfulness-focused RAG pipeline to a new
question: instead of (or in addition to) measuring whether an answer is
grounded in its context, can the system also tell *when it doesn't know*, and
say so explicitly instead of guessing?

The target task is checking **SEP-1** ("Severe Sepsis/Septic Shock Early
Management Bundle") compliance from clinical notes -- e.g. "were antibiotics
given within 3 hours?" -- across 5 core bundle elements. The approach is
inspired by [COMPOSER (Boussina et al., *npj Digital Medicine* 2024)](https://www.nature.com/articles/s41746-024-01029-2)
from Dr. Shamim Nemati's lab at UCSD, which uses conformal prediction so a
sepsis prediction model can flag a case as "indeterminate" instead of forcing
a low-confidence guess, cutting false alarms by ~75% versus the prior model.

This project doesn't implement conformal prediction itself (that needs a
calibration set with a formal coverage guarantee, which doesn't exist here
yet). It borrows the same *philosophy* -- combine multiple uncertainty
signals into a confidence score, and gate low-confidence answers to an
explicit "I don't know" state -- applied to a RAG question-answering setting
instead of a real-time prediction model. See `src/uncertainty.py` for the
concrete mechanism: retrieval similarity + self-consistency across repeated
LLM samples, combined and thresholded.

## Why synthetic data (for now)

The obvious first choice, **MIMIC-III Clinical Database Demo**, does not work
for this: its `NOTEEVENTS` table has been stripped of all row data in the
public demo release (verified directly against `NOTEEVENTS.csv` on
PhysioNet -- the file is 95 bytes, header only, no note text). **MIMIC-IV**
(without "-Note") also doesn't help -- it's structured/tabular data only, no
free text. Free-text clinical notes live in the separate **MIMIC-IV-Note**
project on PhysioNet, which requires its own credentialed access approval.

So `data/generate_synthetic_data.py` writes 5 hand-written, fake clinical
notes (not real patient data) in the MIMIC-IV-Note schema
(`subject_id`, `hadm_id`, `text`), deliberately mixing clear notes (should
yield high-confidence, correct answers) with sparse/ambiguous ones (should
get flagged `INSUFFICIENT EVIDENCE`) -- along with a ground truth CSV for
`evaluate.py` to score against.

**With synthetic data, evaluation results are not clinically meaningful** --
they only confirm the pipeline's logic runs correctly end to end. Real,
meaningful accuracy/coverage numbers require running this against actual
MIMIC-IV-Note data.

### Switching to real MIMIC-IV-Note data

Once credentialed access is approved:

1. Export/format the real notes as JSONL matching the schema in
   `data/generate_synthetic_data.py` (`subject_id`, `hadm_id`, `text`).
2. Build a ground truth CSV in the same shape as
   `data/synthetic/ground_truth.csv` (`subject_id,hadm_id,question_id,ground_truth`).
3. Point `config.py` at the new files -- either edit `NOTES_PATH` /
   `GROUND_TRUTH_PATH` directly, or set the `SEPSIS_NOTES_PATH` /
   `SEPSIS_GROUND_TRUTH_PATH` environment variables.

No code in `src/` needs to change.

## Structure

```
sepsis_extension/
├── config.py                      # all paths/params in one place
├── data/
│   ├── generate_synthetic_data.py # writes the fake notes + ground truth
│   ├── synthetic/                 # generated fixtures (committed, tiny)
│   └── processed/                 # generated chunks/embeddings/FAISS index (gitignored)
├── src/
│   ├── data_loader.py             # load notes, chunk with overlap
│   ├── retrieval.py                # embed + FAISS index + similarity search
│   ├── generation.py               # Claude call -> structured Yes/No/INSUFFICIENT EVIDENCE
│   ├── uncertainty.py              # retrieval confidence + self-consistency -> confidence gate
│   └── evaluate.py                 # run everything, score vs ground truth, bucket by confidence
├── output/                         # generated eval CSVs (gitignored)
└── requirements.txt
```

## How to run

```bash
# From the sepsis_extension/ directory, with the repo's venv activated
# (or a fresh one -- this subfolder's requirements.txt is self-contained).
pip install -r requirements.txt

# 1. Generate the synthetic notes + ground truth
python data/generate_synthetic_data.py

# 2. Run the full pipeline and evaluate
python src/evaluate.py
```

Set `ANTHROPIC_API_KEY` in the environment (or the repo root `.env`) to use
real Claude generation. Without it, `generation.py` automatically falls back
to a rule-based mock (printing a clear `[MOCK MODE]` warning) so the rest of
the pipeline can still be exercised.

`evaluate.py` prints a table of accuracy and "insufficient evidence" rate by
confidence bucket (Low/Medium/High), and writes:

- `output/eval_detailed.csv` -- one row per (patient, question)
- `output/eval_summary.csv` -- accuracy/insufficient-evidence rate per bucket

The goal to demonstrate (even on synthetic data, as a logic check): accuracy
should increase from the Low to the High confidence bucket, and the Low
bucket should have a higher `INSUFFICIENT EVIDENCE` flag rate than the High
bucket.
