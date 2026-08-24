"""Generate synthetic clinical notes + ground truth for the SEP-1 pipeline.

IMPORTANT -- why this exists instead of a real MIMIC dataset:
The obvious first choice, MIMIC-III Clinical Database Demo, does NOT work for
this: its NOTEEVENTS table has been stripped of all row data in the public
demo release (checked directly against the NOTEEVENTS.csv on PhysioNet --
the file is 95 bytes, i.e. header only, no note text). MIMIC-IV (without
"-Note") also does not help: it only has structured/tabular data, no
free-text notes. Free-text clinical notes live in the separate MIMIC-IV-Note
project on PhysioNet, which requires its own credentialed access approval.

So: everything below is FAKE, hand-written data (not real patient records),
used only to exercise the pipeline's logic (chunking, retrieval, generation,
uncertainty scoring, evaluation) end-to-end before real data is available.
It intentionally mixes clear notes (should yield high-confidence, correct
answers) with sparse/ambiguous notes (should get flagged "INSUFFICIENT
EVIDENCE"). Ground truth is "None" wherever the note genuinely does not
document enough to answer -- that's the case the confidence gate is meant
to catch.

Once MIMIC-IV-Note credentialed access is approved, point config.py's
NOTES_PATH / GROUND_TRUTH_PATH (or the SEPSIS_NOTES_PATH / SEPSIS_GROUND_TRUTH_PATH
env vars) at real extracts in this same schema -- no code in src/ needs to change.
"""

import csv
import json
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import GROUND_TRUTH_PATH, NOTES_PATH, SEP1_QUESTIONS  # noqa: E402

# Each note: (subject_id, hadm_id, text, {question_id: ground_truth}).
# ground_truth is "Yes" / "No" / "None" ("None" = not determinable from this note text).
SYNTHETIC_NOTES = [
    (
        10001,
        20001,
        "ED sepsis note. 14:00 patient meets SIRS criteria with suspected pneumonia, "
        "sepsis protocol activated. Blood cultures x2 drawn at 14:10, prior to any "
        "antibiotic administration. Vancomycin and piperacillin-tazobactam started at "
        "14:45 (45 minutes after presentation). Initial lactate drawn at 14:15 resulted "
        "4.2 mmol/L (elevated). Repeat lactate drawn at 18:00 resulted 2.1 mmol/L, "
        "trending down. Patient hypotensive on arrival (MAP 58); 30 mL/kg crystalloid "
        "bolus (2100 mL for 70 kg patient) started 14:20, completed by 15:30. "
        "Patient stabilized, transferred to ICU for continued monitoring.",
        {
            "q1_antibiotics_3h": "Yes",
            "q2_blood_culture_before_antibiotics": "Yes",
            "q3_lactate_6h": "Yes",
            "q4_lactate_repeat": "Yes",
            "q5_fluid_30ml_kg": "Yes",
        },
    ),
    (
        10002,
        20002,
        "Sepsis identified 08:00 (fever, tachycardia, suspected UTI source). Blood "
        "cultures were NOT obtained -- note states 'cultures deferred, unable to "
        "obtain IV access initially.' Antibiotics (ceftriaxone) were not started "
        "until 12:30, 4.5 hours after presentation, due to pharmacy delay. Lactate "
        "drawn at 08:30 resulted 1.4 mmol/L, within normal range. Blood pressure "
        "remained stable throughout (MAP 74-80); no hypotension documented and no "
        "fluid bolus was given.",
        {
            "q1_antibiotics_3h": "No",
            "q2_blood_culture_before_antibiotics": "No",
            "q3_lactate_6h": "Yes",
            "q4_lactate_repeat": "No",
            "q5_fluid_30ml_kg": "No",
        },
    ),
    (
        10003,
        20003,
        "Nursing note: Patient with fever and tachycardia, sepsis protocol initiated "
        "per team discussion. Primary team aware and at bedside. Plan to follow up. "
        "Will reassess.",
        {
            "q1_antibiotics_3h": "None",
            "q2_blood_culture_before_antibiotics": "None",
            "q3_lactate_6h": "None",
            "q4_lactate_repeat": "None",
            "q5_fluid_30ml_kg": "None",
        },
    ),
    (
        10004,
        20004,
        "ED sepsis note. Suspected intra-abdominal source. Blood cultures x2 drawn "
        "09:05. Ceftriaxone and metronidazole administered 09:40 (35 minutes after "
        "cultures, well within the window). Labs pending at time of this note, "
        "including lactate -- result not yet available. IV fluids running per orders; "
        "rate and total volume not specified in this note, and no mention of blood "
        "pressure or hypotension status.",
        {
            "q1_antibiotics_3h": "Yes",
            "q2_blood_culture_before_antibiotics": "Yes",
            "q3_lactate_6h": "None",
            "q4_lactate_repeat": "None",
            "q5_fluid_30ml_kg": "None",
        },
    ),
    (
        10005,
        20005,
        "Sepsis recognized 20:00 (suspected cholangitis). Blood cultures drawn 20:05, "
        "prior to antibiotics. Piperacillin-tazobactam started 21:45 (1 hour 45 "
        "minutes after presentation). Initial lactate drawn 20:10 resulted 3.8 "
        "mmol/L (elevated). Repeat lactate drawn 23:30 resulted 3.5 mmol/L. Patient "
        "normotensive throughout admission (MAP 70-80, SBP never below 90); no "
        "hypotension or septic shock criteria met, and no fluid bolus was given as "
        "it was not indicated.",
        {
            "q1_antibiotics_3h": "Yes",
            "q2_blood_culture_before_antibiotics": "Yes",
            "q3_lactate_6h": "Yes",
            "q4_lactate_repeat": "Yes",
            "q5_fluid_30ml_kg": "No",
        },
    ),
]


def write_notes(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for subject_id, hadm_id, text, _ in SYNTHETIC_NOTES:
            f.write(
                json.dumps(
                    {"subject_id": subject_id, "hadm_id": hadm_id, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_ground_truth(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    question_ids = [q["id"] for q in SEP1_QUESTIONS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "hadm_id", "question_id", "ground_truth"])
        for subject_id, hadm_id, _, answers in SYNTHETIC_NOTES:
            for qid in question_ids:
                writer.writerow([subject_id, hadm_id, qid, answers[qid]])


def main():
    write_notes(NOTES_PATH)
    write_ground_truth(GROUND_TRUTH_PATH)
    print(f"[SYNTHETIC DATA] {len(SYNTHETIC_NOTES)} fake notes -- NOT real patient data.")
    print(f"Saved notes to {NOTES_PATH}")
    print(f"Saved ground truth to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
