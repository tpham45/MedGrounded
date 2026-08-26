"""Generate synthetic SEP-1 clinical notes + ground truth (Step 1 of the sepsis pipeline).

Parallel to load_data.py, which pulls PubMedQA from HuggingFace into data/raw/ --
this script is the sepsis-task equivalent of that first pipeline step, except the
"source" is procedurally generated instead of downloaded.

Why synthetic data instead of a real MIMIC extract: MIMIC-III Clinical Database
Demo does NOT work here -- its NOTEEVENTS table has been stripped of all row data
in the public demo release (checked directly against NOTEEVENTS.csv on PhysioNet --
the file is 95 bytes, header only, no note text, removed for privacy reasons when
MIT built the public demo). MIMIC-IV (without "-Note") also doesn't help: it only
has structured/tabular data, no free-text notes. Free-text clinical notes live in
the separate MIMIC-IV-Note project on PhysioNet, which requires its own
credentialed access approval.

So: everything below is FAKE, procedurally generated text (not real patient
records), used only to exercise the pipeline's logic (chunking, retrieval,
generation, uncertainty scoring, evaluation) end-to-end at a more realistic scale
before real data is available. A small set of hand-written "keyword-only" notes
would make retrieval/generation trivially easy and wouldn't stress-test the
uncertainty gate -- so notes are composed from 5 clinician-voice archetypes
(attending, bedside nurse, resident, fellow, discharge summary), each with its
own sentence templates, abbreviation habits, and typical note length, combined
with randomized clinical facts and a randomized "documentation completeness"
level per note (some notes state every SEP-1-relevant fact clearly; some omit
several; a few omit almost everything). Ground truth is "None" wherever a fact
was not included in the generated text -- exactly the case the confidence gate
in src/eval/uncertainty.py is meant to catch. Everything is driven by a fixed
random seed (RANDOM_SEED) for reproducibility.

Once MIMIC-IV-Note credentialed access is approved, point config paths (or just
replace the two output files below) at real extracts in this same schema
(subject_id, hadm_id, text / subject_id, hadm_id, ground_truth). Nothing
downstream (chunk_and_embed_sepsis.py, retrieve_sepsis.py, generate_sepsis.py,
uncertainty.py, evaluate_sepsis.py) needs to change.
"""

import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from generation.generate_sepsis import SEP1_QUESTIONS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
NOTES_PATH = BASE_DIR / "data" / "raw" / "sepsis_notes_synthetic.jsonl"
GROUND_TRUTH_PATH = BASE_DIR / "data" / "raw" / "sepsis_ground_truth_synthetic.jsonl"

NUM_NOTES = 60
RANDOM_SEED = 42

INFECTION_SOURCES = [
    "suspected pneumonia",
    "suspected urinary tract infection",
    "suspected intra-abdominal infection / cholangitis",
    "cellulitis / soft tissue infection",
    "suspected meningitis",
    "bacteremia of unclear source",
    "post-surgical wound infection",
    "suspected osteomyelitis",
]

ARCHETYPES = ["attending_formal", "nurse_shorthand", "resident_terse", "fellow_verbose", "discharge_summary"]

# Higher = more likely each SEP-1-relevant fact is actually stated in the note text.
DOC_QUALITY_LEVELS = {"high": 0.93, "medium": 0.68, "low": 0.32}

TYPO_MAP = {
    "antibiotics": ["antibotics", "antiobiotics"],
    "administered": ["administerd", "adminstered"],
    "hypotension": ["hypotention"],
    "received": ["recieved"],
    "culture": ["cluture"],
    "lactate": ["loctate"],
    "patient": ["pt"],
}


def fmt_time(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def add_minutes(hour: int, minute: int, delta: int) -> tuple[int, int]:
    total = hour * 60 + minute + delta
    return (total // 60) % 24, total % 60


# --- Per-archetype sentence builders --------------------------------------
# Each returns a sentence (str) or None if that fact isn't documented in this note.


def build_attending_formal(rng, facts):
    lines = [
        f"ED sepsis note. {facts['presentation_time']} patient meets SIRS criteria, "
        f"{facts['source']}, sepsis protocol activated."
    ]
    if facts["q1_documented"]:
        verdict = "within" if facts["q1_yes"] else "beyond"
        lines.append(
            f"Broad-spectrum antibiotics ({facts['abx_name']}) administered at "
            f"{facts['abx_time']}, {facts['abx_delay_min']} minutes after presentation "
            f"({verdict} the 3-hour window)."
        )
    if facts["q2_documented"]:
        rel = "prior to" if facts["q2_yes"] else "after"
        lines.append(f"Blood cultures x2 drawn {rel} antibiotic administration.")
    if facts["q3_documented"]:
        elevated = "elevated" if facts["q3_elevated"] else "within normal range"
        lines.append(
            f"Initial lactate drawn at {facts['lactate_time']} resulted "
            f"{facts['lactate_value']} mmol/L ({elevated})."
        )
    if facts["q4_documented"]:
        lines.append(
            f"Repeat lactate drawn at {facts['repeat_lactate_time']} resulted "
            f"{facts['repeat_lactate_value']} mmol/L."
        )
    if facts["q5_documented"]:
        if facts["hypotensive"]:
            if facts["q5_yes"]:
                lines.append(
                    f"Patient hypotensive on arrival (MAP {facts['map']}); 30 mL/kg "
                    f"crystalloid bolus started {facts['bolus_time']}."
                )
            else:
                lines.append(
                    f"Patient hypotensive on arrival (MAP {facts['map']}); no fluid "
                    f"bolus was given despite hypotension."
                )
        else:
            lines.append(
                f"Patient normotensive throughout (MAP {facts['map']}); no fluid bolus "
                f"given, none indicated."
            )
    lines.append("Patient stabilized, transferred to floor for continued monitoring.")
    return " ".join(lines)


def build_nurse_shorthand(rng, facts):
    lines = [f"{facts['presentation_time']} pt febrile, tachy, sepsis protocol per MD, {facts['source']}."]
    if facts["q1_documented"]:
        lines.append(f"Abx ({facts['abx_name']}) given {facts['abx_time']}.")
    if facts["q2_documented"]:
        lines.append("BCx sent" + (" prior to abx." if facts["q2_yes"] else " after abx started."))
    if facts["q3_documented"]:
        lines.append(f"Lactic {facts['lactate_value']} at {facts['lactate_time']}.")
    if facts["q4_documented"]:
        lines.append(f"Repeat lactic {facts['repeat_lactate_value']} at {facts['repeat_lactate_time']}.")
    if facts["q5_documented"]:
        if facts["hypotensive"] and facts["q5_yes"]:
            lines.append(f"MAP {facts['map']}, IVF bolus 30mL/kg running.")
        elif facts["hypotensive"]:
            lines.append(f"MAP {facts['map']}, no bolus given yet.")
        else:
            lines.append(f"VSS, MAP {facts['map']}, no bolus needed.")
    lines.append("Will reassess q1h, MD aware.")
    return " ".join(lines)


def build_resident_terse(rng, facts):
    lines = [f"Sepsis w/u initiated {facts['presentation_time']}, {facts['source']}."]
    parts = []
    if facts["q1_documented"]:
        parts.append(f"abx {facts['abx_time']}")
    if facts["q2_documented"]:
        parts.append("cx before abx" if facts["q2_yes"] else "cx after abx")
    if facts["q3_documented"]:
        parts.append(f"lactate {facts['lactate_value']}")
    if parts:
        lines.append(", ".join(parts) + ".")
    if facts["q4_documented"]:
        lines.append(f"Repeat lactate {facts['repeat_lactate_value']}.")
    if facts["q5_documented"]:
        if facts["hypotensive"]:
            lines.append("Bolus given." if facts["q5_yes"] else "No bolus given.")
        else:
            lines.append("Normotensive, no bolus.")
    lines.append("F/u am.")
    return " ".join(lines)


def build_fellow_verbose(rng, facts):
    lines = [
        f"Called to evaluate patient at {facts['presentation_time']} for concern of sepsis "
        f"in the setting of {facts['source']}. Reviewed the chart in detail and discussed "
        f"with the primary team at length regarding the overall clinical trajectory."
    ]
    if facts["q1_documented"]:
        verdict = "appropriately within" if facts["q1_yes"] else "unfortunately outside"
        lines.append(
            f"After extensive discussion, {facts['abx_name']} was ultimately started at "
            f"{facts['abx_time']}, which upon calculation falls {verdict} the SEP-1 3-hour "
            f"target given the presentation time noted above."
        )
    if facts["q2_documented"]:
        rel = "were obtained prior to" if facts["q2_yes"] else "were unfortunately not obtained until after"
        lines.append(f"Blood cultures {rel} the first dose of antibiotics being given.")
    if facts["q3_documented"]:
        elevated = "notably elevated" if facts["q3_elevated"] else "reassuringly within normal limits"
        lines.append(
            f"An initial lactate was sent and resulted at {facts['lactate_value']} mmol/L, "
            f"which is {elevated}."
        )
    if facts["q4_documented"]:
        lines.append(
            f"Given the above, a repeat lactate was subsequently drawn and resulted "
            f"{facts['repeat_lactate_value']} mmol/L."
        )
    if facts["q5_documented"]:
        if facts["hypotensive"] and facts["q5_yes"]:
            lines.append(
                f"The patient was hypotensive (MAP {facts['map']}) and received a full "
                f"30 mL/kg crystalloid bolus as per protocol."
            )
        elif facts["hypotensive"]:
            lines.append(
                f"Despite documented hypotension (MAP {facts['map']}), a bolus does not "
                f"appear to have been administered at this time, which I will follow up on."
            )
        else:
            lines.append(
                f"Blood pressure has remained within normal limits (MAP {facts['map']}) "
                f"throughout, so a fluid bolus was not clinically indicated."
            )
    lines.append("Will continue to follow closely and provide further recommendations as needed.")
    return " ".join(lines)


def build_discharge_summary(rng, facts):
    lines = [
        f"Hospital Course: Patient presented with {facts['source']} and met criteria for "
        f"sepsis; the sepsis protocol was activated shortly after arrival."
    ]
    if facts["q1_documented"]:
        rel = "promptly, well within the recommended window" if facts["q1_yes"] else "later than the recommended window"
        lines.append(f"Antibiotics ({facts['abx_name']}) were started {rel} after presentation.")
    if facts["q2_documented"]:
        rel = "before" if facts["q2_yes"] else "after"
        lines.append(f"Blood cultures had been collected {rel} antibiotics were started.")
    if facts["q3_documented"]:
        elevated = "elevated" if facts["q3_elevated"] else "normal"
        lines.append(f"An initial lactate was {elevated} at {facts['lactate_value']} mmol/L.")
    if facts["q4_documented"]:
        lines.append(f"A repeat lactate later in the course was {facts['repeat_lactate_value']} mmol/L.")
    if facts["q5_documented"]:
        if facts["hypotensive"] and facts["q5_yes"]:
            lines.append("The patient required a fluid bolus for hypotension early in the admission.")
        elif facts["hypotensive"]:
            lines.append("The patient was noted to be hypotensive, though no bolus is documented in the chart.")
        else:
            lines.append("The patient did not require fluid resuscitation, remaining hemodynamically stable.")
    lines.append("The patient's condition improved over the course of admission and was discharged in stable condition.")
    return " ".join(lines)


BUILDERS = {
    "attending_formal": build_attending_formal,
    "nurse_shorthand": build_nurse_shorthand,
    "resident_terse": build_resident_terse,
    "fellow_verbose": build_fellow_verbose,
    "discharge_summary": build_discharge_summary,
}


def inject_typos(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    for i, word in enumerate(words):
        stripped = word.strip(".,;()")
        for key, variants in TYPO_MAP.items():
            if stripped.lower() == key and rng.random() < 0.5:
                replacement = rng.choice(variants)
                words[i] = word.replace(stripped, replacement)
                break
    return " ".join(words)


def generate_facts(rng: random.Random) -> dict:
    facts = {}
    facts["source"] = rng.choice(INFECTION_SOURCES)

    presentation_hour = rng.randint(0, 23)
    presentation_minute = rng.choice([0, 5, 10, 15, 20, 30, 45])
    facts["presentation_time"] = fmt_time(presentation_hour, presentation_minute)

    doc_quality = rng.choices(list(DOC_QUALITY_LEVELS), weights=[0.40, 0.35, 0.25])[0]
    doc_prob = DOC_QUALITY_LEVELS[doc_quality]
    facts["doc_quality"] = doc_quality

    # q1 -- antibiotics within 3h
    facts["q1_documented"] = rng.random() < doc_prob
    if facts["q1_documented"]:
        delay_bucket = rng.choices(["fast", "moderate", "slow"], weights=[0.55, 0.25, 0.20])[0]
        delay_min = {
            "fast": rng.randint(15, 170),
            "moderate": rng.randint(190, 350),
            "slow": rng.randint(400, 600),
        }[delay_bucket]
        facts["abx_delay_min"] = delay_min
        facts["q1_yes"] = delay_min < 180
        h, m = add_minutes(presentation_hour, presentation_minute, delay_min)
        facts["abx_time"] = fmt_time(h, m)
        facts["abx_name"] = rng.choice(
            ["vancomycin and piperacillin-tazobactam", "ceftriaxone", "meropenem", "cefepime and metronidazole"]
        )

    # q2 -- blood culture before antibiotics
    facts["q2_documented"] = rng.random() < doc_prob
    if facts["q2_documented"]:
        facts["q2_yes"] = rng.random() < 0.75

    # q3 -- initial lactate within 6h
    facts["q3_documented"] = rng.random() < doc_prob
    if facts["q3_documented"]:
        within_6h = rng.random() < 0.85
        h, m = add_minutes(
            presentation_hour, presentation_minute, rng.randint(5, 340 if within_6h else 500)
        )
        facts["lactate_time"] = fmt_time(h, m)
        facts["q3_yes"] = within_6h
        facts["q3_elevated"] = rng.random() < 0.55
        facts["lactate_value"] = (
            round(rng.uniform(2.1, 6.5), 1) if facts["q3_elevated"] else round(rng.uniform(0.5, 1.9), 1)
        )

    # q4 -- repeat lactate, only meaningful if initial was documented AND elevated
    if facts["q3_documented"] and facts["q3_elevated"]:
        facts["q4_documented"] = rng.random() < doc_prob
        if facts["q4_documented"]:
            facts["q4_yes"] = rng.random() < 0.6
            h, m = add_minutes(presentation_hour, presentation_minute, rng.randint(200, 500))
            facts["repeat_lactate_time"] = fmt_time(h, m)
            facts["repeat_lactate_value"] = round(rng.uniform(1.0, 5.5), 1)
    else:
        facts["q4_documented"] = False

    # q5 -- 30 mL/kg bolus for hypotension/septic shock
    facts["q5_documented"] = rng.random() < doc_prob
    if facts["q5_documented"]:
        facts["hypotensive"] = rng.random() < 0.55
        facts["map"] = rng.randint(50, 64) if facts["hypotensive"] else rng.randint(70, 88)
        if facts["hypotensive"]:
            facts["q5_yes"] = rng.random() < 0.7
            h, m = add_minutes(presentation_hour, presentation_minute, rng.randint(5, 60))
            facts["bolus_time"] = fmt_time(h, m)
    else:
        facts["hypotensive"] = rng.random() < 0.3  # used only for internal consistency, not documented

    return facts


def facts_to_ground_truth(facts: dict) -> dict:
    gt = {}
    gt["q1_antibiotics_3h"] = "Yes" if facts["q1_documented"] and facts["q1_yes"] else (
        "No" if facts["q1_documented"] else "None"
    )
    gt["q2_blood_culture_before_antibiotics"] = (
        "Yes" if facts["q2_documented"] and facts["q2_yes"] else ("No" if facts["q2_documented"] else "None")
    )
    gt["q3_lactate_6h"] = "Yes" if facts["q3_documented"] and facts["q3_yes"] else (
        "No" if facts["q3_documented"] else "None"
    )
    if not facts["q3_documented"] or not facts["q3_elevated"]:
        # Repeat lactate only applies when the initial was documented AND elevated;
        # otherwise the question's premise doesn't hold, so "None" rather than a
        # forced No avoids penalizing a model for correctly noting the premise fails.
        gt["q4_lactate_repeat"] = "None"
    else:
        gt["q4_lactate_repeat"] = "Yes" if facts["q4_documented"] and facts["q4_yes"] else (
            "No" if facts["q4_documented"] else "None"
        )
    if not facts["q5_documented"]:
        gt["q5_fluid_30ml_kg"] = "None"
    elif not facts["hypotensive"]:
        gt["q5_fluid_30ml_kg"] = "No"
    else:
        gt["q5_fluid_30ml_kg"] = "Yes" if facts["q5_yes"] else "No"
    return gt


def generate_note_text(rng: random.Random, archetype: str, facts: dict) -> str:
    text = BUILDERS[archetype](rng, facts)
    if rng.random() < 0.3:
        text = inject_typos(text, rng)
    return text


def generate_notes(n: int = NUM_NOTES, seed: int = RANDOM_SEED):
    rng = random.Random(seed)
    notes = []
    for i in range(n):
        subject_id = 10001 + i
        hadm_id = 20001 + i
        archetype = rng.choice(ARCHETYPES)
        facts = generate_facts(rng)
        text = generate_note_text(rng, archetype, facts)
        ground_truth = facts_to_ground_truth(facts)
        notes.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "text": text,
                "archetype": archetype,
                "doc_quality": facts["doc_quality"],
                "ground_truth": ground_truth,
            }
        )
    return notes


def save_notes(notes: list[dict], path: Path = NOTES_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for note in notes:
            f.write(
                json.dumps(
                    {"subject_id": note["subject_id"], "hadm_id": note["hadm_id"], "text": note["text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_ground_truth(notes: list[dict], path: Path = GROUND_TRUTH_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for note in notes:
            f.write(
                json.dumps(
                    {
                        "subject_id": note["subject_id"],
                        "hadm_id": note["hadm_id"],
                        "ground_truth": note["ground_truth"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    notes = generate_notes()
    save_notes(notes)
    save_ground_truth(notes)

    archetype_counts = {a: sum(1 for n in notes if n["archetype"] == a) for a in ARCHETYPES}
    quality_counts = {q: sum(1 for n in notes if n["doc_quality"] == q) for q in DOC_QUALITY_LEVELS}

    print(f"[SYNTHETIC DATA] {len(notes)} fake notes (seed={RANDOM_SEED}) -- NOT real patient data.")
    print(f"Archetype mix: {archetype_counts}")
    print(f"Documentation-quality mix: {quality_counts}")
    print(f"Saved notes to {NOTES_PATH}")
    print(f"Saved ground truth to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
