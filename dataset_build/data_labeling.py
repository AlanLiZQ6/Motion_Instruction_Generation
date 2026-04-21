from google import genai
import json
import os
import pandas as pd
import re
import sys
import time
import argparse

# Set the API key (reads from env var GEMINI_API_KEY)
_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not _API_KEY:
    print("ERROR: GEMINI_API_KEY env var is not set.", file=sys.stderr)
    sys.exit(1)
gemini_model = genai.Client(api_key=_API_KEY)

# Testers whose videos were hflipped (left-handed); only relabel entries
# where beginner or expert belongs to this set.
TARGET_TESTERS = {"p5", "p7", "p8", "p19", "p24", "p46", "p48", "p52"}


def _player_id(video_name):
    m = re.match(r"(p\d+)_", video_name)
    return m.group(1) if m else None


def _is_target_entry(entry):
    return (_player_id(entry.get("beginner_video_name", "")) in TARGET_TESTERS
            or _player_id(entry.get("expert_video_name", "")) in TARGET_TESTERS)

# Load dataset.json
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Build lookup: beginner_video_name -> dataset entry
dataset_lookup = {entry["beginner_video_name"]: entry for entry in dataset}

csv_dir = os.path.join(os.path.dirname(__file__), 'label_csv')
csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

# ============================================================
# Mode 1: Labeling — generate 6 labels from video pairs
# ============================================================

LABELING_PROMPT = """You are a professional tennis coach analyzing two motion videos.

The FIRST video shows a learner performing a tennis stroke.
The SECOND video shows an expert performing the same stroke.

Compare the two motions and identify the SINGLE most impactful motion difference — the one that, if fixed, would give the largest improvement to the learner's technique.

Generate EXACTLY 1 coaching instruction for that difference.

RULES:
1. Focus ONLY on observable motion differences: body posture, joint angles, timing, weight transfer, rotation, balance, range of motion.
2. Do NOT name or classify the stroke type (e.g., do NOT say "forehand", "serve", "backhand", "volley", "slice", "smash", etc.).
3. Describe WHAT is wrong and HOW to fix it.
4. Use concise, imperative coaching language.
5. Output a single complete sentence.

BAD examples (do NOT generate these):
- "To improve your forehand, rotate your torso..."  (names the stroke)
- "Your serve lacks power because..."  (names the stroke)
- "For a better backhand slice..."  (names the stroke)

Output format: Return exactly 1 instruction. Nothing else."""

# ============================================================
# Mode 2: Augmentation — rephrase first label 5 times
# (Mimics CoachMe's GPT-4 augmentation strategy)
# ============================================================

AUGMENTATION_SYSTEM = """You are an experienced tennis coach who specializes in helping students improve their tennis skills.
Your task is to rephrase the given instruction in different ways while preserving the exact technical meaning.
Please follow this guideline when rewriting:
1. Use simple and clear language that beginners can easily understand and apply.
2. Maintain a clear and neutral tone with a professional and objective style.
3. Keep ALL key technical information from the original — do not omit body parts, directions, or actions mentioned.
4. Do not introduce any information, body parts, or advice not present in the original.
5. Avoid phrasing that sounds too strict or overly commanding.
6. Focus on offering constructive suggestions that help students feel motivated to improve."""

AUGMENTATION_USER = """If the target instruction does not begin with a directive verb such as "Keep", "Try", "Focus on", "Ensure", "Consider", "Aim", "Avoid", "Make sure", or "Remember", you should avoid introducing one in the rephrased version. Maintain a neutral tone. Do not introduce any information that is not present in the target instruction.
Do NOT name or classify the stroke type (e.g., do NOT say "forehand", "serve", "backhand", "volley", "slice", "smash", etc.).
Please provide exactly 5 alternative ways to rephrase the instruction, one per line, numbered 1-5. Nothing else.
Target instruction: {instruction}"""


def parse_response(text):
    """Parse numbered response into list of instructions."""
    lines = text.strip().split('\n')
    instructions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+[\.\)\:]\s*', '', line)
        if line:
            if not line.endswith('.'):
                line += '.'
            instructions.append(line)
    return instructions[:1]


def rephrase_batch(instruction):
    """Generate 5 rephrases in a single API call."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Gemini API call timed out after 60 seconds")

    user_msg = AUGMENTATION_SYSTEM + "\n\n" + AUGMENTATION_USER.format(instruction=instruction)

    # Set 60 second timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    try:
        response = gemini_model.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                {"role": "user", "parts": [{"text": user_msg}]}
            ]
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Parse 5 rephrases from response
    lines = response.text.strip().split('\n')
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\"\']|[\"\']$', '', line.strip())
        line = re.sub(r'^\d+[\.\)\:]\s*', '', line.strip())
        if line:
            if not line.endswith('.'):
                line += '.'
            results.append(line)
    return results[:5]


def run_labeling():
    """Mode 1: Generate 6 labels from video pairs."""
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        file_index = pd.read_csv(os.path.join(csv_dir, csv_file))
        for index, row in file_index.iterrows():
            beginner_id = row["id"]
            if beginner_id in dataset_lookup and len(dataset_lookup[beginner_id]["labels"]) >= 6:
                print(f"Skipping {beginner_id}: already has 6+ labels")
                continue

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    beginner_video = gemini_model.files.upload(file=row["beginner_avi_path"])
                    expert_video = gemini_model.files.upload(file=row["inference_avi_path"])

                    while beginner_video.state.name == "PROCESSING" or expert_video.state.name == "PROCESSING":
                        print("Waiting for file to be processed...")
                        time.sleep(5)
                        beginner_video = gemini_model.files.get(name=beginner_video.name)
                        expert_video = gemini_model.files.get(name=expert_video.name)
                        if beginner_video.state.name == "FAILED" or expert_video.state.name == "FAILED":
                            raise ValueError("File processing failed.")

                    response = gemini_model.models.generate_content(
                        model="gemini-3.1-flash-lite-preview",
                        contents=[
                            beginner_video,
                            expert_video,
                            LABELING_PROMPT,
                        ]
                    )

                    instructions = parse_response(response.text)
                    if beginner_id in dataset_lookup:
                        dataset_lookup[beginner_id]["labels"] = instructions
                        print(f"Generated {len(instructions)} labels for {beginner_id}")

                        with open(dataset_path, 'w') as f:
                            json.dump(dataset, f, indent=4)

                        try:
                            gemini_model.files.delete(name=beginner_video.name)
                            gemini_model.files.delete(name=expert_video.name)
                        except Exception:
                            pass

                        time.sleep(2)
                    else:
                        print(f"Warning: {beginner_id} not found in dataset.json")
                    break

                except Exception as e:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Error (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print(f"  Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  FAILED after {max_retries} attempts, skipping {beginner_id}")


def run_augmentation():
    """Mode 2: Rephrase first label 5 times for each sample."""
    total = len(dataset)
    processed = 0
    skipped = 0

    for i, entry in enumerate(dataset):
        beginner_id = entry["beginner_video_name"]

        # Skip if already augmented
        if "augmented_labels" in entry and len(entry.get("augmented_labels", [])) >= 5:
            skipped += 1
            continue

        original_label = entry["labels"][0]
        augmented = []
        max_retries = 5
        success = True

        print(f"\n[{i+1}/{total}] Augmenting {beginner_id}...")
        print(f"  Original: {original_label[:100]}")
        for attempt in range(max_retries):
            try:
                augmented = rephrase_batch(original_label)
                for idx, rep in enumerate(augmented):
                    print(f"  Rephrase {idx+1}: {rep[:100]}")
                time.sleep(2)
                break
            except Exception as e:
                wait_time = 30 * (attempt + 1)
                print(f"  Error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  FAILED after {max_retries} attempts, skipping {beginner_id}")
                    success = False

        if success and len(augmented) == 5:
            entry["labels"] = [original_label]
            entry["augmented_labels"] = augmented
            processed += 1
            print(f"  Done! {processed} augmented so far.")
        else:
            print(f"  Incomplete augmentation for {beginner_id}, skipping")
            continue

        # Save progress after each entry
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=4)

    # Final save
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"\nDone! Augmented: {processed}, Skipped: {skipped}, Total: {total}")


def run_relabel_targets():
    """Combined relabel for entries involving TARGET_TESTERS.
    For each target entry: call Gemini to generate 1 label from videos,
    then rephrase into 5 augmented labels, then save. Non-target entries untouched.

    Resumable: records completed beginner_ids in a sidecar file
    `relabel_done.txt` next to dataset.json. On restart, already-done entries
    are skipped to avoid re-calling Gemini.
    """
    # Build csv row lookup by beginner_id to fetch avi paths
    csv_rows = {}
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(csv_dir, csv_file))
        for _, row in df.iterrows():
            csv_rows[row["id"]] = row

    done_path = os.path.join(os.path.dirname(dataset_path), 'relabel_done.txt')
    done = set()
    if os.path.exists(done_path):
        with open(done_path, 'r') as f:
            done = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(done)} already-done entries from {done_path}")

    def mark_done(bid):
        done.add(bid)
        with open(done_path, 'a') as f:
            f.write(bid + '\n')

    targets = [e for e in dataset if _is_target_entry(e)]
    total = len(targets)
    already = sum(1 for e in targets if e["beginner_video_name"] in done)
    print(f"Target entries: {total} / {len(dataset)}  (already done: {already})")

    processed = 0
    failed = []
    for i, entry in enumerate(targets):
        beginner_id = entry["beginner_video_name"]
        if beginner_id in done:
            continue
        if beginner_id not in csv_rows:
            print(f"[{i+1}/{total}] {beginner_id}: not in csv, skip")
            failed.append(beginner_id)
            continue
        row = csv_rows[beginner_id]

        # ---- Step A: labeling (1 instruction) ----
        new_label = None
        for attempt in range(5):
            try:
                beginner_video = gemini_model.files.upload(file=row["beginner_avi_path"])
                expert_video = gemini_model.files.upload(file=row["inference_avi_path"])

                while beginner_video.state.name == "PROCESSING" or expert_video.state.name == "PROCESSING":
                    time.sleep(5)
                    beginner_video = gemini_model.files.get(name=beginner_video.name)
                    expert_video = gemini_model.files.get(name=expert_video.name)
                    if beginner_video.state.name == "FAILED" or expert_video.state.name == "FAILED":
                        raise ValueError("File processing failed.")

                response = gemini_model.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    contents=[beginner_video, expert_video, LABELING_PROMPT],
                )
                instructions = parse_response(response.text)
                if not instructions:
                    raise ValueError("Empty label returned")
                new_label = instructions[0]

                try:
                    gemini_model.files.delete(name=beginner_video.name)
                    gemini_model.files.delete(name=expert_video.name)
                except Exception:
                    pass
                time.sleep(2)
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"  [{beginner_id}] label error attempt {attempt+1}/5: {e}; retry {wait}s")
                if attempt < 4:
                    time.sleep(wait)

        if new_label is None:
            print(f"[{i+1}/{total}] {beginner_id}: LABEL FAILED")
            failed.append(beginner_id)
            continue

        # Save label immediately
        entry["labels"] = [new_label]
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"[{i+1}/{total}] {beginner_id} LABEL: {new_label[:100]}")

        # ---- Step B: augmentation (5 rephrases) ----
        augmented = None
        for attempt in range(5):
            try:
                augmented = rephrase_batch(new_label)
                if len(augmented) == 5:
                    break
                raise ValueError(f"Got {len(augmented)} rephrases, expected 5")
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"  [{beginner_id}] aug error attempt {attempt+1}/5: {e}; retry {wait}s")
                if attempt < 4:
                    time.sleep(wait)
                    augmented = None

        if augmented is None or len(augmented) != 5:
            print(f"  [{beginner_id}] AUG FAILED; keeping label only")
            failed.append(beginner_id + " (aug)")
            continue

        entry["augmented_labels"] = augmented
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        mark_done(beginner_id)
        processed += 1
        for idx, r in enumerate(augmented):
            print(f"    aug{idx+1}: {r[:100]}")
        time.sleep(2)

    print(f"\nDone. Relabeled {processed}/{total}. Failed: {len(failed)}")
    if failed:
        print("Failures:")
        for f in failed:
            print(f"  {f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        choices=['label', 'augment', 'relabel-targets'],
                        required=True,
                        help=('label: generate labels from videos (all entries). '
                              'augment: rephrase existing labels (all entries). '
                              'relabel-targets: label+augment for target testers only.'))
    args = parser.parse_args()

    if args.mode == 'label':
        run_labeling()
    elif args.mode == 'augment':
        run_augmentation()
    elif args.mode == 'relabel-targets':
        run_relabel_targets()

