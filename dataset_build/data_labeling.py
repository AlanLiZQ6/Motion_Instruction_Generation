from google import genai
import json
import os
import pandas as pd
import sys
import time
import re
import argparse

# Set the API key
gemini_model = genai.Client(api_key="")

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

LABELING_PROMPT = """You are a professional sports coach analyzing two motion videos.

The FIRST video shows a learner performing a movement.
The SECOND video shows an expert performing the same movement.

Compare the two motions and generate EXACTLY 6 coaching instructions for the learner.

RULES:
1. Focus ONLY on observable motion differences: body posture, joint angles, timing, weight transfer, rotation, balance, range of motion.
2. Do NOT name or classify the action type (e.g., do NOT say "forehand", "serve", "backhand", "volley", "slice", "smash", etc.).
3. Each instruction should describe WHAT is wrong and HOW to fix it.
4. Use concise, imperative coaching language.
5. Each instruction should be a single complete sentence.

GOOD examples:
- "Rotate your torso further back during the preparation phase to generate more rotational power."
- "Shift your weight onto your front foot at the moment of contact instead of staying on your back foot."
- "Keep your elbow higher and further from your body to create a wider, more powerful swing arc."

BAD examples (do NOT generate these):
- "To improve your forehand, rotate your torso..."  (names the action)
- "Your serve lacks power because..."  (names the action)
- "For a better backhand slice..."  (names the action)

Output format: Return EXACTLY 6 instructions, one per line, numbered 1-6. Nothing else."""

# ============================================================
# Mode 2: Augmentation — rephrase first label 5 times
# (Mimics CoachMe's GPT-4 augmentation strategy)
# ============================================================

AUGMENTATION_SYSTEM = """You are an experienced tennis coach who specializes in helping students improve their tennis skills.
Your task is to rephrase the instruction.
Please follow this guideline when rewriting:
1. Use simple and clear language that beginners can easily understand and apply.
2. Maintain a clear and neutral tone with a professional and objective style.
3. Feel free to omit parts of the original instruction that are not particularly helpful.
4. Avoid phrasing that sounds too strict or overly commanding.
5. Focus on offering constructive suggestions that help students feel motivated to improve."""

AUGMENTATION_USER = """If the target instruction does not begin with a directive verb such as "Keep", "Try", "Focus on", "Ensure", "Consider", "Aim", "Avoid", "Make sure", or "Remember", you should avoid introducing one in the rephrased version. Maintain a neutral tone. Do not introduce any information that is not present in the target instruction.
Do NOT name or classify the action type (e.g., do NOT say "forehand", "serve", "backhand", "volley", "slice", "smash", etc.).
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
    return instructions[:6]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['label', 'augment'], required=True,
                        help='label: generate labels from videos. augment: rephrase existing labels.')
    args = parser.parse_args()

    if args.mode == 'label':
        run_labeling()
    elif args.mode == 'augment':
        run_augmentation()

