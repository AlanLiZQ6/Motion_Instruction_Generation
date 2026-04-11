"""
Data augmentation: rephrase each sample's first label 5 times using Gemini.
Mimics CoachMe's GPT-4 augmentation strategy — all 6 labels per sample
are semantically consistent (same meaning, different wording).

Result: dataset.json labels become [original, rephrase1, ..., rephrase5]
"""

from google import genai
import json
import os
import time
import re

# Set the API key
gemini_model = genai.Client(api_key="")

# Load dataset.json
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

SYSTEM_PROMPT = """You are an experienced tennis coach who specializes in helping students improve their tennis skills.
Your task is to rephrase the instruction.
Please follow this guideline when rewriting:
1. Use simple and clear language that beginners can easily understand and apply.
2. Maintain a clear and neutral tone with a professional and objective style.
3. Feel free to omit parts of the original instruction that are not particularly helpful.
4. Avoid phrasing that sounds too strict or overly commanding.
5. Focus on offering constructive suggestions that help students feel motivated to improve."""

USER_TEMPLATE = """If the target instruction does not begin with a directive verb such as "Keep", "Try", "Focus on", "Ensure", "Consider", "Aim", "Avoid", "Make sure", or "Remember", you should avoid introducing one in the rephrased version. Maintain a neutral tone. Do not introduce any information that is not present in the target instruction.
Please provide exactly one alternative way to rephrase the instruction.
Do NOT name or classify the action type (e.g., do NOT say "forehand", "serve", "backhand", "volley", "slice", "smash", etc.).
Target instruction: {instruction}"""


def rephrase_instruction(instruction):
    """Generate one rephrase of the given instruction."""
    user_msg = USER_TEMPLATE.format(instruction=instruction)
    response = gemini_model.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_msg}]}
        ]
    )
    result = response.text.strip()
    # Clean up: remove quotes, numbering, etc.
    result = re.sub(r'^[\"\']|[\"\']$', '', result.strip())
    result = re.sub(r'^\d+[\.\)\:]\s*', '', result.strip())
    if not result.endswith('.'):
        result += '.'
    return result


def main():
    total = len(dataset)
    skipped = 0
    processed = 0

    for i, entry in enumerate(dataset):
        beginner_id = entry["beginner_video_name"]

        # Check if already augmented (has 'augmented_labels' with 5 entries)
        if "augmented_labels" in entry and len(entry.get("augmented_labels", [])) >= 5:
            skipped += 1
            continue

        # Take the first label as the original
        original_label = entry["labels"][0]
        augmented = []

        max_retries = 5
        success = True
        for rep_idx in range(5):
            for attempt in range(max_retries):
                try:
                    rephrased = rephrase_instruction(original_label)
                    augmented.append(rephrased)
                    time.sleep(1)  # Rate limiting
                    break
                except Exception as e:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Error on {beginner_id} rephrase {rep_idx+1} (attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        print(f"  Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  FAILED rephrase {rep_idx+1} for {beginner_id}")
                        success = False

        if success and len(augmented) == 5:
            # Keep only the first original label + 5 rephrases
            entry["labels"] = [original_label]
            entry["augmented_labels"] = augmented
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{total - skipped} (skipped {skipped})")
        else:
            print(f"  Incomplete augmentation for {beginner_id}, skipping")
            continue

        # Save progress every 10 entries
        if processed % 10 == 0:
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f, indent=4)

    # Final save
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Total: {total}")


if __name__ == '__main__':
    main()
