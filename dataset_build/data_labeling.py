from google import genai
import json
import os
import pandas as pd
import sys
import time

# Set the API key
gemini_model = genai.Client(api_key="AIzaSyDDKO-w2_vn9twNBEDhjE76YK5__u0207Y")

# Load dataset.json
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Build lookup: beginner_video_name -> dataset entry
dataset_lookup = {entry["beginner_video_name"]: entry for entry in dataset}

csv_dir = os.path.join(os.path.dirname(__file__), 'label_csv')
csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

PROMPT = """You are a professional sports coach analyzing two motion videos.

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


def parse_response(text):
    """Parse numbered response into list of instructions."""
    lines = text.strip().split('\n')
    instructions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1.", "1)", "1:", etc.
        import re
        line = re.sub(r'^\d+[\.\)\:]\s*', '', line)
        if line:
            if not line.endswith('.'):
                line += '.'
            instructions.append(line)
    return instructions[:6]


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
                        PROMPT,
                    ]
                )

                instructions = parse_response(response.text)
                if beginner_id in dataset_lookup:
                    dataset_lookup[beginner_id]["labels"] = instructions
                    print(f"Generated {len(instructions)} labels for {beginner_id}")

                    # Save after each entry to avoid losing progress
                    with open(dataset_path, 'w') as f:
                        json.dump(dataset, f, indent=4)

                    # Cleanup uploaded files
                    try:
                        gemini_model.files.delete(name=beginner_video.name)
                        gemini_model.files.delete(name=expert_video.name)
                    except Exception:
                        pass

                    time.sleep(2)
                else:
                    print(f"Warning: {beginner_id} not found in dataset.json")
                break  # success, exit retry loop

            except Exception as e:
                wait_time = 30 * (attempt + 1)
                print(f"  Error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  FAILED after {max_retries} attempts, skipping {beginner_id}")

