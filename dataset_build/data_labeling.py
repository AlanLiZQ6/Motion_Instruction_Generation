# This file will use the Gemini API to generate the label for each pair of the video.
# Citations:
# Gemini API: https://ai.google.dev/gemini-api/docs

from google import genai
import json
import os
import pandas as pd
import re
import sys
import time
import argparse
import signal


# Set the API key (reads from env var GEMINI_API_KEY)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("GEMINI_API_KEY need to be set at first", file=sys.stderr)
    sys.exit(1)
gemini_model = genai.Client(api_key=API_KEY)


# Load dataset.json
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
with open(dataset_path, 'r') as f:
    dataset = json.load(f)
dataset_lookup = {entry["beginner_video_name"]: entry for entry in dataset}
csv_dir = os.path.join(os.path.dirname(__file__), 'label_csv')
csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])


#Labeling: generate 1 coaching label from each video pair
LABELING_PROMPT = """Your job is to use the professional tennis coaching laugnage to compare and analyze the two motion videos.
The first video is from a tennis beginner, and the second video is from a tennis expert.
Compare the two motions and identify the SINGLE and the most essential motion difference, which can give the largest improvement to the learner's technique.
Then, you need to generate only 1 coaching instruction for the difference.

Here is the rules which you need to follow:
1. You only need to focus on the motion differences, such as body posture, joint angles, timing, weight transfer, rotation, balance, and any other similar to these.
2. Do not classify the tennis stroke type. For example, you should not use the word "forehand", "serve", "backhand", and any professional word to name or classify the player's action.
3. Describe the drawbacks of the player in the beginner video and give the advise about how to improve it.
4. Use concise coaching language.

Here is the Bad examples, and you should not generate these:
"To improve your forehand, rotate your torso..."  (names or classify the action)
"Your serve lacks power because..." (names or classify the action)
"For a better backhand slice..."  (names or classify the action)

Output format: Return exactly 1 instruction."""

#rephrase first label 5 times
AUGMENTATION_PROMPT = """You are an experienced tennis coach who specializes in helping players to improve their tennis skills.
Your task is to rephrase the given instruction in different ways and keeps the exact technical meaning.
Please follow this rules :

1. Use simple and clear language.
2. Maintain a neutral tone with a professional and objective style to be the output.
3. Keep all key technical information from the original, and you should not omit body parts, directions, or actions mentioned.
4. Do not introduce any information, body parts, or advice not present in the original videos.
5. Avoid use the words which sounds too strict or commanding.
6. Focus on offering constructive and encouraging suggestions, which can help students feel motivated to improve."""

AUGMENTATION_USER = """If the target instruction does not begin with a directive verb such as "Keep", "Try", "Focus on", "Ensure", "Consider", "Aim", "Avoid", "Make sure", or "Remember", you should avoid introducing one in the rephrased version.
Do not introduce any information which is not present in the target instruction.
Do not classify the tennis stroke type. For example, you should not use the word "forehand", "serve", "backhand", and any professional word to name or classify the player's action.
Please provide exactly 5 alternative sentences which are rephrased from the instruction, one per line, numbered 1-5.
Target instruction: {instruction}"""


def parse_response(text):
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

    def timeout_handler(signum, frame):
        raise TimeoutError("Gemini API call timed out after 60 seconds")

    combined_message = AUGMENTATION_PROMPT + "\n\n" + AUGMENTATION_USER.format(instruction=instruction)

    # Set 60 second timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    try:
        response = gemini_model.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                {"role": "user", "parts": [{"text": combined_message}]}
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
    
    #Generate 1 coaching label per video pair.
    for csv_file in csv_files:
        
        print(f"\n Load {csv_file}")
        file_index = pd.read_csv(os.path.join(csv_dir, csv_file))
        for index, row in file_index.iterrows():
            beginner_id = row["id"]
            
            # This is used for restore labeling after breaking
            if beginner_id in dataset_lookup and len(dataset_lookup[beginner_id]["labels"]) >= 1:
                print(f"Skip the {beginner_id}, it has already labeled")
                continue

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    beginner_video = gemini_model.files.upload(file=row["beginner_avi_path"])
                    expert_video = gemini_model.files.upload(file=row["inference_avi_path"])

                    while beginner_video.state.name == "PROCESSING" or expert_video.state.name == "PROCESSING":
                        print("Waiting for file to be processed.")
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
                            # clean the uploaded video
                            gemini_model.files.delete(name=beginner_video.name)
                            gemini_model.files.delete(name=expert_video.name)
                        except Exception:
                            pass

                        time.sleep(2)
                    else:
                        print(f"{beginner_id} is not found in dataset.json")
                    break

                except Exception as e:
                    print(f"  Error (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print(f"Retry 50 seconds")
                        time.sleep(50)
                    else:
                        print(f"Failing to generate the text, skipping {beginner_id} now")


def run_augmentation():
    
    #Rephrase first label 5 times for each sample.
    
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
        max_retries = 3
        success = True

        for attempt in range(max_retries):
            try:
                augmented = rephrase_batch(original_label)
                for idx, generated_response in enumerate(augmented):
                    print(f"Rephrase {idx+1}: {generated_response[:100]}")
                time.sleep(2)
                break
            except Exception as e:
                print(f" Error ")
                if attempt < max_retries - 1:
                    print(f"  Retry 50 seocnds")
                    time.sleep(50)
                else:
                    print(f"Failing to generate the text, skipping {beginner_id} now")
                    success = False

        if success and len(augmented) == 5:
            entry["labels"] = [original_label]
            entry["augmented_labels"] = augmented
            processed += 1
            print(f"Done! {processed} augmented so far.")
        else:
            print(f"Incomplete augmentation for {beginner_id}, skipping")
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
    parser.add_argument('--mode',
                        choices=['label', 'augment'],
                        required=True)
    args = parser.parse_args()

    if args.mode == 'label':
        run_labeling()
    elif args.mode == 'augment':
        run_augmentation()

