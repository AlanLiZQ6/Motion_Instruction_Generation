from google import genai
import json
import os
import pandas as pd
import sys
import time

# Set the API key
gemini_model = genai.Client(api_key="AIzaSyCgMb0uIZnt0SoH0iQoROKClHRG6AGZTsI")

# Load dataset.json
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Build lookup: beginner_video_name -> dataset entry
dataset_lookup = {entry["beginner_video_name"]: entry for entry in dataset}

csv_dir = os.path.join(os.path.dirname(__file__), 'label_csv')
csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

for csv_file in csv_files:
    print(f"\nProcessing {csv_file}...")
    file_index = pd.read_csv(os.path.join(csv_dir, csv_file))
    for index, row in file_index.iterrows():
        beginner_id = row["id"]
        if beginner_id in dataset_lookup and dataset_lookup[beginner_id]["labels"]:
            print(f"Skipping {beginner_id}: labels already exist")
            continue

        beginner_video = gemini_model.files.upload(file=row["beginner_avi_path"])
        expert_video = gemini_model.files.upload(file=row["inference_avi_path"])

        while beginner_video.state.name == "PROCESSING" and expert_video.state.name == "PROCESSING":
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
                """The first video shows a beginner performing a tennis
                flat serve. The second video shows an expert performing
                the same serve.

                As a professional tennis coach, compare the two and
                generate 2-3 sentences of specific coaching instruction
                for the beginner..."""
            ]
        )

        # Split response into individual sentences and append to labels
        if beginner_id in dataset_lookup:
            sentences = [s.strip() for s in response.text.split(". ") if s.strip()]
            sentences = [s if s.endswith(".") else s + "." for s in sentences]
            dataset_lookup[beginner_id]["labels"].extend(sentences)
            print(f"Added {len(sentences)} labels for {beginner_id}")

            # Save after each label to avoid losing progress
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f, indent=4)
        else:
            print(f"Warning: {beginner_id} not found in dataset.json")

