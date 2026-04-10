"""
Relabel dataset.json with action-agnostic coaching instructions.

Instead of telling the LLM which action type the video shows,
we ask it to focus purely on motion differences between beginner and expert.
This prevents the downstream CoachMe model from learning
"video → action classification → template" instead of
"motion difference → coaching instruction".

Usage:
    python data_relabeling.py [--dry-run] [--start INDEX] [--limit N]
"""

from google import genai
import json
import os
import argparse
import time
import copy

PROMPT_TEMPLATE = """You are a professional sports coach analyzing two skeleton-based motion videos.

The FIRST video shows a learner performing a movement.
The SECOND video shows an expert performing the same movement.

Compare the two motions and generate EXACTLY 3 coaching instructions for the learner.

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

Output format: Return EXACTLY 3 instructions, one per line, numbered 1-3. Nothing else."""


def parse_response(text):
    """Parse numbered response into list of 3 instructions."""
    lines = text.strip().split('\n')
    instructions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1.", "1)", "1:", etc.
        for prefix in ['1.', '2.', '3.', '1)', '2)', '3)', '1:', '2:', '3:']:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            # Ensure ends with period
            if not line.endswith('.'):
                line += '.'
            instructions.append(line)
    return instructions[:3]


def main():
    parser = argparse.ArgumentParser(description='Relabel dataset with action-agnostic instructions')
    parser.add_argument('--dry-run', action='store_true', help='Print prompt for first entry without calling API')
    parser.add_argument('--start', type=int, default=0, help='Start index in dataset')
    parser.add_argument('--limit', type=int, default=None, help='Max entries to process')
    parser.add_argument('--api-key', type=str, default=None, help='Gemini API key (or set GEMINI_API_KEY env)')
    parser.add_argument('--dataset', type=str, default='./dataset.json', help='Path to dataset.json')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: overwrite input)')
    parser.add_argument('--backup', action='store_true', default=True, help='Create backup before overwriting')
    args = parser.parse_args()

    dataset_path = args.dataset
    output_path = args.output or dataset_path

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} entries from {dataset_path}")

    if args.dry_run:
        entry = dataset[args.start]
        print(f"\n--- Dry run for entry {args.start} ---")
        print(f"Beginner: {entry['beginner_video_name']}")
        print(f"Expert: {entry['expert_video_name']}")
        print(f"Motion type: {entry['motion_type']}")
        print(f"Current labels: {entry['labels']}")
        print(f"\nPrompt that would be sent:\n{PROMPT_TEMPLATE}")
        return

    # Backup
    if args.backup and output_path == dataset_path:
        backup_path = dataset_path.replace('.json', '_backup_before_relabel.json')
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                json.dump(dataset, f, indent=4)
            print(f"Backup saved to {backup_path}")

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print("ERROR: No API key. Set GEMINI_API_KEY env or pass --api-key")
        return

    client = genai.Client(api_key=api_key)

    # Determine video directory base
    # Videos are .avi files in preprocessed directories
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(args.dataset)), 'label_csv')

    end = min(args.start + args.limit, len(dataset)) if args.limit else len(dataset)

    success_count = 0
    fail_count = 0

    for i in range(args.start, end):
        entry = dataset[i]
        beginner_name = entry['beginner_video_name']
        expert_name = entry['expert_video_name']

        print(f"\n[{i+1}/{end}] Processing {beginner_name} vs {expert_name}")

        # Check if already relabeled (has 'labels_v2' field)
        if 'labels_v2' in entry and entry['labels_v2']:
            print(f"  Skipping: already has labels_v2")
            continue

        try:
            # Find video files - look in preprocessed directories
            motion_type = entry['motion_type']
            base_dir = '/workspace/dataset/VIDEO_RGB'

            # Map motion_type to directory name
            motion_type_to_dir = {
                "backhand": "backhand",
                "backhand2h": "backhand2hands",
                "bslice": "backhand_slice",
                "bvolley": "backhand_volley",
                "foreflat": "forehand_flat",
                "foreopen": "forehand_openstands",
                "fslice": "forehand_slice",
                "fvolley": "forehand_volley",
                "serflat": "flat_service",
                "serkick": "kick_service",
                "serslice": "slice_service",
                "smash": "smash",
            }

            dir_name = motion_type_to_dir.get(motion_type, motion_type)

            # Beginner video: .npy name -> .avi name
            beginner_avi = beginner_name.replace('.npy', '.avi')
            expert_avi = expert_name.replace('.npy', '.avi')

            beginner_path = os.path.join(base_dir, dir_name, 'beginner', beginner_avi)
            expert_path = os.path.join(base_dir, dir_name, 'experts', expert_avi)

            if not os.path.exists(beginner_path):
                print(f"  WARNING: beginner video not found: {beginner_path}")
                fail_count += 1
                continue
            if not os.path.exists(expert_path):
                print(f"  WARNING: expert video not found: {expert_path}")
                fail_count += 1
                continue

            # Upload videos
            beginner_video = client.files.upload(file=beginner_path)
            expert_video = client.files.upload(file=expert_path)

            # Wait for processing
            for _ in range(60):  # max 5 min wait
                beginner_video = client.files.get(name=beginner_video.name)
                expert_video = client.files.get(name=expert_video.name)
                if (beginner_video.state.name != "PROCESSING" and
                    expert_video.state.name != "PROCESSING"):
                    break
                time.sleep(5)

            if beginner_video.state.name == "FAILED" or expert_video.state.name == "FAILED":
                print(f"  ERROR: Video processing failed")
                fail_count += 1
                continue

            # Generate instructions
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    beginner_video,
                    expert_video,
                    PROMPT_TEMPLATE,
                ]
            )

            instructions = parse_response(response.text)
            if len(instructions) < 3:
                print(f"  WARNING: Only got {len(instructions)} instructions, expected 3")
                print(f"  Raw response: {response.text[:200]}")

            # Store as labels_v2 (keep original labels intact)
            entry['labels_v2'] = instructions
            success_count += 1
            print(f"  OK: {len(instructions)} instructions generated")

            # Save after each entry
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=4)

            # Cleanup uploaded files
            try:
                client.files.delete(name=beginner_video.name)
                client.files.delete(name=expert_video.name)
            except Exception:
                pass

            # Rate limiting
            time.sleep(2)

        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1
            time.sleep(5)

    print(f"\n--- Done ---")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    main()
