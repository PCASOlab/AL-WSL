from encord import EncordUserClient
import json
from pathlib import Path

# -----------------------------
# 1. Authenticate via SSH key
# -----------------------------
user_client = EncordUserClient.create_with_ssh_private_key(
    ssh_private_key_path="encord-manasa_pub_key-private-key.ed25519"
)

# -----------------------------
# 2. Load project
# -----------------------------
PROJECT_ID = "50bdca23-1906-4de9-9cc2-4cb2aa1a7491"
project = user_client.get_project(PROJECT_ID)
print(f"Connected to project: {project.title}")

# -----------------------------
# 3. Get all label rows
# -----------------------------
label_rows = project.list_label_rows_v2()
print(f"Found {len(label_rows)} label rows.")

# -------------------------
# 4. Initialize label rows using bundle
# -----------------------------
with project.create_bundle() as bundle:
    for label_row in label_rows:
        label_row.initialise_labels(bundle=bundle)

# -----------------------------
# 5. Create output directory
# -----------------------------
OUTPUT_DIR = Path("testset")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# 6. Export each label row as separate JSON
# -----------------------------
for lr in label_rows:
    # Convert label row to dictionary
    lr_dict = lr.to_encord_dict()

    # Get video name and remove ".mp4" if present
    video_filename = lr.data_title
    if video_filename.lower().endswith(".mp4"):
        video_filename = video_filename[:-4]

    json_path = OUTPUT_DIR / f"{video_filename}.json"
    # Save JSON
    with open(json_path, "w") as f:
        json.dump(lr_dict, f, indent=4)

    print(f"Saved {json_path}")

