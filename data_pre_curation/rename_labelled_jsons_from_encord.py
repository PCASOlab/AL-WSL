import os
import json
import sys

def rename_json_files(folder_path, dry_run=True):
    """
    Renames all .json files in folder_path using the 'data_title' field from each file.
    
    Args:
        folder_path: Path to the folder containing JSON files.
        dry_run: If True, only prints what would be done without actually renaming.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
    if not json_files:
        print(f"No JSON files found in '{folder_path}'.")
        return

    renamed_count = 0
    errors = []

    for filename in json_files:
        filepath = os.path.join(folder_path, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            errors.append(f"Could not read {filename}: {e}")
            continue

        # Extract data_title (expecting the JSON to be a list with one dictionary)
        try:
            if isinstance(data, list) and len(data) > 0:
                data_title = data[0].get('data_title')
            elif isinstance(data, dict):
                data_title = data.get('data_title')
            else:
                raise ValueError("Unexpected JSON structure")
        except Exception as e:
            errors.append(f"Could not extract data_title from {filename}: {e}")
            continue

        if not data_title:
            errors.append(f"No data_title found in {filename}")
            continue

        # Build new filename: replace .mp4 (or any extension) with .json
        base_name = os.path.splitext(data_title)[0]  # removes .mp4, .avi, etc.
        new_filename = base_name + '.json'
        new_filepath = os.path.join(folder_path, new_filename)

        if os.path.exists(new_filepath) and new_filepath != filepath:
            errors.append(f"Target file {new_filename} already exists, skipping {filename}")
            continue

        if dry_run:
            print(f"[DRY RUN] Would rename: {filename} -> {new_filename}")
        else:
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1

    if dry_run:
        print(f"\nDry run complete. Would rename {len(json_files) - len(errors)} files.")
    else:
        print(f"\nRenamed {renamed_count} files. Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"  {err}")

if __name__ == "__main__":
    # Usage: python rename_script.py /path/to/folder [--dry-run] [--execute]
    if len(sys.argv) < 2:
        print("Usage: python rename_json_files.py <folder_path> [--dry-run] [--execute]")
        print("  --dry-run : preview changes without renaming (default)")
        print("  --execute : actually rename the files")
        sys.exit(1)

    folder = sys.argv[1]
    dry_run = True  # default safe mode

    if "--execute" in sys.argv:
        dry_run = False
    elif "--dry-run" in sys.argv:
        dry_run = True

    rename_json_files(folder, dry_run=dry_run)