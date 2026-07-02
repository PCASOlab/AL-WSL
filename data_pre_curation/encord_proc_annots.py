
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import shutil

# Define the color mapping for each tool (same as your existing script)
categories = [
    'Grasper',      #0   
    'Bipolar',      #1    
    'Hook',         #2    
    'Scissors',     #3      
    'Clipper',      #4       
    'Irrigator',    #5    
    'SpecimenBag',  #6
]

category_colors = {
    'Grasper': (0, 0, 255),        # Blue (BGR format in OpenCV)
    'Bipolar': (0, 255, 0),        # Green
    'Hook': (255, 0, 0),           # Red
    'Scissors': (255, 255, 0),     
    'Clipper': (255, 0, 255),      # Magenta
    'Irrigator': (0, 165, 255),    # Orange (BGR: 0, 165, 255)
    'SpecimenBag': (128, 0, 128),   # Purple
}

def extract_video_from_annotation(annotation_file):
    """
    Extract video information from Encord annotation file
    """
    print(f"Extracting video info from {annotation_file}")
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
    
    # Extract video information - check multiple possible locations
    data_title = annotation_data.get('data_title', 'unknown')
    if data_title == 'unknown' and 'data_units' in annotation_data:
        # Try to get from data_units
        data_unit_key = list(annotation_data['data_units'].keys())[0]
        data_unit = annotation_data['data_units'][data_unit_key]
        data_title = data_unit.get('data_title', 'unknown')
    
    data_hash = annotation_data.get('data_hash', 'unknown')
    file_link = annotation_data.get('file_link', '')
    
    print(f"  Data title: {data_title}")
    print(f"  Data hash: {data_hash}")
    
    return data_title, data_hash, file_link

def convert_encord_to_cholec_format(encord_annotation):
    """
    Convert Encord annotation format to Cholec annotation format
    """
    cholec_annotations = {}
    
    try:
        print("Converting Encord to Cholec format...")
        
        # Check for the structure shown in your example
        if 'data_units' in encord_annotation:
            print("  Found data_units structure")
            data_unit_key = list(encord_annotation['data_units'].keys())[0]
            data_unit = encord_annotation['data_units'][data_unit_key]
            
            if 'labels' in data_unit:
                labels = data_unit['labels']
                
                # Process each frame
                for frame_key, frame_data in labels.items():
                    try:
                        frame_idx = int(frame_key)
                        cholec_annotations[str(frame_idx)] = {'objects': []}
                        
                        # Check if there are objects in this frame
                        if 'objects' in frame_data and isinstance(frame_data['objects'], list):
                            for obj_data in frame_data['objects']:
                                name = obj_data.get('name', 'unknown')
                                
                                # Check for polygon data
                                if 'polygon' in obj_data or 'polygons' in obj_data:
                                    polygons_list = []
                                    
                                    # Handle polygon format
                                    if 'polygon' in obj_data:
                                        polygon_data = obj_data['polygon']
                                        # Convert polygon dict to list format
                                        if isinstance(polygon_data, dict):
                                            # Convert {0: {x:, y:}, 1: {x:, y:}, ...} to list of [x, y] pairs
                                            polygon_points = []
                                            for key in sorted(polygon_data.keys()):
                                                point = polygon_data[key]
                                                if 'x' in point and 'y' in point:
                                                    polygon_points.append([point['x'], point['y']])
                                            if polygon_points:
                                                polygons_list.append(polygon_points)
                                    
                                    # Handle polygons format
                                    if 'polygons' in obj_data:
                                        polygons_data = obj_data['polygons']
                                        if isinstance(polygons_data, list):
                                            for poly in polygons_data:
                                                if isinstance(poly, list) and len(poly) > 0:
                                                    polygons_list.append(poly[0])  # Get the inner list
                                    
                                    if polygons_list:
                                        cholec_annotations[str(frame_idx)]['objects'].append({
                                            'name': name,
                                            'polygons': polygons_list
                                        })
                                        print(f"    Added {name} with {len(polygons_list)} polygon(s) in frame {frame_idx}")
                        
                    except Exception as e:
                        print(f"  Error processing frame {frame_key}: {e}")
                        continue
        
        # Also check for object_answers structure (older format)
        elif 'object_answers' in encord_annotation:
            print("  Found object_answers structure")
            for frame_key, frame_data in encord_annotation['object_answers'].items():
                try:
                    frame_idx = int(frame_key)
                    cholec_annotations[str(frame_idx)] = {'objects': []}
                    
                    for obj_data in frame_data:
                        name = obj_data.get('name', 'unknown')
                        # Extract polygon coordinates from Encord format
                        if 'boundingBox' in obj_data:
                            bbox = obj_data['boundingBox']
                            # Convert bounding box to polygon (approximate)
                            x = bbox.get('x', 0)
                            y = bbox.get('y', 0)
                            w = bbox.get('w', 0)
                            h = bbox.get('h', 0)
                            
                            polygon = [
                                [x, y],
                                [x + w, y],
                                [x + w, y + h],
                                [x, y + h]
                            ]
                            
                            cholec_annotations[str(frame_idx)]['objects'].append({
                                'name': name,
                                'polygons': [polygon]
                            })
                        elif 'polygon' in obj_data:
                            # Handle polygon annotations
                            polygon_data = obj_data['polygon']
                            cholec_annotations[str(frame_idx)]['objects'].append({
                                'name': name,
                                'polygons': [polygon_data]
                            })
                        elif 'instance' in obj_data and 'polygon' in obj_data['instance']:
                            # Another possible structure
                            polygon_data = obj_data['instance']['polygon']
                            cholec_annotations[str(frame_idx)]['objects'].append({
                                'name': name,
                                'polygons': [polygon_data]
                            })
                except Exception as e:
                    print(f"  Error processing frame {frame_key}: {e}")
                    continue
        
        elif 'label_row' in encord_annotation:
            print("  Found label_row structure")
            label_row = encord_annotation['label_row']
            
            if 'object_answers' in label_row:
                print("  Found object_answers in label_row")
                for frame_key, frame_data in label_row['object_answers'].items():
                    try:
                        frame_idx = int(frame_key)
                        cholec_annotations[str(frame_idx)] = {'objects': []}
                        
                        for obj_data in frame_data:
                            name = obj_data.get('name', 'unknown')
                            if 'polygon' in obj_data:
                                polygon_data = obj_data['polygon']
                                cholec_annotations[str(frame_idx)]['objects'].append({
                                    'name': name,
                                    'polygons': [polygon_data]
                                })
                    except Exception as e:
                        print(f"  Error processing frame {frame_key}: {e}")
                        continue
        
        print(f"  Converted annotations for {len(cholec_annotations)} frames")
        
        # Debug: print frame counts
        if cholec_annotations:
            print("  Frame annotation summary:")
            for frame_idx in range(min(29, max([int(k) for k in cholec_annotations.keys()] + [0]) + 1)):
                frame_key = str(frame_idx)
                if frame_key in cholec_annotations:
                    obj_count = len(cholec_annotations[frame_key].get('objects', []))
                    if obj_count > 0:
                        print(f"    Frame {frame_idx}: {obj_count} object(s)")
                        for obj in cholec_annotations[frame_key]['objects']:
                            print(f"      - {obj['name']}: {len(obj.get('polygons', []))} polygon(s)")
                else:
                    print(f"    Frame {frame_idx}: No annotations")
        else:
            print("  No annotations found in any frame")
        
        return cholec_annotations
    
    except Exception as e:
        print(f"Error converting annotation format: {e}")
        print(f"Annotation keys: {encord_annotation.keys() if isinstance(encord_annotation, dict) else 'Not a dict'}")
        import traceback
        traceback.print_exc()
        return {}

def create_mask_from_stacked_polygon(polygons, frame_shape):
    """
    Create a binary mask from stacked polygon coordinates with normalized values.
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    height, width = frame_shape[:2]
    
    if not polygons:
        return mask
    
    for polygon in polygons:
        try:
            if isinstance(polygon, list):
                if len(polygon) > 0:
                    # Handle nested lists
                    if isinstance(polygon[0], list):
                        # List of [x, y] pairs
                        scaled_points = []
                        for point in polygon:
                            if len(point) >= 2:
                                x = int(point[0] * width)
                                y = int(point[1] * height)
                                scaled_points.append([x, y])
                        
                        if len(scaled_points) >= 3:
                            pts = np.array([scaled_points], dtype=np.int32)
                            cv2.fillPoly(mask, pts, color=1)
                    else:
                        # Flat list [x0, y0, x1, y1, ...]
                        if len(polygon) >= 6:  # Need at least 3 points (6 values)
                            scaled_points = []
                            for i in range(0, len(polygon), 2):
                                if i + 1 < len(polygon):
                                    x = int(polygon[i] * width)
                                    y = int(polygon[i + 1] * height)
                                    scaled_points.append([x, y])
                            
                            if len(scaled_points) >= 3:
                                pts = np.array([scaled_points], dtype=np.int32)
                                cv2.fillPoly(mask, pts, color=1)
            else:
                # Try to convert to list
                polygon_list = list(polygon)
                if isinstance(polygon_list[0], list):
                    scaled_points = []
                    for point in polygon_list:
                        if len(point) >= 2:
                            x = int(point[0] * width)
                            y = int(point[1] * height)
                            scaled_points.append([x, y])
                    
                    if len(scaled_points) >= 3:
                        pts = np.array([scaled_points], dtype=np.int32)
                        cv2.fillPoly(mask, pts, color=1)
        
        except Exception as e:
            print(f"  Warning: Error creating polygon mask: {e}")
            print(f"  Polygon type: {type(polygon)}")
            print(f"  Polygon: {polygon}")
            continue
    
    return mask

def apply_masks_to_frame(frame, annotations, frame_idx):
    """Apply masks to a single frame based on annotations"""
    # Make sure frame is in BGR format (OpenCV default)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 3:
        # Already BGR (from cv2.imread)
        pass
    elif frame.shape[2] == 4:
        # RGBA to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    # Create a copy of the frame to apply masks
    overlay = frame.copy()
    mask_applied = False
    
    # Check if the frame has annotations
    frame_key = str(frame_idx)
    if frame_key in annotations:
        frame_annotations = annotations[frame_key]
        
        for obj in frame_annotations.get('objects', []):
            tool_name = obj.get('name', 'unknown')
            
            # Check if polygons exist
            if 'polygons' not in obj:
                continue
                
            polygons = obj['polygons']
            
            color = category_colors.get(tool_name, (255, 255, 255))
            mask = create_mask_from_stacked_polygon(polygons, frame.shape)
            
            # Apply the mask with the tool's color
            if mask.sum() > 0:  # Only apply if mask has non-zero pixels
                overlay[mask == 1] = color
                mask_applied = True
    
    if mask_applied:
        alpha = 0.5  # 50% opacity
        masked_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    else:
        # No masks applied, return original frame
        masked_frame = frame
    
    return masked_frame

def create_pkl_file_from_encord(clip_name, original_video_path, annotations, output_pkl_dir, max_frames=29):
    """
    Create PKL file from original video and Encord annotations
    """
    print(f"Creating PKL file for {clip_name}...")
    
    # Read original video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video {original_video_path}")
        return
    
    video_frames = []
    video_masks = []
    present_tools = set()
    
    frame_num = 0
    frames_extracted = 0
    while frame_num < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"  End of video reached at frame {frame_num}")
            break
        
        # Skip frames if needed (only extract every nth frame)
        # For now, extract all frames up to max_frames
        if frames_extracted < max_frames:
            # Convert BGR to RGB for storage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Initialize 7-channel mask (one channel per tool category)
            frame_mask = np.zeros((len(categories), *frame.shape[:2]), dtype=np.uint8)
            
            # Apply annotations if available
            frame_key = str(frames_extracted)
            if frame_key in annotations:
                frame_annotations = annotations[frame_key]
                
                for obj in frame_annotations.get('objects', []):
                    tool_name = obj.get('name', 'unknown')
                    if tool_name in categories and 'polygons' in obj:
                        tool_idx = categories.index(tool_name)
                        polygons = obj['polygons']
                        tool_mask = create_mask_from_stacked_polygon(polygons, frame.shape)
                        frame_mask[tool_idx] = tool_mask
                        present_tools.add(tool_name)
            
            video_frames.append(frame_rgb)
            video_masks.append(frame_mask)
            frames_extracted += 1
        
        frame_num += 1
    
    cap.release()
    
    if not video_frames:
        print(f"  Error: No frames processed for {clip_name}")
        return
    
    print(f"  Extracted {len(video_frames)} frames")
    print(f"  Tools present: {present_tools}")
    
    # Convert to numpy arrays
    video_frames = np.array(video_frames)  # Shape: (T, H, W, 3)
    video_masks = np.array(video_masks)    # Shape: (T, 7, H, W)
    
    # Transpose to (C, T, H, W) format
    video_frames = np.transpose(video_frames, (3, 0, 1, 2))  # (3, T, H, W)
    video_masks = np.transpose(video_masks, (1, 0, 2, 3))    # (7, T, H, W)
    
    label_dict = {category: 1 if category in present_tools else 0 for category in categories}
    
    # Create output dictionary
    data_dict = {
        'frames': video_frames.astype(np.uint8),
        'masks': video_masks.astype(np.uint8),
        'labels': label_dict
    }
    
    # Save as PKL file
    pkl_filename = f"{clip_name}.pkl"
    pkl_path = os.path.join(output_pkl_dir, pkl_filename)
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"  Saved PKL for {clip_name} with {len(video_frames[0])} frames and 7-channel masks")

def process_encord_annotations(annotation_dir, original_videos_dir, output_dir, round_number, gt_dir):
    """
    Process all Encord annotation files and create masked videos + PKL files
    """
    print(f"Processing Encord annotations from: {annotation_dir}")
    print(f"Original videos directory: {original_videos_dir}")
    print(f"Output base directory: {output_dir}")
    round_dir = os.path.join(output_dir, f"AL_round_{round_number}")
    after_encord = os.path.join(round_dir, "after_encord")
    os.makedirs(after_encord, exist_ok=True)
    # Define output directories
    output_frames_dir = os.path.join(after_encord, 'frame_sequence')
    output_masked_dir = os.path.join(after_encord, 'masked_framest')

    output_pkl_dir = gt_dir
    
    # Create output directories
    os.makedirs(output_frames_dir, exist_ok=True)
    os.makedirs(output_masked_dir, exist_ok=True)
    os.makedirs(output_pkl_dir, exist_ok=True)
    
    # Get all annotation files
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
    
    if not annotation_files:
        print("No annotation files found!")
        return
    
    # Process each annotation file
    for annotation_file in tqdm(annotation_files, desc="Processing annotations"):
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        try:
            # Extract video information
            data_title, data_hash, file_link = extract_video_from_annotation(annotation_path)
            clip_name = data_title.split('.')[0] if '.' in data_title else data_title
            
            # Load annotation data
            with open(annotation_path, 'r') as f:
                encord_data = json.load(f)
            
            # Convert to Cholec format
            cholec_annotations = convert_encord_to_cholec_format(encord_data)
            
            if not cholec_annotations:
                print(f"  Warning: No annotations converted for {clip_name}")
                continue
            
            # Find original video file
            original_video_path = None
            for video_file in os.listdir(original_videos_dir):
                if clip_name in video_file and video_file.endswith(('.mp4', '.avi', '.mov')):
                    original_video_path = os.path.join(original_videos_dir, video_file)
                    break
            
            if not original_video_path:
                print(f"  Warning: Original video not found for {clip_name}")
                # Try alternative naming
                alt_clip_name = f"clip_{clip_name}"
                for video_file in os.listdir(original_videos_dir):
                    if alt_clip_name in video_file and video_file.endswith(('.mp4', '.avi', '.mov')):
                        original_video_path = os.path.join(original_videos_dir, video_file)
                        print(f"  Found with alternative name: {alt_clip_name}")
                        break
            
            if not original_video_path:
                print(f"  Error: Could not find original video for {clip_name}")
                continue
            
            
            # Step 1: Extract frames
            video_frames_dir = os.path.join(output_frames_dir, clip_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(original_video_path)
            if not cap.isOpened():
                print(f"  Error: Could not open video {original_video_path}")
                continue
            
            frame_count = 0
            save_count = 0
            
            while cap.isOpened() and save_count < 29:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save every frame (or adjust as needed)
                if frame_count % 1 == 0:  # Save every frame
                    frame_filename = f"{save_count:05d}.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        save_count += 1
                    else:
                        print(f"  Warning: Failed to save frame {frame_filename}")
                
                frame_count += 1
            
            cap.release()
            
            if save_count == 0:
                print(f"  Error: No frames extracted for {clip_name}")
                continue
            
            # Step 2: Create masked frames
            video_masked_dir = os.path.join(output_masked_dir, clip_name)
            os.makedirs(video_masked_dir, exist_ok=True)
            
            masked_frames_count = 0
            for frame_num in range(0, min(29, save_count)):
                frame_filename = f"{frame_num:05d}.jpg"
                frame_path = os.path.join(video_frames_dir, frame_filename)
                
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        masked_frame = apply_masks_to_frame(frame, cholec_annotations, frame_num)
                        output_path = os.path.join(video_masked_dir, frame_filename)
                        success = cv2.imwrite(output_path, masked_frame)
                        if success:
                            masked_frames_count += 1
                        else:
                            print(f"  Warning: Failed to save masked frame {frame_filename}")
                    else:
                        print(f"  Warning: Could not read frame {frame_path}")
                else:
                    print(f"  Warning: Frame not found {frame_path}")
                        
            # Step 3: Create PKL file
            create_pkl_file_from_encord(clip_name, original_video_path, cholec_annotations, output_pkl_dir)
            
            print(f" Successfully processed {clip_name}")
            
        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
            import traceback
            traceback.print_exc()
            continue