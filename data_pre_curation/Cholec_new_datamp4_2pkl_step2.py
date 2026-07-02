# # import os
# # import json
# # import cv2
# # import numpy as np
# # from tqdm import tqdm
# # import pickle

# # # Define the color mapping for each tool
# # categories = [
# #     'Grasper',      #0   
# #     'Bipolar',      #1    
# #     'Hook',         #2    
# #     'Scissors',     #3      
# #     'Clipper',      #4       
# #     'Irrigator',    #5    
# #     'SpecimenBag',  #6                  
# # ]

# # category_colors = {
# #     'Grasper': (0, 0, 255),        # Blue
# #     'Bipolar': (0, 255, 0),        # Green
# #     'Hook': (255, 0, 0),           # Red
# #     'Scissors': (255, 255, 0),     # Yellow
# #     'Clipper': (255, 0, 255),      # Magenta
# #     'Irrigator': (255, 165, 0),    # Orange
# #     'SpecimenBag': (128, 0, 128)   # Purple
# # }

# # def create_mask_from_stacked_polygon(polygons, frame_shape):
# #     """
# #     Create a binary mask from stacked polygon coordinates with normalized values.
    
# #     Args:
# #         polygons: List of arrays, each containing normalized coordinates [x0,y0,x1,y1,x2,y2,...]
# #         frame_shape: Shape of the target frame (height, width, channels)
    
# #     Returns:
# #         Binary mask with the same height/width as frame_shape, where polygon regions are filled with 1
# #     """
# #     # Initialize empty mask with frame dimensions
# #     mask = np.zeros(frame_shape[:2], dtype=np.uint8)
# #     height, width = frame_shape[:2]
    
# #     # Process each polygon in the stack
# #     for polygon in polygons:
# #         # Convert the list to a numpy array if it's not already one
# #         if isinstance(polygon, list):
# #             polygon = np.array(polygon[0])
        
# #         # Reshape to get pairs of (x, y) coordinates
# #         points = polygon.reshape(-1, 2)
        
# #         # Scale normalized coordinates to image dimensions
# #         scaled_points = []
# #         for point in points:
# #             x = int(point[0] * width)
# #             y = int(point[1] * height)
# #             scaled_points.append([x, y])
        
# #         # Convert to numpy array in the format OpenCV expects
# #         pts = np.array([scaled_points], dtype=np.int32)
        
# #         # Draw filled polygon on the mask
# #         cv2.fillPoly(mask, pts, color=1)
    
# #     return mask

# # def apply_masks_to_frame(frame, annotations, frame_idx):
# #     """Apply masks to a single frame based on annotations"""
# #     # Convert frame to RGB if it's grayscale
# #     if len(frame.shape) == 2:
# #         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
# #     # Create a copy of the frame to apply masks
# #     overlay = frame.copy()
    
# #     # Check if the frame has annotations
# #     if str(frame_idx) in annotations:
# #         frame_annotations = annotations[str(frame_idx)]
        
# #         # Process each object in the frame
# #         for obj in frame_annotations.get('objects', []):
# #             tool_name = obj['name']
# #             polygons = obj['polygons']
            
# #             # Get the color for this tool
# #             color = category_colors.get(tool_name, (255, 255, 255))  # Default to white if not found
            
# #             # Create mask for this tool
# #             mask = create_mask_from_stacked_polygon(polygons, frame.shape)
            
# #             # Apply the mask with the tool's color
# #             overlay[mask == 1] = color
    
# #     alpha = 0.5  # 50% opacity
# #     masked_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
# #     return masked_frame

# # def generate_one_hot_instance_mask(num_instances, H, W, instance_masks):
# #     """
# #     Generate a one-hot mask for up to 7 instances.
# #     """
# #     max_instances = 7
# #     one_hot_mask = np.zeros((max_instances, H, W), dtype=np.uint8)

# #     for i in range(min(num_instances, max_instances)):
# #         one_hot_mask[i] = instance_masks[i]

# #     return one_hot_mask

# # def create_pkl_file(clip_name, frames_dir, annotations, output_pkl_dir):
# #     """
# #     Create a PKL file containing original frames and their 14-channel masks.
# #     """
# #     video_frames = []
# #     video_masks = []
# #     present_tools = set()
    
# #     for frame_num in range(0, 29):
# #         frame_filename = f"{frame_num:05d}.jpg"
# #         frame_path = os.path.join(frames_dir, clip_name, frame_filename)
        
# #         if not os.path.exists(frame_path):
# #             print(f"Warning: Missing frame {frame_filename} in {clip_name}")
# #             continue
            
# #         # Read frame and convert to RGB
# #         frame = cv2.imread(frame_path)
# #         # frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)

# #         if frame is None:
# #             continue
# #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
# #         # Initialize 14-channel mask (one channel per tool category)
# #         frame_mask = np.zeros((len(categories), *frame.shape[:2]), dtype=np.uint8)
        
# #         # Check if frame has annotations (using 1-based index in JSON)
# #         if str(frame_num) in annotations:
# #             frame_annotations = annotations[str(frame_num)]
            
# #             # Process each object in frame
# #             for obj in frame_annotations.get('objects', []):
# #                 tool_name = obj['name']
# #                 if tool_name in categories:
# #                     tool_idx = categories.index(tool_name)
# #                     polygons = obj['polygons']
# #                     tool_mask = create_mask_from_stacked_polygon(polygons, frame.shape)
# #                     frame_mask[tool_idx] = tool_mask
# #                     present_tools.add(tool_name)
# #         video_frames.append(frame)
# #         video_masks.append(frame_mask)
    
# #     if not video_frames:
# #         print(f"Error: No frames processed for {clip_name}")
# #         return
    
# #     # Convert lists to numpy arrays
# #     video_frames = np.array(video_frames)  # Shape: (T, H, W, 3)
# #     video_masks = np.array(video_masks)    # Shape: (T, 14, H, W)
    
# #     # Transpose to (C, T, H, W) format
# #     video_frames = np.transpose(video_frames, (3, 0, 1, 2))  # (3, T, H, W)
# #     video_masks = np.transpose(video_masks, (1, 0, 2, 3))    # (14, T, H, W)
# #     label_dict = {category: 1 if category in present_tools else 0 for category in categories}    
# #     # Create output dictionary
# #     data_dict = {
# #         'frames': video_frames.astype(np.uint8),
# #         'masks': video_masks.astype(np.uint8),
# #         'labels': label_dict
# #     }
    
# #     # Save as PKL file
# #     pkl_filename = f"{clip_name}.pkl"
# #     pkl_path = os.path.join(output_pkl_dir, pkl_filename)
    
# #     with open(pkl_path, 'wb') as f:
# #         pickle.dump(data_dict, f)
# #     print(f"Saved PKL for {clip_name} with {len(video_frames[0])} frames and 14-channel masks")

# # def has_annotations(clip_annotations):
# #     """Check if there are any annotations in any frame of the clip"""
# #     for frame_idx in range(0, 29):
# #         if str(frame_idx) in clip_annotations:
# #             frame_ann = clip_annotations[str(frame_idx)]
# #             if frame_ann.get('objects'):
# #                 return True
# #     return False

# # def process_clip(clip_name, annotations_data, frames_dir, output_dir, output_pkl_dir):
# #     """Process all frames for a single clip"""
# #     # Get the annotations for this clip
# #     clip_annotations = annotations_data[0]['data_units'][list(annotations_data[0]['data_units'].keys())[0]]['labels']
    
# #     # Check if the clip has any annotations
# #     if not has_annotations(clip_annotations):
# #         print(f"Skipping {clip_name} - no annotations found")
# #         return
    
# #     # Create output directory for this clip
# #     clip_output_dir = os.path.join(output_dir, clip_name)
# #     os.makedirs(clip_output_dir, exist_ok=True)
    
# #     # Process each frame to create masked images
# #     for frame_num in range(0, 29):
# #         frame_filename = f"{frame_num:05d}.jpg"
# #         frame_path = os.path.join(frames_dir, clip_name, frame_filename)
        
# #         if os.path.exists(frame_path):
# #             # Read the frame
# #             frame = cv2.imread(frame_path)
            
# #             # Apply masks
# #             masked_frame = apply_masks_to_frame(frame, clip_annotations, frame_num)
            
# #             # Save the masked frame
# #             output_path = os.path.join(clip_output_dir, frame_filename)
# #             cv2.imwrite(output_path, masked_frame)
    
# #     # Create PKL file for this clip
# #     create_pkl_file(clip_name, frames_dir, clip_annotations, output_pkl_dir)

# # def main():
# #     # Define paths
# #     # frames_base_dir = '/home/maniden/Desktop/WSL_Tool_classification/data/Cholec_new_selected/frame_sequence1/' # Directory containing clip folders with frames
# #     # annotations_dir = '/home/maniden/Desktop/WSL_Tool_classification/data/gt_videos/'  # Directory containing JSON annotation files
# #     # output_dir = '/home/maniden/Desktop/WSL_Tool_classification/data/Cholec_new_selected/masked_frames1/' # Output directory for masked frames
# #     # output_pkl_dir = '/home/maniden/Desktop/WSL_Tool_classification/data/Cholec_new_selected/gt_pk1l/'  # Output directory for PKL files
# #     frames_base_dir = "/home/maniden/Desktop/WSL_Tool_classification/data/AL_run/frame_sequence"
# #     annotations_dir = "/home/maniden/data/output/temporal_consistent/CholecDINOv3/active_learning/round_1_selected"
# #     output_dir = "/home/maniden/Desktop/WSL_Tool_classification/data/masked_framest"
# #     output_pkl_dir = "/home/maniden/Desktop/WSL_Tool_classification/data/cholec_pkl"
# #     # Create output directories
# #     os.makedirs(output_dir, exist_ok=True)
# #     os.makedirs(output_pkl_dir, exist_ok=True)
    
# #     # Process each clip
# #     for clip_json in tqdm(os.listdir(annotations_dir)):
# #         if clip_json.endswith('.json'):
# #             json_path = os.path.join(annotations_dir, clip_json)
# #             with open(json_path, 'r') as f:
# #                 annotations_data = json.load(f)
# #             video_full_name = annotations_data[0]['data_title']
# #             clip_name = video_full_name.split('.')[0]
            
# #             # Check if the corresponding frames directory exists
# #             clip_frames_dir = os.path.join(frames_base_dir, clip_name)
# #             if os.path.exists(clip_frames_dir):
# #                 process_clip(clip_name, annotations_data, frames_base_dir, output_dir, output_pkl_dir)

# # if __name__ == "__main__":
# #     main()

# import os
# import json
# import cv2
# import numpy as np
# from tqdm import tqdm
# import pickle

# # Define the color mapping for each tool
# categories = [
#     'Grasper',      #0   
#     'Bipolar',      #1    
#     'Hook',         #2    
#     'Scissors',     #3      
#     'Clipper',      #4       
#     'Irrigator',    #5    
#     'SpecimenBag',  #6                  
# ]

# category_colors = {
#     'Grasper': (0, 0, 255),        # Blue
#     'Bipolar': (0, 255, 0),        # Green
#     'Hook': (255, 0, 0),           # Red
#     'Scissors': (255, 255, 0),     # Yellow
#     'Clipper': (255, 0, 255),      # Magenta
#     'Irrigator': (255, 165, 0),    # Orange
#     'SpecimenBag': (128, 0, 128)   # Purple
# }

# def create_mask_from_stacked_polygon(polygons, frame_shape):
#     """
#     Create a binary mask from stacked polygon coordinates with normalized values.
    
#     Args:
#         polygons: List of arrays, each containing normalized coordinates [x0,y0,x1,y1,x2,y2,...]
#         frame_shape: Shape of the target frame (height, width, channels)
    
#     Returns:
#         Binary mask with the same height/width as frame_shape, where polygon regions are filled with 1
#     """
#     # Initialize empty mask with frame dimensions
#     mask = np.zeros(frame_shape[:2], dtype=np.uint8)
#     height, width = frame_shape[:2]
#     if not polygons or len(polygons)==0:
#         return mask 
    
#     # Process each polygon in the stack
#     for polygon in polygons:
#         if polygon is None or len(polygon)==0:
#             continue
#         # Convert the list to a numpy array if it's not already one
#         if isinstance(polygon, list):
#             polygon = np.array(polygon[0])
        
#         # Reshape to get pairs of (x, y) coordinates
#         points = polygon.reshape(-1, 2)
        
#         # Scale normalized coordinates to image dimensions
#         scaled_points = []
#         for point in points:
#             x = int(point[0] * width)
#             y = int(point[1] * height)
#             scaled_points.append([x, y])
        
#         # Convert to numpy array in the format OpenCV expects
#         pts = np.array([scaled_points], dtype=np.int32)
        
#         # Draw filled polygon on the mask
#         cv2.fillPoly(mask, pts, color=1)
    
#     return mask

# def apply_masks_to_frame(frame, annotations, frame_idx):
#     """Apply masks to a single frame based on annotations"""
#     # Convert frame to RGB if it's grayscale
#     if len(frame.shape) == 2:
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
#     # Create a copy of the frame to apply masks
#     overlay = frame.copy()
    
#     # Check if the frame has annotations
#     if str(frame_idx) in annotations:
#         frame_annotations = annotations[str(frame_idx)]
        
#         # Process each object in the frame
#         for obj in frame_annotations.get('objects', []):
#             tool_name = obj['name']
#             polygons = obj['polygons']
            
#             # Get the color for this tool
#             color = category_colors.get(tool_name, (255, 255, 255))  # Default to white if not found
            
#             # Create mask for this tool
#             mask = create_mask_from_stacked_polygon(polygons, frame.shape)
            
#             # Apply the mask with the tool's color
#             overlay[mask == 1] = color
    
#     alpha = 0.5  # 50% opacity
#     masked_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
#     return masked_frame

# def create_pkl_file(clip_name, frames_dir, annotations, output_pkl_dir):
#     """
#     Create a PKL file containing original frames and their 7-channel masks.
#     """
#     video_frames = []
#     video_masks = []
#     present_tools = set()
    
#     for frame_num in range(0, 29):
#         frame_filename = f"{frame_num:05d}.jpg"
#         frame_path = os.path.join(frames_dir, clip_name, frame_filename)
        
#         if not os.path.exists(frame_path):
#             print(f"Warning: Missing frame {frame_filename} in {clip_name}")
#             continue
            
#         # Read frame and convert to RGB
#         frame = cv2.imread(frame_path)
        
#         if frame is None:
#             continue
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Initialize 7-channel mask (one channel per tool category)
#         frame_mask = np.zeros((len(categories), *frame.shape[:2]), dtype=np.uint8)
        
#         # Check if frame has annotations
#         if str(frame_num) in annotations:
#             frame_annotations = annotations[str(frame_num)]
            
#             # Process each object in frame
#             for obj in frame_annotations.get('objects', []):
#                 tool_name = obj['name']
#                 if tool_name in categories:
#                     tool_idx = categories.index(tool_name)
#                     polygons = obj['polygons']
#                     tool_mask = create_mask_from_stacked_polygon(polygons, frame.shape)
#                     frame_mask[tool_idx] = tool_mask
#                     present_tools.add(tool_name)
        
#         video_frames.append(frame)
#         video_masks.append(frame_mask)
    
#     if not video_frames:
#         print(f"Error: No frames processed for {clip_name}")
#         return
    
#     # Convert lists to numpy arrays
#     video_frames = np.array(video_frames)  # Shape: (T, H, W, 3)
#     video_masks = np.array(video_masks)    # Shape: (T, 7, H, W)
    
#     # Transpose to (C, T, H, W) format
#     video_frames = np.transpose(video_frames, (3, 0, 1, 2))  # (3, T, H, W)
#     video_masks = np.transpose(video_masks, (1, 0, 2, 3))    # (7, T, H, W)
    
#     label_dict = {category: 1 if category in present_tools else 0 for category in categories}
    
#     # Create output dictionary
#     data_dict = {
#         'frames': video_frames.astype(np.uint8),
#         'masks': video_masks.astype(np.uint8),
#         'labels': label_dict
#     }
    
#     # Save as PKL file
#     pkl_filename = f"{clip_name}.pkl"
#     pkl_path = os.path.join(output_pkl_dir, pkl_filename)
    
#     with open(pkl_path, 'wb') as f:
#         pickle.dump(data_dict, f)
    
#     print(f"Saved PKL for {clip_name} with {len(video_frames[0])} frames and 7-channel masks")

# # def has_annotations(clip_annotations):
# #     """Check if there are any annotations in any frame of the clip"""
# #     if not isinstance(clip_annotations, dict):
# #         return False
    
# #     for frame_idx in range(0, 29):
# #         if str(frame_idx) in clip_annotations:
# #             frame_ann = clip_annotations[str(frame_idx)]
# #             if isinstance(frame_ann, dict) and frame_ann.get('objects'):
# #                 return True
# #     return False

# def load_annotations(json_path):
#     """Load and parse annotations from JSON file"""
#     try:
#         with open(json_path, 'r') as f:
#             annotations_data = json.load(f)
#     except Exception as e:
#         print(f"Error loading JSON {json_path}: {e}")
#         return {}
#     # with open(json_path, 'r') as f:
#     #     annotations_data = json.load(f)
    
#     # Check if annotations_data is a list or dictionary
#     if isinstance(annotations_data, list):
#         if annotations_data:
#             # Try different possible structures
#             for item in annotations_data:
#                 if isinstance(item, dict):
#                     # Check for different possible structures in the item
#                     if 'data_units' in item:
#                         data_unit_key = list(item['data_units'].keys())[0]
#                         return item['data_units'][data_unit_key].get('labels', {})
#                     elif 'labels' in item:
#                         return item['labels']
#                     elif 'data_title' in item:
#                         # Try to find data_units if not at top level
#                         if 'data_units' in item:
#                             data_unit_key = list(item['data_units'].keys())[0]
#                             return item['data_units'][data_unit_key].get('labels', {})
    
#     elif isinstance(annotations_data, dict):
#         # If it's a dictionary, check for common structures
#         if 'data_units' in annotations_data:
#             data_unit_key = list(annotations_data['data_units'].keys())[0]
#             return annotations_data['data_units'][data_unit_key].get('labels', {})
#         elif 'labels' in annotations_data:
#             return annotations_data['labels']
#         elif 'annotations' in annotations_data:
#             return annotations_data['annotations']
#         elif 'data_title' in annotations_data:
#             # This might be a top-level item with data_units inside
#             if 'data_units' in annotations_data:
#                 data_unit_key = list(annotations_data['data_units'].keys())[0]
#                 return annotations_data['data_units'][data_unit_key].get('labels', {})
    
#     print(f"Warning: Unexpected JSON structure in {json_path}")
#     return {}

# def get_clip_name_from_json(json_path):
#     """Extract clip name from JSON file"""
#     with open(json_path, 'r') as f:
#         annotations_data = json.load(f)
    
#     # Check different possible structures
#     if isinstance(annotations_data, list) and annotations_data:
#         item = annotations_data[0]
#         if isinstance(item, dict) and 'data_title' in item:
#             video_full_name = item['data_title']
#             return video_full_name.split('.')[0]
    
#     elif isinstance(annotations_data, dict):
#         if 'data_title' in annotations_data:
#             video_full_name = annotations_data['data_title']
#             return video_full_name.split('.')[0]
    
#     # Fallback: use filename
#     filename = os.path.basename(json_path)
#     clip_name = os.path.splitext(filename)[0].replace('clip_', '')
#     return clip_name

# def process_clip(clip_name, json_path, frames_dir, output_dir, output_pkl_dir):
#     """Process all frames for a single clip"""
#     # Load the JSON annotations
#     clip_annotations = load_annotations(json_path)
    
#     # # Check if the clip has any annotations
#     # if not has_annotations(clip_annotations):
#     #     print(f"Skipping {clip_name} - no annotations found")
#     #     return
    
#     # Create output directory for this clip
#     clip_output_dir = os.path.join(output_dir, clip_name)
#     os.makedirs(clip_output_dir, exist_ok=True)
    
#     # Process each frame to create masked images
#     for frame_num in range(0, 29):
#         frame_filename = f"{frame_num:05d}.jpg"
#         frame_path = os.path.join(frames_dir, clip_name, frame_filename)
        
#         if os.path.exists(frame_path):
#             # Read the frame
#             frame = cv2.imread(frame_path)
            
#             if frame is not None:
#                 # Apply masks
#                 masked_frame = apply_masks_to_frame(frame, clip_annotations, frame_num)
                
#                 # Save the masked frame
#                 output_path = os.path.join(clip_output_dir, frame_filename)
#                 cv2.imwrite(output_path, masked_frame)
    
#     # Create PKL file for this clip
#     create_pkl_file(clip_name, frames_dir, clip_annotations, output_pkl_dir)

# def main():
#     # Define paths
#     # selected_folder= '/home/maniden/data/cholec80/70trainvids/'
#     frames_base_dir = '/home/maniden/Desktop/WSL_Tool_classification/data/70trainannots_frames'
#     # frames_base_dir = "/home/maniden/Desktop/WSL_Tool_classification/data/AL_run/frame_sequence"
#     annotations_dir = "/home/maniden/Desktop/WSL_Tool_classification/train70annots"
#     output_dir = "/home/maniden/Desktop/WSL_Tool_classification/data/train70annots/masked_frames"
#     output_pkl_dir = "/home/maniden/Desktop/WSL_Tool_classification/data/train70annots/cholec_pkl"
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(output_pkl_dir, exist_ok=True)
    
#     # Get all JSON files
#     json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
#     # Debug: Print structure of first JSON file
#     if json_files:
#         first_json = os.path.join(annotations_dir, json_files[0])
#         print(f"Sample JSON structure from {json_files[0]}:")
#         with open(first_json, 'r') as f:
#             sample_data = json.load(f)
#         print(f"  Type: {type(sample_data)}")
#         if isinstance(sample_data, list):
#             print(f"  List length: {len(sample_data)}")
#             if sample_data:
#                 print(f"  First item type: {type(sample_data[0])}")
#                 if isinstance(sample_data[0], dict):
#                     print(f"  First item keys: {sample_data[0].keys()}")
#         elif isinstance(sample_data, dict):
#             print(f"  Dictionary keys: {sample_data.keys()}")
#         print()
    
#     # Process each clip
#     for clip_json in tqdm(json_files, desc="Processing clips"):
#         json_path = os.path.join(annotations_dir, clip_json)
        
#         # Get clip name
#         clip_name = get_clip_name_from_json(json_path)
        
#         # Check if the corresponding frames directory exists
#         clip_frames_dir = os.path.join(frames_base_dir, clip_name)
#         if os.path.exists(clip_frames_dir):
#             print(f"Processing {clip_name}...")
#             process_clip(clip_name, json_path, frames_base_dir, output_dir, output_pkl_dir)
#         else:
#             print(f"Warning: Frames directory not found for {clip_name}")
#             # Try alternative naming: clip_{name}
#             alt_clip_name = f"clip_{clip_name}"
#             alt_clip_frames_dir = os.path.join(frames_base_dir, alt_clip_name)
#             if os.path.exists(alt_clip_frames_dir):
#                 print(f"  Found with alternative name: {alt_clip_name}")
#                 process_clip(alt_clip_name, json_path, frames_base_dir, output_dir, output_pkl_dir)

# if __name__ == "__main__":
#     main()
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import pickle

# Define the color mapping for each tool
categories = [
    'Grasper',      #0   
    'Bipolar',      #1    
    'Hook',         #2    
    'Scissors',     #3      
    'Clipper',      #4       
    'Irrigator',    #5    
    'SpecimenBag',  #6
    # 'SelectedFrame', #7               
]

category_colors = {
    'Grasper': (0, 0, 255),        # Blue (BGR format in OpenCV)
    'Bipolar': (0, 255, 0),        # Green
    'Hook': (255, 0, 0),           # Red
    'Scissors': (255, 255, 0),     
    'Clipper': (255, 0, 255),      # Magenta
    'Irrigator': (0, 165, 255),    # Orange (BGR: 0, 165, 255)
    'SpecimenBag': (128, 0, 128),   # Purple
    # 'SelectedFrame': (255, 20, 147)
}

def create_mask_from_stacked_polygon(polygons, frame_shape):
    """
    Create a binary mask from stacked polygon coordinates with normalized values.
    
    Args:
        polygons: List of arrays, each containing normalized coordinates [x0,y0,x1,y1,x2,y2,...]
        frame_shape: Shape of the target frame (height, width, channels)
    
    Returns:
        Binary mask with the same height/width as frame_shape, where polygon regions are filled with 1
    """
    # Initialize empty mask with frame dimensions
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    # If no polygons or empty list, return empty mask
    if not polygons or len(polygons) == 0:
        return mask
    
    height, width = frame_shape[:2]
    
    # Process each polygon in the stack
    for polygon in polygons:
        # Skip if polygon is None or empty
        if polygon is None or len(polygon) == 0:
            continue
            
        # Convert the list to a numpy array if it's not already one
        if isinstance(polygon, list):
            if len(polygon) == 0:
                continue
            polygon = np.array(polygon[0])
        
        # Reshape to get pairs of (x, y) coordinates
        points = polygon.reshape(-1, 2)
        
        # Scale normalized coordinates to image dimensions
        scaled_points = []
        for point in points:
            x = int(point[0] * width)
            y = int(point[1] * height)
            scaled_points.append([x, y])
        
        # Convert to numpy array in the format OpenCV expects
        pts = np.array([scaled_points], dtype=np.int32)
        
        # Draw filled polygon on the mask
        cv2.fillPoly(mask, pts, color=1)
    
    return mask

def apply_masks_to_frame(frame, annotations, frame_idx):
    """Apply masks to a single frame based on annotations"""
    # Convert frame to RGB if it's grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Create a copy of the frame to apply masks
    overlay = frame.copy()
    
    # Check if the frame has annotations
    if isinstance(annotations, dict) and str(frame_idx) in annotations:
        frame_annotations = annotations[str(frame_idx)]
        
        # Process each object in the frame if there are objects
        if isinstance(frame_annotations, dict) and 'objects' in frame_annotations:
            for obj in frame_annotations.get('objects', []):
                tool_name = obj.get('name', '')
                polygons = obj.get('polygons', [])
                
                # Get the color for this tool
                color = category_colors.get(tool_name, (255, 255, 255))  # Default to white if not found
                
                # Create mask for this tool
                mask = create_mask_from_stacked_polygon(polygons, frame.shape)
                
                # Apply the mask with the tool's color
                overlay[mask == 1] = color
    
    alpha = 0.5  # 50% opacity
    masked_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return masked_frame

def create_pkl_file(clip_name, frames_dir, annotations, output_pkl_dir):
    """
    Create a PKL file containing original frames and their 7-channel masks.
    Always creates masks even when no annotations are found (empty masks).
    """
    video_frames = []
    video_masks = []
    present_tools = set()
    
    for frame_num in range(0, 29):
        frame_filename = f"{frame_num:05d}.jpg"
        frame_path = os.path.join(frames_dir, clip_name, frame_filename)
        
        if not os.path.exists(frame_path):
            print(f"Warning: Missing frame {frame_filename} in {clip_name}")
            # Create empty frame if missing
            continue
            
        # Read frame and convert to RGB
        frame = cv2.imread(frame_path)
        
        if frame is None:
            # Create empty frame if can't read
            continue
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize 7-channel mask (one channel per tool category)
        frame_mask = np.zeros((len(categories), *frame.shape[:2]), dtype=np.uint8)
        
        # Check if frame has annotations
        if isinstance(annotations, dict) and str(frame_num) in annotations:
            frame_annotations = annotations[str(frame_num)]
            
            # Process each object in frame if there are objects
            if isinstance(frame_annotations, dict) and 'objects' in frame_annotations:
                for obj in frame_annotations.get('objects', []):
                    tool_name = obj.get('name', '')
                    if tool_name in categories:
                        tool_idx = categories.index(tool_name)
                        polygons = obj.get('polygons', [])
                        tool_mask = create_mask_from_stacked_polygon(polygons, frame.shape)
                        frame_mask[tool_idx] = tool_mask
                        present_tools.add(tool_name)
        
        video_frames.append(frame)
        video_masks.append(frame_mask)
    
    if not video_frames:
        print(f"Error: No frames processed for {clip_name}")
        return
    
    # Convert lists to numpy arrays
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
    
    print(f"Saved PKL for {clip_name} with {len(video_frames[0])} frames and 7-channel masks")

def load_annotations(json_path):
    """Load and parse annotations from JSON file. Returns empty dict if no annotations."""
    try:
        with open(json_path, 'r') as f:
            annotations_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return {}
    
    # Check if annotations_data is a list or dictionary
    if isinstance(annotations_data, list):
        if annotations_data:
            # Try different possible structures
            for item in annotations_data:
                if isinstance(item, dict):
                    # Check for different possible structures in the item
                    if 'data_units' in item:
                        data_unit_key = list(item['data_units'].keys())[0]
                        labels = item['data_units'][data_unit_key].get('labels', {})
                        if labels:
                            return labels
                    elif 'labels' in item:
                        labels = item['labels']
                        if labels:
                            return labels
                    elif 'data_title' in item:
                        # Try to find data_units if not at top level
                        if 'data_units' in item:
                            data_unit_key = list(item['data_units'].keys())[0]
                            labels = item['data_units'][data_unit_key].get('labels', {})
                            if labels:
                                return labels
    
    elif isinstance(annotations_data, dict):
        # If it's a dictionary, check for common structures
        if 'data_units' in annotations_data:
            data_unit_key = list(annotations_data['data_units'].keys())[0]
            labels = annotations_data['data_units'][data_unit_key].get('labels', {})
            if labels:
                return labels
        elif 'labels' in annotations_data:
            labels = annotations_data['labels']
            if labels:
                return labels
        elif 'annotations' in annotations_data:
            labels = annotations_data['annotations']
            if labels:
                return labels
        elif 'data_title' in annotations_data:
            # This might be a top-level item with data_units inside
            if 'data_units' in annotations_data:
                data_unit_key = list(annotations_data['data_units'].keys())[0]
                labels = annotations_data['data_units'][data_unit_key].get('labels', {})
                if labels:
                    return labels
    
    # Return empty dictionary if no annotations found
    print(f"No annotations found in {json_path}, using empty annotations")
    return {}

def get_clip_name_from_json(json_path):
    """Extract clip name from JSON file"""
    try:
        with open(json_path, 'r') as f:
            annotations_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        # Fallback: use filename
        filename = os.path.basename(json_path)
        clip_name = os.path.splitext(filename)[0].replace('clip_', '')
        return clip_name
    
    # Check different possible structures
    if isinstance(annotations_data, list) and annotations_data:
        item = annotations_data[0]
        if isinstance(item, dict) and 'data_title' in item:
            video_full_name = item['data_title']
            return video_full_name.split('.')[0]
    
    elif isinstance(annotations_data, dict):
        if 'data_title' in annotations_data:
            video_full_name = annotations_data['data_title']
            return video_full_name.split('.')[0]
    
    # Fallback: use filename
    filename = os.path.basename(json_path)
    clip_name = os.path.splitext(filename)[0].replace('clip_', '')
    return clip_name

def process_clip(clip_name, json_path, frames_dir, output_dir, output_pkl_dir):
    """Process all frames for a single clip. Always processes even with empty annotations."""
    # Load the JSON annotations (returns empty dict if no annotations)
    clip_annotations = load_annotations(json_path)
    
    # Always create output directory for this clip
    clip_output_dir = os.path.join(output_dir, clip_name)
    os.makedirs(clip_output_dir, exist_ok=True)
    
    print(f"Processing {clip_name} (annotations: {'found' if clip_annotations else 'empty'})")
    
    # Process each frame to create masked images
    frames_processed = 0
    for frame_num in range(0, 29):
        frame_filename = f"{frame_num:05d}.jpg"
        frame_path = os.path.join(frames_dir, clip_name, frame_filename)
        
        if os.path.exists(frame_path):
            # Read the frame
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                # Apply masks (will create empty masks if no annotations)
                masked_frame = apply_masks_to_frame(frame, clip_annotations, frame_num)
                
                # Save the masked frame
                output_path = os.path.join(clip_output_dir, frame_filename)
                cv2.imwrite(output_path, masked_frame)
                frames_processed += 1
    
    if frames_processed == 0:
        print(f"Warning: No frames found for {clip_name}")
    
    # Always create PKL file for this clip (even with empty annotations)
    create_pkl_file(clip_name, frames_dir, clip_annotations, output_pkl_dir)

def main():
    # Define paths
    frames_base_dir = '/home/maniden/data/cholec80/trainframes/'
    annotations_dir = "/home/maniden/Desktop/WSL_Tool_classification/testset/trainingTest/"
    output_dir = "/home/maniden/data/cholec80/newtrain/masked_frames"
    output_pkl_dir = "/home/maniden/data/cholec80/newtrain/testcholec_pkl"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_pkl_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    # Debug: Print structure of first JSON file
    if json_files:
        first_json = os.path.join(annotations_dir, json_files[0])
        print(f"Sample JSON structure from {json_files[0]}:")
        try:
            with open(first_json, 'r') as f:
                sample_data = json.load(f)
            print(f"  Type: {type(sample_data)}")
            if isinstance(sample_data, list):
                print(f"  List length: {len(sample_data)}")
                if sample_data:
                    print(f"  First item type: {type(sample_data[0])}")
                    if isinstance(sample_data[0], dict):
                        print(f"  First item keys: {sample_data[0].keys()}")
            elif isinstance(sample_data, dict):
                print(f"  Dictionary keys: {sample_data.keys()}")
        except Exception as e:
            print(f"  Error reading sample JSON: {e}")
        print()
    
    # Process each clip
    for clip_json in tqdm(json_files, desc="Processing clips"):
        json_path = os.path.join(annotations_dir, clip_json)
        
        # Get clip name
        clip_name = get_clip_name_from_json(json_path)
        
        # Check if the corresponding frames directory exists
        clip_frames_dir = os.path.join(frames_base_dir, clip_name)
        if os.path.exists(clip_frames_dir):
            # Always process clip even if no annotations found
            process_clip(clip_name, json_path, frames_base_dir, output_dir, output_pkl_dir)
        else:
            print(f"Warning: Frames directory not found for {clip_name}")
            # Try alternative naming: clip_{name}
            alt_clip_name = f"clip_{clip_name}"
            alt_clip_frames_dir = os.path.join(frames_base_dir, alt_clip_name)
            if os.path.exists(alt_clip_frames_dir):
                print(f"  Found with alternative name: {alt_clip_name}")
                process_clip(alt_clip_name, json_path, frames_base_dir, output_dir, output_pkl_dir)
            else:
                print(f"  Creating empty PKL for {clip_name} (no frames found)")
                # Create empty PKL even if no frames directory exists
                empty_annotations = {}
                create_pkl_file(clip_name, frames_base_dir, empty_annotations, output_pkl_dir)

if __name__ == "__main__":
    main()