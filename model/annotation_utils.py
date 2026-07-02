# annotation_utils.py
import json
import torch
import numpy as np
import cv2
import os
import uuid
import datetime
import time
from typing import List, Dict, Optional

def binary_mask_to_polygons(binary_mask, min_area=10):
    """
    Convert binary mask to polygons in COCO format using similar method as mask_to_polygon
    Returns list of polygons in format [x1,y1,x2,y2,...]
    """
    polygons = []
    
    # Ensure the mask is the right type and properly normalized
    if binary_mask.dtype != np.uint8:
        if np.max(binary_mask) <= 1.0:
            binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask_uint8 = binary_mask.astype(np.uint8)
    else:
        binary_mask_uint8 = binary_mask
    
    # Apply morphological operations to clean up the mask (optional but helpful)
    kernel = np.ones((3, 3), np.uint8)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_CLOSE, kernel)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask_cleaned, cv2.MORPH_OPEN, kernel)
        
    # Use the same approach as mask_to_polygon
    contours, _ = cv2.findContours(
        binary_mask_cleaned, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE  # Use SIMPLE instead of NONE for efficiency
    )
    
    for i, contour in enumerate(contours):
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Filter by minimum area
        if area < min_area:
            continue
        
        # Use the same logic as mask_to_polygon
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            # Convert contour to polygon format [x1,y1,x2,y2,...]
            polygon_points = contour.squeeze(1)  # Remove the extra dimension
            
            # Convert to relative coordinates (0-1 range)
            height, width = binary_mask.shape
            polygon_points = polygon_points.astype(float)
            polygon_points[:, 0] = polygon_points[:, 0] / width   # x coordinates
            polygon_points[:, 1] = polygon_points[:, 1] / height  # y coordinates
            
            # Convert to flat list [x1,y1,x2,y2,...]
            flat_polygon = polygon_points.flatten().tolist()
            polygons.append(flat_polygon)
    
    return polygons
def convert_predictions_to_coco_format(cam3D, model_output, video_name, image_size=(224, 224)):
    """
    Convert model predictions to COCO format frame predictions WITHOUT thresholding
    """
    frame_predictions = []
    
    # Check if cam3D is available
    if cam3D is None:
        print(f"Warning: No CAM maps available for {video_name}")
        return frame_predictions
        
    # Handle different tensor shapes and dimensions
    if len(cam3D.shape) == 5:  # Batch format: (batch_size, num_classes, depth, height, width)
        batch_size, num_classes, depth, height, width = cam3D.shape
        
        for batch_idx in range(batch_size):
            for frame_idx in range(depth):
                frame_pred = {}
                
                # Process each class
                for class_idx, class_name in enumerate(["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]):
                    if class_idx >= num_classes:
                        continue
                        
                    # Get CAM for this class and frame
                    class_cam = cam3D[batch_idx, class_idx, frame_idx]
                    
                    # Handle both tensor and numpy array inputs
                    if hasattr(class_cam, 'cpu'):  # It's a tensor
                        cam_np = class_cam.cpu().numpy()
                    else:  # It's already a numpy array
                        cam_np = class_cam
                                        
                    # NO THRESHOLDING - use the raw CAM values directly
                    # Just normalize to 0-1 range for polygon conversion
                    if np.max(cam_np) > 0:
                        normalized_cam = cam_np / np.max(cam_np)
                    else:
                        normalized_cam = cam_np
                                        
                    # Convert to polygons WITHOUT any pixel count filtering
                    polygons = binary_mask_to_polygons(normalized_cam, min_area=1)
                                     
                    if polygons:
                        # Calculate confidence from model output
                        confidence = 0.5  # default
                        if model_output is not None:
                            if hasattr(model_output, 'cpu'):  # Tensor
                                if len(model_output.shape) == 2:  # [B, C]
                                    confidence = float(torch.sigmoid(model_output[batch_idx, class_idx]).item())
                                else:
                                    confidence = float(torch.sigmoid(model_output[batch_idx, class_idx]).mean().item())
                            else:  # Numpy array
                                if len(model_output.shape) == 2:  # [B, C]
                                    confidence = float(1.0 / (1.0 + np.exp(-model_output[batch_idx, class_idx])))
                                else:
                                    confidence = float(1.0 / (1.0 + np.exp(-model_output[batch_idx, class_idx])).mean())
                        
                        # Don't cap confidence too much
                        confidence = max(0.01, min(0.99, confidence))
                                
                        frame_pred[class_name] = {
                            "polygons": polygons,
                            "confidence": confidence,
                            "pixel_count": int(np.sum(normalized_cam > 0))  # Count non-zero pixels
                        }
                        # print(f"DEBUG: Added {class_name} with {len(polygons)} polygons, confidence: {confidence:.3f}")
                    # else:
                        # print(f"DEBUG: {class_name} - no polygons generated from CAM")
                
                # Add frame even if it has no tools (empty frame_pred)
                frame_predictions.append(frame_pred)
                # print(f"DEBUG: Frame {frame_idx} has {len(frame_pred)} tools")
    
    print(f"Converted {len(frame_predictions)} frames to COCO format for {video_name}")
    total_annotations = sum(len(frame) for frame in frame_predictions)
    print(f"Total annotations across all frames: {total_annotations}")
    
    return frame_predictions

def create_coco_annotation_json(video_name, frame_predictions, output_dir, image_size=(256, 256)):
    """
    Create COCO format JSON for Encord import with proper polygon handling
    """
    frame_width, frame_height = image_size
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Grasper", "supercategory": "tool"},
            {"id": 2, "name": "Bipolar", "supercategory": "tool"},
            {"id": 3, "name": "Hook", "supercategory": "tool"},
            {"id": 4, "name": "Scissors", "supercategory": "tool"},
            {"id": 5, "name": "Clipper", "supercategory": "tool"},
            {"id": 6, "name": "Irrigator", "supercategory": "tool"},
            {"id": 7, "name": "SpecimenBag", "supercategory": "tool"}
        ],
        "info": {
            "description": f"Surgical Tool Predictions - {video_name}",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "Active Learning System",
            "date_created": datetime.datetime.now().isoformat()
        }
    }
    
    annotation_id = 1
    total_polygons = 0
    
    for frame_idx, frame_pred in enumerate(frame_predictions):
        # Create image entry for each frame
        image_id = len(coco_data["images"]) + 1
        coco_data["images"].append({
            "id": image_id,
            "width": frame_width,
            "height": frame_height,
            "file_name": f"{video_name}_frame{frame_idx:06d}.jpg",
            "frame_index": frame_idx,  # Important for Encord mapping
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.datetime.now().isoformat()
        })
        
        # Process each instrument type
        for instrument_type, instrument_data in frame_pred.items():
            # Map instrument name to category ID
            category_id_map = {
                "Grasper": 1, "Bipolar": 2, "Hook": 3, "Scissors": 4,
                "Clipper": 5, "Irrigator": 6, "SpecimenBag": 7
            }
            category_id = category_id_map.get(instrument_type, 1)
            
            # Create annotation for each polygon
            for polygon in instrument_data.get("polygons", []):
                if len(polygon) >= 6:  # Need at least 3 points (x,y,x,y,x,y)
                    # Convert relative coordinates back to absolute for COCO
                    absolute_polygon = []
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            x = polygon[i] * frame_width
                            y = polygon[i + 1] * frame_height
                            absolute_polygon.extend([x, y])
                    
                    if len(absolute_polygon) >= 6:  # Still need at least 3 points after conversion
                        # Calculate bounding box from absolute coordinates
                        x_coords = absolute_polygon[::2]
                        y_coords = absolute_polygon[1::2]
                        bbox = [
                            float(min(x_coords)),  # x
                            float(min(y_coords)),  # y
                            float(max(x_coords) - min(x_coords)),  # width
                            float(max(y_coords) - min(y_coords))   # height
                        ]
                        
                        area = bbox[2] * bbox[3]  # Approximate area
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [absolute_polygon],  # Absolute coordinates
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "score": float(instrument_data.get("confidence", 0.5))
                        })
                        annotation_id += 1
                        total_polygons += 1
    
    print(f"Created COCO data with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    print(f"Total polygons created: {total_polygons}")
    
    # Save COCO format JSON
    output_file = os.path.join(output_dir, f"{video_name}_coco_annotations.json")
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"COCO format annotations saved: {output_file}")
    
    # Print summary for debugging
    print("COCO Export Summary:")
    print(f"  - Images: {len(coco_data['images'])}")
    print(f"  - Annotations: {len(coco_data['annotations'])}")
    print(f"  - Categories: {len(coco_data['categories'])}")
    if coco_data['annotations']:
        print(f"  - First annotation: {coco_data['annotations'][0]}")
    
    return output_file

def create_encord_annotation_json(video_name, frame_predictions, output_dir, image_size=(256, 256)):
    """
    Create Encord-compatible JSON from model predictions
    This is an alternative to COCO format if needed
    """
    frame_width, frame_height = image_size
    
    # Create the main annotation structure
    annotation_data = {
        "label_hash": str(uuid.uuid4()),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_edited_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_title": f"{video_name}.mp4",
        "data_hash": str(uuid.uuid4()),
        "data_type": "video",
        "dataset_hash": str(uuid.uuid4()),
        "dataset_title": f"{video_name}_dataset",
        "annotation_task_status": "IN_PROGRESS",
        "labels": {
            "objects": {},
            "classifications": {},
            "data_units": {}
        },
        "object_answers": {},
        "classification_answers": {},
        "object_actions": {},
        "label_status": "LABELLED",
        "is_valid": True
    }
    
    # Process each frame
    for frame_idx, frame_pred in enumerate(frame_predictions):
        frame_objects = []
        
        for instrument_type, instrument_data in frame_pred.items():
            for polygon in instrument_data.get("polygons", []):
                # Create unique object hash
                object_hash = str(uuid.uuid4())[:8]
                
                # Create polygon object
                polygon_obj = {
                    "name": instrument_type,
                    "color": _get_color_for_category(instrument_type),
                    "value": instrument_type.lower(),
                    "createdAt": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "confidence": float(instrument_data.get("confidence", 1.0)),
                    "objectHash": object_hash,
                    "shape": "polygon",
                    "manualAnnotation": False
                }
                
                # Add polygon coordinates in Encord format
                polygon_points = {}
                for point_idx in range(0, len(polygon), 2):
                    if point_idx + 1 < len(polygon):
                        point_key = str(point_idx // 2)
                        polygon_points[point_key] = {
                            "x": float(polygon[point_idx]),
                            "y": float(polygon[point_idx + 1])
                        }
                
                polygon_obj["polygon"] = polygon_points
                
                frame_objects.append(polygon_obj)
                annotation_data["labels"]["objects"][object_hash] = polygon_obj
        
        # Create data unit for this frame
        if frame_objects:
            frame_key = str(frame_idx)
            annotation_data["labels"]["data_units"][frame_key] = {
                "data_sequence": frame_key,
                "labels": {
                    "objects": frame_objects,
                    "classifications": []
                },
                "width": frame_width,
                "height": frame_height
            }
    
    # Create output filename
    output_filename = f"{video_name}_encord_annotations.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save JSON file
    with open(output_path, 'w') as f:
        json.dump([annotation_data], f, indent=2)
    
    print(f"Encord annotations saved to: {output_path}")
    return output_path

def _get_color_for_category(category_name: str) -> str:
    """Get a consistent color for each category"""
    color_map = {
        'Grasper': '#FF0000',
        'Bipolar': '#00FF00', 
        'Hook': '#0000FF',
        'Scissors': '#FFFF00',
        'Clipper': '#FF00FF',
        'Irrigator': '#00FFFF',
        'SpecimenBag': '#FFA500'
    }
    return color_map.get(category_name, '#808080')