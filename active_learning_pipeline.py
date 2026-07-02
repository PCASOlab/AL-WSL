import os
import json
import uuid
import time
from dataset import io
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import requests
from pathlib import Path
from encord import EncordUserClient, Dataset
from encord.objects import LabelRowV2
import pickle
import random
from sklearn.cluster import KMeans
from scipy import stats
from encord.utilities.coco.datastructure import FrameIndex

# Import everything from annotation_utils
from model.annotation_utils import (
    binary_mask_to_polygons,
    convert_predictions_to_coco_format,
    create_coco_annotation_json,
    create_encord_annotation_json,
    _get_color_for_category
)

class ActiveLearningPipeline:
    def __init__(self, ssh_private_key_path: str, project_title: str, output_root: str, project_hash: str = None, al_dir: str=None):
        self.ssh_private_key_path = ssh_private_key_path
        self.user_client = self._setup_encord_with_ssh(ssh_private_key_path)
        self.project_title = project_title
        self.project = None
        self.project_hash = project_hash
        self.output_root = output_root
        self.sent_videos = set()
        
        # Active learning state
        self.current_round = 0
        self.best_loss = float('inf')
        self.loss_history = []
        self._ontology = None
        self._ontology_features = None
        self._data_hash_lookup = {}  # Add data hash lookup cache
        
        # Create active learning directory
        self.al_dir = os.path.join(output_root, al_dir)
        os.makedirs(self.al_dir, exist_ok=True)
        
        # Initialize project connection
        self._initialize_project()
        # Build data hash lookup immediately
        self._build_data_hash_lookup()
        
    def _setup_encord_with_ssh(self, ssh_private_key_path: str):
        """Initialize Encord client with SSH private key"""
        try:
            client = EncordUserClient.create_with_ssh_private_key(
                ssh_private_key_path=ssh_private_key_path
            )
            print("Successfully authenticated with Encord using SSH key")
            return client
        except Exception as e:
            print(f"Encord SSH authentication failed: {e}")
            raise
    
    def _initialize_project(self):
        """Find and initialize the project by title or hash"""
        try:
            if self.project_hash:
                self.project = self.user_client.get_project(self.project_hash)
            else:
                projs = self.user_client.get_projects()
                for p in self.user_client.get_projects():
                    if p["project"].title == self.project_title:
                        self.project = p["project"]
                        self.project_hash = p["project"].project_hash
                        self.project = self.user_client.get_project(self.project_hash)
                        break
                
                if self.project is None:
                    raise ValueError(f"Project with title '{self.project_title}' not found")
                    
            # Load ontology
            self._ontology = self.get_ontology()
                    
        except Exception as e:
            print(f"Error initializing project: {e}")
            raise

    def _build_data_hash_lookup(self):
        """
        Build a mapping from video/file base-name -> data_row_hash (supports multiple SDK versions).
        This is used by get_data_hash_for_video.
        """
        self._data_hash_lookup = {}

        try:
            # Iterate datasets in the project
            for pd in self.project.list_datasets():
                dataset = self.user_client.get_dataset(pd.dataset_hash)
                for dr in dataset.list_data_rows():
                    # Try modern attribute names first
                    data_hash = None
                    data_title = None
                    
                    # Modern Encord: data_row_hash
                    if hasattr(dr, "data_row_hash"):
                        data_hash = str(dr.data_row_hash)
                    # Older/alternate names
                    elif hasattr(dr, "data_hash"):
                        data_hash = str(dr.data_hash)
                    elif hasattr(dr, "uid"):
                        data_hash = str(dr.uid)
                    elif hasattr(dr, "hash"):
                        data_hash = str(dr.hash)
                    elif hasattr(dr, "id"):
                        data_hash = str(dr.id)

                    # Try several possible title/name attributes
                    data_title = getattr(dr, "data_title", None) or getattr(dr, "title", None) or getattr(dr, "name", None)
                    # Also try file_link or url to extract a filename if title missing
                    if not data_title:
                        file_link = getattr(dr, "file_link", None) or getattr(dr, "url", None)
                        if file_link:
                            data_title = os.path.basename(file_link)

                    if data_hash and data_title:
                        base_name = os.path.splitext(data_title)[0]
                        self._data_hash_lookup[base_name] = data_hash
                        self._data_hash_lookup[data_title] = data_hash
                        # also put the full hash mapping to itself for quicker lookups
                        self._data_hash_lookup[data_hash] = data_hash

                    
            # Method 2: use label rows as an additional source (if available)
            try:
                label_rows = self.project.list_label_rows_v2()
            except Exception:
                label_rows = []

            for lr in label_rows:
                # LabelRowV2 often exposes data_row_hash or data_hash
                data_hash = None
                data_title = None
                
                if hasattr(lr, "data_row_hash"):
                    data_hash = str(lr.data_row_hash)
                elif hasattr(lr, "data_hash"):
                    data_hash = str(lr.data_hash)
                elif hasattr(lr, "data_uid"):
                    data_hash = str(lr.data_uid)
                elif hasattr(lr, "uid"):
                    data_hash = str(lr.uid)
                elif hasattr(lr, "hash"):
                    data_hash = str(lr.hash)

                data_title = getattr(lr, "data_title", None) or getattr(lr, "title", None) or getattr(lr, "name", None)
                if data_hash and data_title:
                    base_name = os.path.splitext(data_title)[0]
                    self._data_hash_lookup[base_name] = data_hash
                    self._data_hash_lookup[data_title] = data_hash
                    self._data_hash_lookup[data_hash] = data_hash

            if not self._data_hash_lookup:
                for pd in self.project.list_datasets():
                    dataset = self.user_client.get_dataset(pd.dataset_hash)
                    first_dr = next(iter(dataset.list_data_rows()), None)
                    if first_dr:
                        break

        except Exception as e:
            import traceback
            traceback.print_exc()

    def build_data_row_lookup(self) -> Dict:
        """
        Build data_row_hash -> file_link map (SDK-compatible).
        Returns a dict where keys are data_row_hash strings (and some alt keys) and values are file links.
        """
        data_row_lookup = {}
        try:
            for pd in self.project.list_datasets():
                dataset = self.user_client.get_dataset(pd.dataset_hash)
                for dr in dataset.list_data_rows():
                    # Prefer data_row_hash (modern)
                    key = None
                    if hasattr(dr, "data_row_hash"):
                        key = str(dr.data_row_hash)
                    elif hasattr(dr, "data_hash"):
                        key = str(dr.data_hash)
                    elif hasattr(dr, "uid"):
                        key = str(dr.uid)
                    elif hasattr(dr, "hash"):
                        key = str(dr.hash)
                    elif hasattr(dr, "id"):
                        key = str(dr.id)

                    # Get a file link / url attribute
                    file_link = getattr(dr, "file_link", None) or getattr(dr, "url", None)
                    if key and file_link:
                        data_row_lookup[key] = file_link

            return data_row_lookup

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {}

    def get_data_hash_for_video(self, video_name: str) -> Optional[str]:
        """Get Encord data hash for a video filename"""
        # Remove extension and any path components
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        
        # Try exact match first
        if base_name in self._data_hash_lookup:
            data_hash = self._data_hash_lookup[base_name]
            return data_hash
        
        # Try partial matches
        for stored_name, data_hash in self._data_hash_lookup.items():
            if base_name in stored_name or stored_name in base_name:
                return data_hash
        
        return None

    # def predict_uncertainty(self, input_video,model, dataloader, num_samples: int = 50) -> List[Dict]:
    #     """Predict uncertainty for samples using the model"""
        
    #     uncertain_samples = []
    #     model.eval()
        
    #     with torch.no_grad():
    #         # Sample a subset of the dataloader
    #         sampled_indices = random.sample(range(len(dataloader.all_video_dir_list)), 
    #                                     min(num_samples, len(dataloader.all_video_dir_list)))
            
    #         for idx in sampled_indices:
    #             try:
    #                 # Get the video info
    #                 filename, dataset_source = dataloader.all_video_dir_list[idx]
    #                 clip_name = os.path.splitext(filename)[0]
                    
    #                 # Load the video data
    #                 folder_path = dataloader._get_folder_path(dataset_source)
    #                 data_dict = io.read_a_pkl(folder_path, clip_name)
                    
    #                 if data_dict is None:
    #                     continue
                        
    #                 video_data = data_dict.get('frames')
    #                 if video_data is None:
    #                     continue
                    
    #                 # Get data hash for this video
    #                 data_hash = self.get_data_hash_for_video(filename)
    #                 if not data_hash:
    #                     continue
                    
    #                 # Prepare input
                    
    #                 # Get model predictions
    #                 output = model.forward(input_video, None, None, Enable_student=False)
    #                 cam_maps = model.cam3D if hasattr(model, 'cam3D') else None                    
                    
    #                 # Calculate uncertainty
    #                 uncertainty = self.calculate_confidence(cam_maps, output)
                    
    #                 uncertain_samples.append({
    #                     'file_name': filename,
    #                     'clip_name': clip_name,
    #                     'dataset_source': dataset_source,
    #                     'uncertainty': uncertainty,
    #                     'cam_maps': cam_maps.cpu().numpy() if cam_maps is not None else None,
    #                     'prediction': output.cpu().numpy() if hasattr(output, 'cpu') else output,
    #                     'data_hash': data_hash
    #                 })
                    
    #             except Exception as e:
    #                 continue
        
    #     # Sort by uncertainty (highest first)
    #     uncertain_samples.sort(key=lambda x: x['uncertainty'], reverse=True)
        
    #     return uncertain_samples
    def predict_uncertainty(self, input_video, model, dataloader, num_samples: int = 50) -> List[Dict]:
        """Predict uncertainty for samples using the model"""
        
        uncertain_samples = []
        model.eval()
        
        # Define transforms (Standard ImageNet stats for 0-255 inputs)
        # Verify these match your specific training setup if possible
        import torch.nn.functional as F
        mean = torch.tensor([124.0, 124.0, 124.0]).view(1, 3, 1, 1, 1).to(model.device)
        std = torch.tensor([60.0, 60.0, 60.0]).view(1, 3, 1, 1, 1).to(model.device)
        target_size = 224 # Ensure this matches your model's expected input size
        
        with torch.no_grad():
            # Sample a subset of the dataloader
            sampled_indices = random.sample(range(len(dataloader.all_video_dir_list)), 
                                        min(num_samples, len(dataloader.all_video_dir_list)))
            
            for idx in sampled_indices:
                try:
                    # Get the video info
                    filename, dataset_source = dataloader.all_video_dir_list[idx]
                    clip_name = os.path.splitext(filename)[0]
                    
                    # Load the video data
                    folder_path = dataloader._get_folder_path(dataset_source)
                    data_dict = io.read_a_pkl(folder_path, clip_name)
                    
                    if data_dict is None:
                        continue
                        
                    video_data = data_dict.get('frames') # Expecting [Frames, H, W, C]
                    if video_data is None:
                        continue
                    
                    # Get data hash for this video
                    data_hash = self.get_data_hash_for_video(filename)
                    if not data_hash:
                        continue
                    
                    # --- FIX 1: Process the Loaded Video Data ---
                    # Convert numpy to tensor
                    curr_video = torch.from_numpy(video_data).float().to(model.device)
                    
                    # Store original dimensions for later
                    orig_frames = curr_video.shape[0]
                    orig_h = curr_video.shape[1]
                    orig_w = curr_video.shape[2]
                    
                    # Rearrange from [D, H, W, C] to [B, C, D, H, W]
                    # Assuming video_data is [Frames, Height, Width, Channels]
                    if curr_video.shape[-1] == 3:
                        curr_video = curr_video.permute(3, 0, 1, 2) # -> [C, D, H, W]
                    
                    # Add batch dimension
                    curr_video = curr_video.unsqueeze(0) # -> [1, C, D, H, W]
                    
                    # Preprocess: Interpolate to Model Input Size (e.g. 224x224)
                    # Note: We keep Depth (Time) as is or resize if model requires fixed depth.
                    # Your model seems to handle variable depth or interpolate internally.
                    # We strictly resize spatial dims here.
                    curr_input = F.interpolate(curr_video, size=(curr_video.shape[2], target_size, target_size), 
                                             mode='trilinear', align_corners=False)
                    
                    # Normalize
                    curr_input = (curr_input - mean) / std

                    # --- Run Inference on the NEW Input ---
                    # Using curr_input instead of static input_video
                    output = model.forward(curr_input, None, None, Enable_student=False)
                    cam_maps = model.cam3D if hasattr(model, 'cam3D') else None                    
                    
                    # --- FIX 2: Interpolate CAMs back to Original 29 Frames ---
                    if cam_maps is not None:
                        # cam_maps shape: [Batch, Classes, Model_Frames, Model_H, Model_W]
                        if isinstance(cam_maps, np.ndarray):
                            cam_maps = torch.from_numpy(cam_maps).to(model.device)
                        
                        # Interpolate temporal dimension back to orig_frames (29)
                        # Interpolate spatial dimension back to orig_h, orig_w (Full HD)
                        cam_maps = F.interpolate(
                            cam_maps, 
                            size=(orig_frames, orig_h, orig_w), 
                            mode='trilinear', 
                            align_corners=False
                        )
                    
                    # Calculate uncertainty
                    uncertainty = self.calculate_confidence(cam_maps, output)
                    
                    uncertain_samples.append({
                        'file_name': filename,
                        'clip_name': clip_name,
                        'dataset_source': dataset_source,
                        'uncertainty': uncertainty,
                        'cam_maps': cam_maps.cpu().numpy() if cam_maps is not None else None,
                        'prediction': output.cpu().numpy() if hasattr(output, 'cpu') else output,
                        'data_hash': data_hash
                    })
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        # Sort by uncertainty (highest first)
        uncertain_samples.sort(key=lambda x: x['uncertainty'], reverse=True)
        
        return uncertain_samples

    def calculate_confidence(self, cam3D, model_output):
        confidence_scores = {}
    
        # Method 1: CAM activation strength and consistency
        if cam3D is not None:
            # Normalize CAM to 0-1 range
            cam_normalized = (cam3D - cam3D.min()) / (cam3D.max() - cam3D.min() + 1e-8)
            
            # Confidence = how "clear" the activation is
            # Low confidence: weak or scattered activations
            # High confidence: strong, focused activations
            activation_strength = torch.mean(cam_normalized).item()
            activation_consistency = 1.0 - torch.std(cam_normalized).item()  # Lower std = more consistent
            
            confidence_scores['cam_activation'] = activation_strength * activation_consistency
        
        # Method 2: Classification-CAM alignment
        if model_output is not None:
            classification_probs = torch.sigmoid(model_output)
            max_class_prob = torch.max(classification_probs).item()
            confidence_scores['classification_alignment'] = max_class_prob
        
        # Method 3: Spatial-temporal consistency
        if cam3D is not None and len(cam3D.shape) == 5:  # [B, C, D, H, W]
            temporal_consistency = self._calculate_temporal_consistency(cam3D)
            confidence_scores['temporal_consistency'] = temporal_consistency
        
        # Overall uncertainty (lower confidence = higher uncertainty for selection)
        if confidence_scores:
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            return 1.0 - overall_confidence  # Convert to uncertainty score
        else:
            return 0.5  # Default uncertainty

    def _calculate_temporal_consistency(self, cam3D):
        """Calculate how consistent CAM activations are across frames"""
        if len(cam3D.shape) != 5:
            return 0.0
        
        # cam3D shape: [B, C, D, H, W]
        batch_size, num_classes, num_frames, height, width = cam3D.shape
        
        consistency_scores = []
        for b in range(batch_size):
            for c in range(num_classes):
                # Get CAM for this class across all frames
                class_cam = cam3D[b, c]  # [D, H, W]
                
                # Calculate frame-to-frame correlation
                if num_frames > 1:
                    frame_correlations = []
                    for i in range(num_frames - 1):
                        # Flatten and compute correlation
                        frame1 = class_cam[i].flatten()
                        frame2 = class_cam[i + 1].flatten()
                        
                        # Use numpy for correlation calculation
                        corr = np.corrcoef(frame1.cpu().numpy(), frame2.cpu().numpy())[0, 1]
                        frame_correlations.append(corr if not np.isnan(corr) else 0.0)
                    
                    avg_correlation = np.mean(frame_correlations) if frame_correlations else 0.0
                    consistency_scores.append(avg_correlation)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0

    def upload_coco_annotations(self, coco_file_path: str, round_number: int):
        """
        Upload COCO format annotations to Encord project using the official SDK
        """
        try:
            print(f"Uploading COCO annotations from: {coco_file_path}")
            
            # Validate file exists
            if not os.path.exists(coco_file_path):
                return False
            
            # Load the COCO data
            with open(coco_file_path, 'r') as f:
                labels_dict = json.load(f)
            
            print(f"COCO data: {len(labels_dict.get('images', []))} images, "
                f"{len(labels_dict.get('annotations', []))} annotations, "
                f"{len(labels_dict.get('categories', []))} categories")
            
            # Check if we have annotations to upload
            if len(labels_dict.get('annotations', [])) == 0:
                return False
            
            # Extract video name from COCO filename
            video_name = os.path.basename(coco_file_path).replace('_coco_annotations.json', '')
            
            # Get data hash for this video using our lookup
            data_hash = self.get_data_hash_for_video(video_name)
            if not data_hash:
                return False
            
            
            # Build a mapping from COCO category IDs to the feature hashes in your Encord Ontology
            category_id_to_feature_hash = {}
            
            for coco_category in labels_dict.get('categories', []):
                category_id = coco_category['id']
                category_name = coco_category['name']
                
                try:
                    # Use ontology structure to find by title (like the reference code)
                    ont_struct = self.project.ontology_structure
                    ont_obj = ont_struct.get_child_by_title(category_name)
                    feature_hash = ont_obj.feature_node_hash
                    category_id_to_feature_hash[category_id] = feature_hash
                except Exception as e:
                    # Create fallback feature hash
                    feature_hash = str(uuid.uuid4())[:8]
                    category_id_to_feature_hash[category_id] = feature_hash
            
            # Build a mapping from COCO image IDs to Encord frame indices
            image_id_to_frame_index = {}
                        
            # Find existing label row or check if we can create one
            label_hash = None
            label_row_exists = False
            
            # Check if label row already exists
            label_rows = self.project.list_label_rows_v2()
            for lr in label_rows:
                lr_data_hash = getattr(lr, "data_row_hash", getattr(lr, "data_hash", None))
                if lr_data_hash and str(lr_data_hash) == data_hash:
                    label_hash = lr.label_hash
                    label_row_exists = True
                    break
            
          
            # Process each image in the COCO data
            for img in labels_dict.get('images', []):
                image_id = img['id']
                
                # Determine frame number
                frame_num = img.get('frame_index', image_id - 1)  # Default to 0-based frame index
                
                # Create FrameIndex object using the data hash and frame number
                image_id_to_frame_index[image_id] = FrameIndex(data_hash, frame=frame_num)
                        
            # Import COCO labels using Encord SDK
            print(f"\nStarting COCO import...")
            print(f"  - Categories: {len(category_id_to_feature_hash)}")
            print(f"  - Frame indices: {len(image_id_to_frame_index)}")
            print(f"  - Data hash: {data_hash}")
            print(f"  - Label row exists: {label_row_exists}")
            
            import_result = self.project.import_coco_labels(
                labels_dict,
                category_id_to_feature_hash,
                image_id_to_frame_index
            )
            
            print(f" Successfully imported COCO annotations to Encord project")
            print(f"   - Project: {self.project_title}")
            print(f"   - Round: {round_number}")
            print(f"   - Video: {video_name}")
            print(f"   - Images processed: {len(labels_dict.get('images', []))}")
            print(f"   - Annotations imported: {len(labels_dict.get('annotations', []))}")
            
            # Save upload report
            upload_report = {
                'round_number': round_number,
                'coco_file': coco_file_path,
                'upload_time': time.time(),
                'upload_status': 'success',
                'video_name': video_name,
                'data_hash': data_hash,
                'label_hash': label_hash,
                'label_row_exists': label_row_exists,
                'images_count': len(labels_dict.get('images', [])),
                'annotations_count': len(labels_dict.get('annotations', [])),
                'categories_count': len(labels_dict.get('categories', [])),
                'category_mappings': category_id_to_feature_hash,
                'project_title': self.project_title,
                'project_hash': self.project_hash,
            }
            
            report_file = os.path.join(self.al_dir, f"round_{round_number}_coco_upload_report.json")
            with open(report_file, 'w') as f:
                json.dump(upload_report, f, indent=2)
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Save error report
            error_report = {
                'round_number': round_number,
                'coco_file': coco_file_path,
                'upload_time': time.time(),
                'upload_status': 'failed',
                'error_message': str(e),
                'project_title': self.project_title,
                'project_hash': self.project_hash
            }
            
            error_file = os.path.join(self.al_dir, f"round_{round_number}_coco_upload_error.json")
            with open(error_file, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            return False
    def get_ontology(self):
        try:
            if self._ontology is not None:
                return self._ontology
            
            
            # Method 1: Try get_ontology() method (newer SDK versions)
            if hasattr(self.project, 'get_ontology'):
                self._ontology = self.project.get_ontology()
                return self._ontology
            
            return None
            
        except Exception as e:
            print(f"Error loading ontology: {e}")
            return None

    def get_ontology_features(self):
        """Get all features from the project ontology with their hashes"""
        if self._ontology_features is not None:
            return self._ontology_features
            
        self._ontology_features = {}
        
        def extract_features(node, path=""):
            # Check for objects (tool features)
            if hasattr(node, 'objects'):
                for obj in node.objects:
                    feature_name = getattr(obj, 'name', 'unnamed')
                    feature_hash = getattr(obj, 'feature_node_hash', None) or getattr(obj, 'hash', None)
                    
                    if feature_hash:
                        full_path = f"{path}/{feature_name}" if path else feature_name
                        self._ontology_features[feature_name.lower()] = {
                            'feature_hash': str(feature_hash),  # Ensure it's string
                            'full_path': full_path,
                            'feature_obj': obj
                        }
            
            # Check for classifications (if any)
            if hasattr(node, 'classifications'):
                for classification in node.classifications:
                    feature_name = getattr(classification, 'name', 'unnamed')
                    feature_hash = getattr(classification, 'feature_node_hash', None) or getattr(classification, 'hash', None)
                    
                    if feature_hash:
                        full_path = f"{path}/{feature_name}" if path else feature_name
                        self._ontology_features[feature_name.lower()] = {
                            'feature_hash': str(feature_hash),  # Ensure it's string
                            'full_path': full_path,
                            'feature_obj': classification
                        }
        
        if self._ontology:
            extract_features(self._ontology)
            
            # Debug: Print available features
            if self._ontology_features:
                for feature_name, feature_data in self._ontology_features.items():
                    print(f"  - {feature_name}: {feature_data['feature_hash']}")
            else:
              
                # Try to access common ontology structures
                for attr in ['objects', 'classifications', 'features', 'ontology']:
                    if hasattr(self._ontology, attr):
                        value = getattr(self._ontology, attr)
                        print(f"  {attr}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
        
        return self._ontology_features
    
    def download_annotations(self, round_number: int, selected_filenames: set, download_dir: str = None):
        """
        Download annotations for selected videos from Encord project.
        
        Args:
            round_number: The round number for logging
            selected_filenames: Set of filenames (without .mp4 extension) to download
            download_dir: Directory to save annotation JSON files
        
        Returns:
            Path to download directory
        """
        import json
        from pathlib import Path
        
        # Create download directory
        download_path = Path(download_dir)
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Get all label rows
        label_rows = self.project.list_label_rows_v2()
        print(f"Found {len(label_rows)} label rows in project.")
        
        # Initialize ALL label rows first (as shown in Encord example)
        print("Initializing label rows...")
        with self.project.create_bundle() as bundle:
            for label_row in label_rows:
                label_row.initialise_labels(bundle=bundle)
        
        # Now filter and download
        successful_downloads = 0
        
        for lr in label_rows:
            # Get base filename
            video_filename = lr.data_title
            base_filename = video_filename
            if video_filename.lower().endswith(".mp4"):
                base_filename = video_filename[:-4]
            
            # Check if this video is in our selected list
            if base_filename not in selected_filenames:
                continue
            
            try:
                # Convert to dictionary - should work now that labels are initialized
                lr_dict = lr.to_encord_dict()
                
                # Create clean filename
                clean_filename = "".join(c for c in base_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
                
                # Save to JSON file
                json_path = download_path / f"{clean_filename}.json"
                with open(json_path, "w") as f:
                    json.dump(lr_dict, f, indent=4)
                
                successful_downloads += 1
                print(f" Saved: {base_filename}")
                
            except Exception as e:
                print(f" Failed for {base_filename}: {str(e)}")
        
        # Summary
        print(f"\nRound {round_number} complete: {successful_downloads} files downloaded")
        print(f"Saved to: {download_path.absolute()}")
        
        return str(download_path.absolute())
    def get_label_rows(self) -> List[LabelRowV2]:
        """Get all label rows from the project"""
        label_rows = self.project.list_label_rows_v2()
        print(f"Found {len(label_rows)} label rows in project")
        return label_rows
        

    def wait_for_human_correction(self, round_number: int):
        """Wait for human annotations in Encord"""
        print(f"\n=== Round {round_number}: Waiting for human annotations ===")
        print("Please annotate the uploaded videos in Encord.")
        print("After completing annotations, press 'c' to continue or 'q' to quit.")
        
        while True:
            user_input = input().strip().lower()
            if user_input == 'c':
                print("Annotations confirmed. Continuing to next round...")
                return True
            elif user_input == 'q':
                print("Quitting active learning pipeline.")
                return False
            else:
                print("Please press 'c' to continue or 'q' to quit.")
            

    def save_round_checkpoint(self, Model_infer, round_number: int):
        """Save model checkpoint for this round"""
        checkpoint_path = os.path.join(self.al_dir, f"round_{round_number}_checkpoint")
        
        torch.save(Model_infer.state_dict(), checkpoint_path + "_full.pth")
        torch.save(Model_infer.VideoNets.state_dict(), checkpoint_path + "_tc.pth")
        torch.save(Model_infer.VideoNets_S.state_dict(), checkpoint_path + "_st.pth")
        torch.save(Model_infer.backbone.state_dict(), checkpoint_path + "_vit.pth")
        
        print(f"Round {round_number} checkpoint saved: {checkpoint_path}")

    def should_continue_training(self, current_loss: float, round_number: int, 
                               max_rounds: int = 10, loss_threshold: float = 0.01) -> bool:
        """Determine if we should continue active learning"""
        if round_number >= max_rounds:
            print(f"Reached maximum rounds ({max_rounds})")
            return False
        
        if current_loss <= loss_threshold:
            print(f"Loss threshold reached ({current_loss:.4f} <= {loss_threshold})")
            return False
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        
        return True

    def get_active_learning_status(self):
        """Get current status of active learning pipeline"""
        return {
            'current_round': self.current_round,
            'best_loss': self.best_loss,
            'total_rounds_completed': len(self.loss_history),
            'project_title': self.project_title,
            'is_complete': self.best_loss <= 0.01 or self.current_round >= 16
        }

    def select_samples_for_annotation(self, uncertain_samples: List[Dict], 
                                    selection_strategy: str = "highest_uncertainty",
                                    num_samples: int = 10) -> List[Dict]:
        """Select samples for annotation based on various strategies"""
        if selection_strategy == "highest_uncertainty":
            # Select samples with highest uncertainty
            selected = sorted(uncertain_samples, key=lambda x: x['uncertainty'], reverse=True)[:num_samples]
        elif selection_strategy == "diversity":
            # Select diverse samples using clustering
            selected = self._select_diverse_samples(uncertain_samples, num_samples)
        elif selection_strategy == "mixed":
            # Mix of highest uncertainty and diversity
            high_uncertainty = sorted(uncertain_samples, key=lambda x: x['uncertainty'], reverse=True)[:num_samples//2]
            diverse = self._select_diverse_samples(uncertain_samples, num_samples//2)
            selected = high_uncertainty + diverse
        else:
            selected = uncertain_samples[:num_samples]
        
        return selected

    def _select_diverse_samples(self, samples: List[Dict], num_samples: int) -> List[Dict]:
        """Select diverse samples using clustering"""
        if len(samples) <= num_samples:
            return samples
        
        # Extract features for clustering (using predictions as features)
        features = []
        for sample in samples:
            if 'prediction' in sample:
                pred = sample['prediction']
                if hasattr(pred, 'flatten'):
                    features.append(pred.flatten())
                else:
                    features.append(np.array(pred).flatten())
            else:
                features.append(np.zeros(10))  # Fallback
        
        features = np.array(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_samples, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select one sample from each cluster
        selected = []
        for cluster_id in range(num_samples):
            cluster_samples = [samples[i] for i in range(len(samples)) if cluster_labels[i] == cluster_id]
            if cluster_samples:
                # Select the most uncertain sample from each cluster
                most_uncertain = max(cluster_samples, key=lambda x: x['uncertainty'])
                selected.append(most_uncertain)
        
        return selected
    def select_random_videos(self, dataloader, num_videos: int = 10) -> List[Dict]:
        """Select random videos that haven't been sent for annotation yet"""
        print(f"Selecting {num_videos} random videos for annotation...")
        
        available_videos = []
        
        # Get all available videos from dataloader
        for idx, (filename, dataset_source) in enumerate(dataloader.all_video_dir_list):
            clip_name = os.path.splitext(filename)[0]
            
            # Skip if already sent
            if clip_name in self.sent_videos:
                continue
                
            # Get data hash for this video
            data_hash = self.get_data_hash_for_video(filename)
            if not data_hash:
                print(f"Could not find data hash for {filename}, skipping...")
                continue
            
            available_videos.append({
                'file_name': filename,
                'clip_name': clip_name,
                'dataset_source': dataset_source,
                'data_hash': data_hash
            })
        
        # If we've sent all videos, reset (or handle as needed)
        if not available_videos:
            print("All available videos have been sent for annotation. Resetting sent_videos set.")
            self.sent_videos = set()
            # Try again with reset set
            return self.select_random_videos(dataloader, num_videos)
        
        # Select random videos
        selected_videos = random.sample(available_videos, min(num_videos, len(available_videos)))
        
        # Mark them as sent
        for video in selected_videos:
            self.sent_videos.add(video['clip_name'])
            print(f" Selected video for annotation: {video['clip_name']}")
        
        print(f"Selected {len(selected_videos)} videos for annotation")
        return selected_videos

    def process_videos_for_annotation(self, dataloader, model, num_videos: int = 10) -> List[Dict]:
        """Process random videos and generate COCO annotations"""
        print(f"Processing {num_videos} random videos for annotation...")
        
        # Select random videos
        selected_videos = self.select_random_videos(dataloader, num_videos)
        
        processed_videos = []
        
        for video_info in selected_videos:
            try:
                filename = video_info['file_name']
                clip_name = video_info['clip_name']
                dataset_source = video_info['dataset_source']
                
                print(f"Processing video: {filename}")
                
                # Load the video data
                folder_path = dataloader._get_folder_path(dataset_source)
                data_dict = io.read_a_pkl(folder_path, clip_name)
                
                if data_dict is None:
                    print(f"Could not load data for {filename}, skipping...")
                    continue
                    
                video_data = data_dict.get('frames')
                if video_data is None:
                    print(f"No frames found for {filename}, skipping...")
                    continue
                
                # Convert to tensor and move to device
                input_video = torch.from_numpy(np.float32(video_data)).to(model.device)
                if len(input_video.shape) == 4:  # [T, H, W, C]
                    input_video = input_video.permute(0, 3, 1, 2)  # [T, C, H, W]
                    input_video = input_video.unsqueeze(0)  # [1, T, C, H, W]
                
                # Generate predictions and COCO annotations
                with torch.no_grad():
                    # Do forward pass to generate SAM masks
                    output = model.forward(
                        input_video, 
                        None,  # No flow
                        None,  # No features  
                        Enable_student=False,
                        epoch=0,
                        active_learning_mode=True,
                        video_name=clip_name,
                        output_dir=os.path.join(self.output_root, "active_learning_coco")
                    )
                
                # Export COCO annotations
                active_learning_dir = os.path.join(self.output_root, "active_learning_coco")
                os.makedirs(active_learning_dir, exist_ok=True)
                
                coco_file = model.export_coco_annotations(input_video, clip_name, active_learning_dir)
                
                if coco_file:
                    video_info['coco_file'] = coco_file
                    processed_videos.append(video_info)
                    print(f" Generated COCO annotations for {clip_name}: {coco_file}")
                else:
                    print(f" Failed to generate COCO annotations for {clip_name}")
                    
            except Exception as e:
                print(f"Error processing video {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Successfully processed {len(processed_videos)} videos for annotation")
        return processed_videos

    def run_active_learning_round(self, dataloader, model, num_videos: int = 10):
        """Run one round of active learning - send random videos to Encord"""
        print(f"\n=== Starting Active Learning Round {self.current_round} ===")
        
        # Process videos and generate COCO annotations
        processed_videos = self.process_videos_for_annotation(dataloader, model, num_videos)
        
        # Upload annotations to Encord
        successful_uploads = 0
        for video_info in processed_videos:
            try:
                success = self.upload_coco_annotations(
                    coco_file_path=video_info['coco_file'],
                    round_number=self.current_round
                )
                
                if success:
                    successful_uploads += 1
                    print(f" Successfully uploaded annotations for {video_info['clip_name']}")
                else:
                    print(f" Failed to upload annotations for {video_info['clip_name']}")
                    
            except Exception as e:
                print(f"Error uploading annotations for {video_info['clip_name']}: {e}")
                continue
        
        print(f"Active Learning Round {self.current_round} completed:")
        print(f"  - Videos processed: {len(processed_videos)}")
        print(f"  - Successful uploads: {successful_uploads}")
        
        # Save checkpoint and increment round
        if successful_uploads > 0:
            self.save_round_checkpoint(model, self.current_round)
            self.current_round += 1
            
            # Wait for human correction
            should_continue = self.wait_for_human_correction(self.current_round)
            if not should_continue:
                return False
        else:
            print("No successful uploads this round. Skipping round increment.")
        
        return True