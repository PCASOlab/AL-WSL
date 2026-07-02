import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from model.vision_transformer import vit_small, vit_base
import os
from model.model_3dcnn_linear_TC_v3 import _VideoCNN
from model.model_3dcnn_linear_ST_v3 import _VideoCNN_S
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_student,Display_final_SAM
# from working_dir_root import Enable_teacher
from dataset.dataset import class_weights
import numpy as np
from image_operator import basic_operator   
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import Enable_student,Random_mask_temporal_feature,Random_mask_patch_feature,Display_fuse_TC_ST
from working_dir_root import Use_max_error_rejection,DINO_pretrain_root,Batch_size,min_lr,Output_root, selected_data
from model import model_operator
from dataset.dataset import label_mask,Mask_out_partial_label
from torch.optim import lr_scheduler
from model.annotation_utils import (
    binary_mask_to_polygons,
    convert_predictions_to_coco_format,
    create_coco_annotation_json,
    create_encord_annotation_json,
    _get_color_for_category
)
import random
import json
from datetime import datetime

if Evaluation == True:
    learningR=0
    Weight_decay=0
def select_gpus(gpu_selection):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of GPUs available:", num_gpus)
        if gpu_selection == "all":
            device = torch.device("cuda" if num_gpus > 0 else "cpu")
            
        elif gpu_selection.isdigit():
            gpu_index = int(gpu_selection)
            device = torch.device("cuda:" + gpu_selection if gpu_index < num_gpus else "cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device
# --- NEW: backbone wrapper for DINO / DINOv2 / DINOv3 ---
import math
import yaml

CONFIG_PATH = "config.yaml"   # Change if needed
with open(CONFIG_PATH, 'r') as f:
    yaml_config = yaml.safe_load(f)

# Override selected variables from YAML
mask_loss_weight = yaml_config['wsl']['mask_loss']


class DinoBackbone(nn.Module):
    """
    Unifies feature extraction for DINO (v1 via timm), DINOv2 (torch.hub), and DINOv3 (HF transformers).
    Returns spatial feature maps [B, C, H, W].
    """
    def __init__(self, version="dinov3", variant=None, device="cuda"):
        super().__init__()
        self.version = version.lower()
        self.variant = variant
        self.device = device

        if self.version == "dinov2":
            # ViT-B/14 by default
            name = (variant or "dinov2_vitb14").lower()
            self.model = torch.hub.load('facebookresearch/dinov2', name).to(device).eval()

            size_key = name.split("vit")[1][0]  # 's' | 'b' | 'l' | 'g'
            self.out_channels = {"s": 384, "b": 768, "l": 1024, "g": 1536}[size_key]

            self.patch = 14
            g = 224 // self.patch  # 224/14 = 16
            self.grid_hw = (g, g)  # (16, 16) for 224x224 input
            self._kind = "dinov2_torchhub"

        elif self.version == "dino":
            # DINO v1 via timm; default ViT-S/16
            import timm
            name = variant or "vit_small_patch16_224.dino"
            self.model = timm.create_model(name, pretrained=True).to(device).eval()
            # infer out_channels from model embed dim
            self.out_channels = self.model.num_features
            # get patch size from name
            if "patch14" in name: self.patch = 14
            elif "patch16" in name: self.patch = 16
            else: self.patch = 16
            g = 224 // self.patch
            self.grid_hw = (g, g)
            self._kind = "dino_timm"

        elif self.version == "dinov3":
            # DINOv3 via HuggingFace Transformers; default to BASE model (768 features)
             
    # DINOv3 via HuggingFace Transformers; default ViT-B/16 (768-dim)
            from transformers import AutoModel
            name = "facebook/dinov3-vits16-pretrain-lvd1689m" 
            REPO_DIR = DINO_pretrain_root + "dinov3"
            self.model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights = DINO_pretrain_root + "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" )

            # Get hidden size and patch size from config
            self.out_channels = self.model.embed_dim  # typically 768 for ViT-B
            self.patch = self.model.patch_size
            g = 224 // self.patch
            self.grid_hw = (g, g)
            self._kind = "dinov3_hf"
        else:
            raise ValueError(f"Unknown dino_version: {version}")

        # freeze by default (you can unfreeze later if you want to finetune)
      
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224] -> returns [B, C, H, W]
        """
        if self._kind == "dinov2_torchhub":
            # DINOv2 torchhub returns dict from forward_features; .forward(x) returns cls only
            feats = self.model.forward_features(x)
            # x_norm_patchtokens: [B, N, C], N=H*W
            tokens = feats["x_norm_patchtokens"]  # (B, N, C)
            B, N, C = tokens.shape
            H, W = self.grid_hw
            assert N == H*W, f"Unexpected #tokens {N} for grid {H}x{W}"
            fmap = tokens.transpose(1, 2).reshape(B, C, H, W)
            return fmap

        elif self._kind == "dino_timm":
            # timm ViT forward_features often returns a dict or tensor; normalize to patch tokens
            if hasattr(self.model, "forward_features"):
                feats = self.model.forward_features(x)
                # Many timm ViTs return a dict with 'tokens' or 'x' (B, N+1, C)
                if isinstance(feats, dict):
                    if "tokens" in feats:
                        tokens = feats["tokens"]  # (B, N+1, C)
                    elif "x" in feats:
                        tokens = feats["x"]
                    else:
                        # fallback: run model.get_intermediate_layers if available
                        raise RuntimeError("Unexpected timm features structure; set a different variant or adapt here.")
                else:
                    tokens = feats  # sometimes raw tokens

                # strip class token if present (N+1)
                if tokens.shape[1] == (self.grid_hw[0]*self.grid_hw[1] + 1):
                    tokens = tokens[:, 1:, :]
                B, N, C = tokens.shape
                H, W = self.grid_hw
                assert N == H*W, f"Unexpected #tokens {N} for grid {H}x{W}"
                return tokens.transpose(1, 2).reshape(B, C, H, W)
            else:
                raise RuntimeError("timm model missing forward_features")

        elif self._kind == "dinov3_hf":
            feats = self.model.forward_features(x)
            tokens = feats["x_norm_patchtokens"]  # (B, N, C)
            B, N, C = tokens.shape
            H, W = self.grid_hw
            assert N == H*W, f"Unexpected #tokens {N} for grid {H}x{W}"
            fmap = tokens.transpose(1, 2).reshape(B, C, H, W)
            return fmap

class _Model_infer(nn.Module):
    def __init__(self, GPU_mode =True,num_gpus=1,Enable_teacher=True,Using_spatial_conv=True,Student_be_teacher=False,gpu_selection = "all",pooling="rank",TPC=False):
        super(_Model_infer, self).__init__()
        
        if GPU_mode ==True:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = select_gpus(gpu_selection)
        else:
            device = torch.device("cpu")
        self.device = device
        self.inter_bz =100* Batch_size
        self.TPC = TPC
        dino_version="dinov3"
        dino_variant = None
         # --- REPLACE old DINO encoder with the new wrapper ---
        self.backbone = DinoBackbone(version=dino_version, variant=dino_variant, device=device)
        inputC = self.backbone.out_channels  # e.g., 384 for ViT-S, 768 for ViT-B
        self.input_size = 224

        # Your heads now use the detected channel size
        if Student_be_teacher == False:
            self.VideoNets = _VideoCNN(inputC=inputC, pooling=pooling)
        else:
            self.VideoNets = _VideoCNN_S(inputC=inputC, Using_spatial_conv=Using_spatial_conv, pooling=pooling)
        self.VideoNets_S = _VideoCNN_S(inputC=inputC, pooling=pooling)
        self.input_size = 224
        resnet18 = models.resnet18(pretrained=True)
        self.gradcam = None
        self.Enable_teacher = Enable_teacher
        self.dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
        self.output_root = Output_root+ "temporal_consistent/" + self.dataset_tag +  "DINOv3"+ "/"
        # Remove the fully connected layers at the end
        partial = nn.Sequential(*list(resnet18.children())[0:-2])
        
        # Modify the last layer to produce the desired feature map size
        self.resnet = nn.Sequential(
            partial,
            nn.ReLU()
        )
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        
        if GPU_mode == True:
            if num_gpus > 1 and gpu_selection == "all":
                
                self.VideoNets = torch.nn.DataParallel(self.VideoNets)
                self.VideoNets_S = torch.nn.DataParallel(self.VideoNets_S)


                self.resnet  = torch.nn.DataParallel(self.resnet )
                self.Vit_encoder   = torch.nn.DataParallel(self.Vit_encoder  )
                self.sam_model  = torch.nn.DataParallel(self.sam_model )
        self.VideoNets.to(device)
        self.VideoNets_S.to(device)
        self.backbone.to(device)


        self.resnet .to(device)
      
        if Evaluation:
            pass
            
            for p in self.backbone.parameters():
                p.requires_grad = False

        else:
            self.VideoNets.train(True)
            self.VideoNets_S.train(True)

        
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        
        self.customeBCE =  torch.nn.MSELoss()
        self.customeBCE_S =  torch.nn.MSELoss()
        self.customeBCE_mask = torch.nn.MSELoss( ) 


        self.optimizer = torch.optim.Adam ([
            #BACKBONE IS FROZEN WHEN COMMENTED
            #  {'params': self.backbone.parameters(),'lr': 0.1*learningR},
        {'params': self.VideoNets.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ], weight_decay=Weight_decay) #
        self.optimizer_s = torch.optim.Adam ([
        {'params': self.VideoNets_S.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ],weight_decay=Weight_decay)
        # if GPU_mode ==True:
        #     if num_gpus > 1:
        #         self.optimizer = torch.nn.DataParallel(optself.optimizerimizer)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10, eta_min=min_lr, last_epoch=-1)  # Optional parameters explained below
        self.schedulers = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_s, 10, eta_min=min_lr, last_epoch=-1)  # Optional parameters explained below

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self, input, input_flows, features, Enable_student, epoch=0, active_learning_mode=False, video_name=None, output_dir=None):
        bz, ch, D, H, W = input.size()
        activationLU = nn.ReLU()


        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        if Load_feature == False or features == None:
            flattened_tensor = self.input_resample.permute(0,2,1,3,4)
            flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
            flattened_tensor = (flattened_tensor-124.0)/60.0

            num_chunks = (bz*D + self.inter_bz - 1) // self.inter_bz
        
            # List to store predicted tensors
            predicted_tensors = []
            
            # Chunk input tensor and predict
            # with torch.no_grad():
            feats_list = []
            for i in range(num_chunks):
                s, e = i * self.inter_bz, min((i + 1) * self.inter_bz, bz * D)
                x_chunk = flattened_tensor[s:e]
                fmap = self.backbone(x_chunk)            # <<< NEW
                feats_list.append(fmap)
                    # torch.cuda.empty_cache()
               
        
            # Concatenate predicted tensors along batch dimension
            concatenated = torch.cat(feats_list, dim=0)       # [bz*D, C, h, w]
            C, h, w = concatenated.size(1), concatenated.size(2), concatenated.size(3)
            self.f = concatenated.reshape(bz, D, C, h, w).permute(0, 2, 1, 3, 4)  # [B, C, D, H, W]
        else:
            with torch.no_grad():
                self.f = features
        flag =random. choice([True, True])
        self.fm =self.f
        if  Random_mask_temporal_feature == True:
            self.fm =   model_operator.random_mask_out_dimension(self.fm, 0.5, dim=2)
        if  Random_mask_patch_feature == True:
            self.fm =   model_operator.hide_patch(self.fm )

        self.output, self.slice_valid, self. cam3D= self.VideoNets(self.fm,flag,epoch=epoch)
        with torch.no_grad():
            self.slice_hard_label,self.binary_masks= model_operator.CAM_to_slice_hardlabel(activationLU(self.cam3D),self.output)
            self.cam3D_target = self.cam3D.detach().clone()
       
        if Enable_student:
            self.output_s,self.slice_valid_s,self.cam3D_s = self.VideoNets_S(self.f,flag,epoch=epoch)
        
        with torch.no_grad():
            output = self.output.detach().clone()
        if Display_student:
            with torch.no_grad():
                if Display_fuse_TC_ST == True:
                    # self.cam3D = (self.cam3D_s.detach().clone() + self.cam3D)/2
                    if hasattr(self, 'cam3D_s') and self.cam3D_s is not None:
                        self.cam3D = (self.cam3D_s.detach().clone() + self.cam3D)/2
                    else:
                        self.cam3D = self.cam3D  # Keep original
                else:
                    # self.cam3D = self.cam3D_s.detach().clone()
                    if hasattr(self, 'cam3D_s') and self.cam3D_s is not None:
                        self.cam3D = self.cam3D_s.detach().clone()
                        
                        #output = self.output_s.detach().clone()
            
        self.raw_cam = self.cam3D.detach().clone()

        ###################################################################################
        #         POST PROCESSING FOR CAM TO MASK 
        #         post_processed_masks (Batch, class, D, H, W) value ~[0, 1]  
        ###################################################################################
        if Display_final_SAM:
            with torch.no_grad():
                post_processed_masks=model_operator.Cam_mask_post_process(activationLU(self.cam3D), input,output)
           
            self.cam3D = post_processed_masks.to(self.device)
       
        with torch.no_grad():
            self.final_output = output.detach().clone()
            self.direct_frame_output = None 
    
    def loss_of_one_scale(self,output,label,BCEtype = 1):
        out_logits = output.view(label.size(0), -1)
        bz,length = out_logits.size()

        label_mask_torch = torch.tensor(label_mask, dtype=torch.float32)
        label_mask_torch = label_mask_torch.repeat(bz, 1)
        label_mask_torch = label_mask_torch.to(self.device)
        if BCEtype == 1:
            loss = self.customeBCE(out_logits * label_mask_torch, label * label_mask_torch)

            
        else:
            loss = self.customeBCE_S(out_logits * label_mask_torch, label * label_mask_torch)
        return loss
    def loss_of_one_scale_with_error_filter(self, output, label, BCEtype=1):
        out_logits = output.view(label.size(0), -1)
        bz, length = out_logits.size()

        label_mask_torch = torch.tensor(label_mask, dtype=torch.float32)
        label_mask_torch = label_mask_torch.repeat(bz, 1)
        label_mask_torch = label_mask_torch.to(self.device)

        error_rejection_mask_torch = label_mask_torch + 1
        error_rejection_mask_torch = (error_rejection_mask_torch > 0.5)*1.0

        # Assign 0 value to max false negative index, 1 kept for others
        for i in range(bz):
            initial_error_batch = torch.abs(out_logits[i] - label[i])
            max_err_index_batch = torch.argmax(initial_error_batch)
            if label[i, max_err_index_batch] == 1:
                error_rejection_mask_torch[i, max_err_index_batch] = 0

        out_logits_masked = out_logits * label_mask_torch * error_rejection_mask_torch
        label_masked = label * label_mask_torch * error_rejection_mask_torch

        if BCEtype == 1:
            loss = self.customeBCE(out_logits_masked, label_masked)
        else:
            loss = self.customeBCE_S(out_logits_masked, label_masked)

        return loss
    def optimization(self, label,Enable_student, input_masks=None):
        self.optimizer.zero_grad()
        self.optimizer_s.zero_grad()
        
        # Weak loss (classification loss)
        if Use_max_error_rejection == False:
            self.loss = self.loss_of_one_scale(self.output, label, BCEtype=1)
        else:
            self.loss = self.loss_of_one_scale_with_error_filter(self.output, label, BCEtype=1)

        total_loss = self.loss  # Start with weak loss
        
        # Mask loss if masks are provided
        if input_masks is not None:
            # Ensure input_masks has shape [B, C, D, H, W]
            if len(input_masks.shape) == 4:
                input_masks = input_masks.unsqueeze(1)  # Add channel dimension
            
            # Get target dimensions
            target_shape = input_masks.shape[-3:]  # [D, H, W]
            
            # Process teacher masks
            if hasattr(self, 'cam3D'):
                predicted_masks = self.cam3D
                if predicted_masks.dim() == 4:
                    predicted_masks = predicted_masks.unsqueeze(1)  # [B, D, H, W] -> [B, 1, D, H, W]
                
                if predicted_masks.shape[-3:] != target_shape:
                    predicted_masks = F.interpolate(
                        predicted_masks,
                        size=target_shape,
                        mode='trilinear',
                        align_corners=False
                    )

                # IMAGE-LEVEL CONVERSION:
                # For each frame (image), determine if tool is present or absent
                # GT: 1 if ANY pixel in that frame has the tool, 0 otherwise
                gt_frame_presence = (input_masks.sum(dim=(-2, -1)) > 0).float()  # [B, C, D]
                # Predicted: Average the CAM values across spatial dimensions for each frame
                # This gives us a "confidence" that the tool is present in that frame
                pred_frame_confidence = predicted_masks.mean(dim=(-2, -1))  # [B, C, D]
                
                # Image-level binary cross entropy loss
                # We're comparing: does the model think tool is in this frame vs. GT says tool is in frame
                self.mask_loss = F.binary_cross_entropy_with_logits(
                    pred_frame_confidence,  # Raw logits (not sigmoidized)
                    gt_frame_presence,      # Binary ground truth
                    reduction='mean'
                )
                
                # # Calculate image-level MSE loss
                total_loss += self.mask_loss * mask_loss_weight  # Lower weight for image-level loss
            else:
                self.mask_loss = 0
        else:
            self.mask_loss=0
        # Backpropagate total loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lossDisplay = total_loss.data.mean()

        if Enable_student:
            self.set_requires_grad(self.VideoNets_S, True)

            # Student weak loss
            if Use_max_error_rejection == False:
                self.loss_s_v = self.loss_of_one_scale(self.output_s, label, BCEtype=2)
            else:
                self.loss_s_v = self.loss_of_one_scale_with_error_filter(self.output_s, label, BCEtype=2)
            
            self.loss_s = self.loss_s_v  # Start with student weak loss
            
            # Student mask loss if masks are provided
            if input_masks is not None and hasattr(self, 'cam3D_s'):
                predicted_masks_s = self.cam3D_s
                
                # Ensure proper dimensions [B, C, D, H, W]
                if predicted_masks_s.dim() == 4:
                    predicted_masks_s = predicted_masks_s.unsqueeze(1)
                
                if predicted_masks_s.shape[-3:] != target_shape:
                    predicted_masks_s = F.interpolate(
                        predicted_masks_s,
                        size=target_shape,
                        mode='trilinear',
                        align_corners=False
                    )
                # IMAGE-LEVEL: Binary presence per frame from GT
                gt_frame_presence = (input_masks.sum(dim=(-2, -1)) > 0).float()  # [B, C, D]
                
                # IMAGE-LEVEL: Average predicted confidence per frame
                pred_frame_confidence_s = predicted_masks_s.mean(dim=(-2, -1))  # [B, C, D]
                
                self.mask_loss_s = F.binary_cross_entropy_with_logits(
                    pred_frame_confidence_s,
                    gt_frame_presence,
                    reduction='mean'
                )
                # predicted_image_level_s = predicted_masks_s.mean(dim=[-2, -1])  # [B, C, D]
                # gt_image_level = input_masks.mean(dim=[-2, -1])  # [B, C, D]
                
                # self.mask_loss_s = self.customeBCE_mask(predicted_image_level_s, gt_image_level)
                self.loss_s += self.mask_loss_s * mask_loss_weight # Lower weight for image-level loss
            else:
                self.mask_loss_s = 0
            # Pixel-wise consistency loss (if TPC is enabled)
            bz, ch, D, H, W = self.cam3D_s.size()
            
            label_valid_repeat = label.reshape(bz, ch, 1, 1, 1).repeat(1, 1, D, H, W)
            valid_masks_repeated = self.slice_hard_label.repeat(1, 1, 1, H, W)
            
            if self.TPC == True:
                predit_mask = self.cam3D_s * valid_masks_repeated
                target_mask = self.cam3D_target * label_valid_repeat * valid_masks_repeated
            else:
                predit_mask = self.cam3D_s 
                target_mask = self.cam3D_target 
            
            self.loss_s_pix = self.customeBCE_mask(predit_mask, target_mask)

            if self.Enable_teacher:
                self.loss_s += 0.00001 * self.loss_s_pix  # Add pixel consistency loss
            
            # Backpropagate student loss
            self.loss_s.backward()
            self.optimizer_s.step()
            self.lossDisplay_s = self.loss_s.data.mean()

    def optimization_slicevalid(self):

        pass
    def convert_predictions_to_coco_format(self, cam3D, model_output, video_name, image_size=(224, 224)):
        """Convert model predictions to COCO format"""
        
        # Initialize COCO structure
        coco_data = {
            "info": {
                "description": f"Active Learning Predictions - {video_name}",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        # Define tool categories
        tool_categories = [
            {"id": 1, "name": "Grasper", "supercategory": "surgical_tool"},
            {"id": 2, "name": "Bipolar", "supercategory": "surgical_tool"},
            {"id": 3, "name": "Hook", "supercategory": "surgical_tool"},
            {"id": 4, "name": "Scissors", "supercategory": "surgical_tool"},
            {"id": 5, "name": "Clipper", "supercategory": "surgical_tool"},
            {"id": 6, "name": "Irrigator", "supercategory": "surgical_tool"},
            {"id": 7, "name": "SpecimenBag", "supercategory": "surgical_tool"}
        ]
        
        coco_data["categories"] = tool_categories
        
        # Process predictions
        annotation_id = 1
        
        # Handle different tensor shapes
        if len(cam3D.shape) == 5:  # Batch format: [B, C, D, H, W]
            batch_size, num_classes, depth, height, width = cam3D.shape
            
            for batch_idx in range(batch_size):
                for frame_idx in range(depth):
                    # Create image entry
                    image_id = batch_idx * depth + frame_idx + 1
                    coco_data["images"].append({
                        "id": image_id,
                        "file_name": f"{video_name}_frame_{frame_idx:06d}.jpg",
                        "width": image_size[0],
                        "height": image_size[1],
                        "video_name": video_name,
                        "frame_index": frame_idx
                    })
                    
                    # Process each class
                    for class_idx in range(min(num_classes, len(tool_categories))):
                        class_id = class_idx + 1
                        class_name = tool_categories[class_idx]["name"]
                        
                        # Get CAM for this class and frame
                        class_cam = cam3D[batch_idx, class_idx, frame_idx]
                        
                        # Convert to binary mask
                        binary_mask = (class_cam > 0.5).float()
                        
                        if binary_mask.sum() > 0:  # Only add if mask exists
                            # Resize mask to image size if needed
                            if binary_mask.shape != image_size:
                                binary_mask_resized = F.interpolate(
                                    binary_mask.unsqueeze(0).unsqueeze(0), 
                                    size=image_size, 
                                    mode='nearest'
                                ).squeeze()
                            else:
                                binary_mask_resized = binary_mask
                            
                            # Convert to numpy for polygon conversion
                            binary_mask_np = binary_mask_resized.cpu().numpy()
                            
                            # Convert to polygons
                            polygons = self.binary_mask_to_polygons(binary_mask_np)
                            
                            # Calculate confidence
                            if model_output is not None:
                                confidence = float(torch.sigmoid(model_output[batch_idx, class_idx]).mean().item())
                            else:
                                confidence = float(torch.sigmoid(class_cam).mean().item())
                            
                            # Add annotation for each polygon
                            for polygon in polygons:
                                if len(polygon) >= 6:  # Need at least 3 points (x,y,x,y,x,y)
                                    # Calculate bounding box
                                    x_coords = polygon[0::2]
                                    y_coords = polygon[1::2]
                                    bbox = [
                                        float(min(x_coords)),
                                        float(min(y_coords)),
                                        float(max(x_coords) - min(x_coords)),
                                        float(max(y_coords) - min(y_coords))
                                    ]
                                    
                                    # Calculate area
                                    area = float(bbox[2] * bbox[3])
                                    
                                    coco_data["annotations"].append({
                                        "id": annotation_id,
                                        "image_id": image_id,
                                        "category_id": class_id,
                                        "bbox": bbox,
                                        "area": area,
                                        "segmentation": [polygon],
                                        "iscrowd": 0,
                                        "confidence": confidence,
                                        "attributes": {
                                            "tool_type": class_name,
                                            "frame_index": frame_idx
                                        }
                                    })
                                    annotation_id += 1
        
        print(f"Created COCO data with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
        return coco_data

    def binary_mask_to_polygons(self, binary_mask):
        """Convert binary mask to polygon coordinates"""
        try:
            import cv2
            polygons = []
            
            # Ensure binary mask is uint8
            mask_uint8 = (binary_mask * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) >= 3:  # Need at least 3 points for a polygon
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Flatten to [x,y,x,y,...] format
                    polygon = approx.flatten().tolist()
                    polygons.append(polygon)
            
            return polygons
            
        except Exception as e:
            print(f"Error converting mask to polygons: {e}")
            return []

    def extract_predictions_for_active_learning(self, input_videos, video_name):
        """Extract predictions for active learning uncertainty calculation"""
        with torch.no_grad():
            # Ensure we have predictions
            if not hasattr(self, 'output'):
                self.forward(input_videos, None, None, Enable_student=False)
            
            predictions = {
                'video_name': video_name,
                'classification_raw': self.output.detach().cpu().numpy(),
                'classification_probs': torch.sigmoid(self.output).detach().cpu().numpy(),
                'cam_maps': self.cam3D.detach().cpu().numpy() if hasattr(self, 'cam3D') else None,
                'timestamp': datetime.now().isoformat()
            }
            
            return predictions

    def calculate_uncertainty(self, predictions):
        """Calculate uncertainty scores from predictions"""
        try:
            classification_probs = predictions['classification_probs']
            
            uncertainty_scores = {}
            
            # Method 1: Prediction entropy
            entropy = -np.sum(classification_probs * np.log(classification_probs + 1e-8), axis=1)
            uncertainty_scores['entropy'] = float(entropy.mean())
            
            # Method 2: Max confidence (1 - max_prob)
            max_probs = np.max(classification_probs, axis=1)
            uncertainty_scores['confidence'] = float((1 - max_probs).mean())
            
            # Method 3: CAM consistency (variance across frames)
            if predictions['cam_maps'] is not None:
                cam_maps = predictions['cam_maps']
                if len(cam_maps.shape) >= 4:  # [B, C, D, H, W] or [C, D, H, W]
                    cam_variance = np.var(cam_maps, axis=2)  # Variance across temporal dimension
                    uncertainty_scores['cam_variance'] = float(cam_variance.mean())
            
            # Overall uncertainty score (weighted combination)
            overall_uncertainty = (
                uncertainty_scores.get('entropy', 0) * 0.4 +
                uncertainty_scores.get('confidence', 0) * 0.4 +
                uncertainty_scores.get('cam_variance', 0) * 0.2
            )
            uncertainty_scores['overall'] = overall_uncertainty
            
            return uncertainty_scores
            
        except Exception as e:
            print(f"Error calculating uncertainty: {e}")
            return {'overall': 0.0}

    def export_coco_annotations(self, input_videos, video_name, output_dir, round_number):
        """Export model predictions as COCO format annotations for active learning"""
        
        if output_dir is None:
            output_dir = os.path.join(self.output_root, "active_learning_coco")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"Exporting annotations for {video_name}")
            
            # # Ensure we have the latest predictions
            # if not hasattr(self, 'cam3D') or self.cam3D is None:
            #     print("No CAM maps available for annotation export")
            #     return None
            
            # # Generate post-processed SAM masks
            # activationLU = nn.ReLU()
            # with torch.no_grad():
            #     post_processed_masks = model_operator.Cam_mask_post_process(
            #         activationLU(self.cam3D), 
            #         input_videos,  # or pass actual input if needed
            #         self.output
            #     )
            import torch.nn as nn
            
            with torch.no_grad():
                # Make sure we have raw CAM
                if not hasattr(self, 'cam3D') or self.cam3D is None:
                    print("No CAM available for active learning")
                    return None
                
                # Generate post-processed masks
                activationLU = nn.ReLU()
                post_processed_masks = model_operator.Cam_mask_post_process(
                    activationLU(self.cam3D), 
                    input_videos,
                    self.output
                )
            
            
            print(f"Generated post-processed masks: {post_processed_masks.shape}")
            # Save mask visualizations for debugging
            round_dir = os.path.join(output_dir, f"AL_round_{round_number}")
            os.makedirs(round_dir, exist_ok=True)
            before_encord_dir = os.path.join(round_dir, "before_encord")
            os.makedirs(before_encord_dir, exist_ok=True)
            # test_dir = "data/AL_run{round_number}"
            # os.makedirs(test_dir, exist_ok=True)
            
            # Visualize masks before conversion
            self.save_mask_visualizations(
                post_processed_masks, 
                input_videos, 
                video_name, 
                before_encord_dir
            )
            # Convert post-processed masks to COCO format using annotation_utils functions
            from model.annotation_utils import convert_predictions_to_coco_format, create_coco_annotation_json
            
            # Get frame predictions with polygons from SAM masks
            frame_predictions = convert_predictions_to_coco_format(
                post_processed_masks, 
                self.output, 
                video_name
            )
            
            print(f"Converted to {len(frame_predictions)} frame predictions")
            
            # Create COCO annotation file
            coco_file = create_coco_annotation_json(
                video_name, 
                frame_predictions, 
                output_dir,
                image_size=(post_processed_masks.shape[-1], post_processed_masks.shape[-2])
            )
            # coco_file = create_coco_annotation_json(
            #     video_name, 
            #     frame_predictions, 
            #     output_dir,
            #     image_size=(post_processed_masks.shape[-1], post_processed_masks.shape[-2])
            # )
            
            print(f"Exported COCO annotations to: {coco_file}")
            return coco_file
            
        except Exception as e:
            print(f"Error exporting annotations for {video_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_mask_visualizations(self, cam3D, input_videos, video_name, output_dir, epoch=0):
        """
        Save visualizations of masks for debugging before sending to Encord
        Creates color-coded masks overlaid on video frames
        """
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Create output directory
        viz_dir = os.path.join(output_dir, video_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Define colors for different tool classes
        tool_colors = [
            (255, 0, 0),    # Red - Grasper
            (0, 255, 0),    # Green - Bipolar
            (0, 0, 255),    # Blue - Hook
            (255, 255, 0),  # Cyan - Scissors
            (255, 0, 255),  # Magenta - Clipper
            (0, 255, 255),  # Yellow - Irrigator
            (128, 0, 128)   # Purple - SpecimenBag
        ]
        
        tool_names = [
            "Grasper", "Bipolar", "Hook", "Scissors", 
            "Clipper", "Irrigator", "SpecimenBag"
        ]
        
        # Handle different tensor shapes
        if len(cam3D.shape) == 5:  # [B, C, D, H, W]
            batch_size, num_classes, depth, height, width = cam3D.shape
            
            # For visualization, we'll process each batch item
            for batch_idx in range(batch_size):
                for frame_idx in range(depth):
                    # Get the frame from input video
                    if input_videos is not None:
                        # Convert frame tensor to numpy image
                        frame = input_videos[batch_idx, :, frame_idx].cpu().numpy()
                        frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
                        
                        # Normalize to 0-255
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                        
                        # Convert to BGR for OpenCV
                        if frame.shape[-1] == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        # Create blank frame if no input video
                        frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Create overlay for masks
                    overlay = frame_bgr.copy()
                    
                    # Process each class
                    for class_idx in range(min(num_classes, len(tool_colors))):
                        # Get CAM for this class and frame
                        class_cam = cam3D[batch_idx, class_idx, frame_idx]
                        
                        # Convert to binary mask
                        binary_mask = (class_cam > 0.75).float().cpu().numpy()
                        
                        if binary_mask.sum() > 0:
                            # Create colored mask
                            color = tool_colors[class_idx]
                            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                            colored_mask[binary_mask > 0] = color
                            
                            # Blend with overlay
                            mask_alpha = 0.3
                            overlay = cv2.addWeighted(overlay, 1, colored_mask, mask_alpha, 0)
                            
                            # Draw contours
                            contours, _ = cv2.findContours(
                                binary_mask.astype(np.uint8), 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(overlay, contours, -1, color, 1)
                    
                    # Save visualization
                    frame_filename = f"frame_{frame_idx:04d}_mask.png"
                    frame_path = os.path.join(viz_dir, frame_filename)
                    cv2.imwrite(frame_path, overlay)
                    
                    # Create a legend image
                    if frame_idx == 0:
                        legend = np.zeros((100, 400, 3), dtype=np.uint8)
                        for i, (color, name) in enumerate(zip(tool_colors[:num_classes], tool_names[:num_classes])):
                            y_start = i * 20 + 10
                            cv2.rectangle(legend, (10, y_start), (30, y_start + 10), color, -1)
                            cv2.putText(legend, name, (40, y_start + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        legend_path = os.path.join(viz_dir, "legend.png")
                        cv2.imwrite(legend_path, legend)
        
        print(f"Saved mask visualizations to: {viz_dir}")
        return viz_dir
    def visualize_masks_on_frames(self, input_video, video_name, frame_indices=None, 
                            class_indices=None, output_dir=None, alpha=0.5):
        """
        Visualize both raw CAM and post-processed masks overlaid on original frames
        FIXED: Raw CAM is properly upsampled to match frame resolution
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        from matplotlib import cm
        import torch.nn.functional as F
        
        if output_dir is None:
            output_dir = os.path.join(self.output_root, "mask_overlays")
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure we have CAM available
        if not hasattr(self, 'cam3D') or self.cam3D is None:
            print("No CAM available. Run forward() first.")
            return
        
        # Get dimensions
        B, C, D, cam_H, cam_W = self.cam3D.shape
        print(f"CAM resolution: {cam_H}x{cam_W}")
        
        # Default frame indices (first, middle, last)
        if frame_indices is None:
            frame_indices = [0, D//2, D-1] if D > 2 else [0]
        
        # Default class indices (first 3 classes)
        if class_indices is None:
            class_indices = list(range(min(3, C)))
        
        # Tool names and colors
        tool_names = ["Grasper", "Bipolar", "Hook", "Scissors", 
                    "Clipper", "Irrigator", "SpecimenBag"]
        tool_colors = [
            (255, 0, 0),    # Red - Grasper
            (0, 255, 0),    # Green - Bipolar
            (0, 0, 255),    # Blue - Hook
            (255, 255, 0),  # Cyan - Scissors
            (255, 0, 255),  # Magenta - Clipper
            (0, 255, 255),  # Yellow - Irrigator
            (128, 0, 128)   # Purple - SpecimenBag
        ]
        
        # Prepare input video for display
        input_np = input_video.cpu().numpy()
        if len(input_np.shape) == 5:  # [B, C, D, H, W]
            input_np = input_np[0]  # Take first batch
            input_np = np.transpose(input_np, (1, 2, 3, 0))  # [D, H, W, C]
            frame_H, frame_W = input_np.shape[1], input_np.shape[2]
        elif len(input_np.shape) == 4:  # [C, D, H, W]
            input_np = np.transpose(input_np, (1, 2, 3, 0))  # [D, H, W, C]
            frame_H, frame_W = input_np.shape[1], input_np.shape[2]
        
        print(f"Frame resolution: {frame_H}x{frame_W}")
        
        # Normalize input to 0-255
        if input_np.max() <= 1.0:
            input_np = (input_np * 255).astype(np.uint8)
        else:
            input_np = input_np.astype(np.uint8)
        
        # Generate post-processed masks
        activationLU = nn.ReLU()
        with torch.no_grad():
            post_processed_masks = model_operator.Cam_mask_post_process(
                activationLU(self.cam3D), 
                input_video,
                self.output
            )
        
        print(f"Post-processed mask resolution: {post_processed_masks.shape[-2]}x{post_processed_masks.shape[-1]}")
        
        # Process each frame
        for frame_idx in frame_indices:
            if frame_idx >= D:
                continue
                    
            # Create figure with 3 columns: Original, Raw CAM Overlay, Post-processed Overlay
            fig, axes = plt.subplots(len(class_indices), 4, figsize=(20, 5*len(class_indices)))
            
            if len(class_indices) == 1:
                axes = axes.reshape(1, -1)
            
            # Get original frame
            orig_frame = input_np[frame_idx]
            if orig_frame.shape[-1] == 3:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            else:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2RGB)
            
            for row, class_idx in enumerate(class_indices):
                # Get class name
                class_name = tool_names[class_idx] if class_idx < len(tool_names) else f"Class_{class_idx}"
                
                # Column 1: Original frame
                axes[row, 0].imshow(orig_frame_rgb)
                axes[row, 0].set_title(f'Original Frame {frame_idx}\n{class_name}')
                axes[row, 0].axis('off')
                
                # Get raw CAM (small resolution)
                raw_cam_small = self.cam3D[0, class_idx, frame_idx].cpu().numpy()
                
                # Column 2: Raw CAM heatmap (small resolution)
                im1 = axes[row, 1].imshow(raw_cam_small, cmap='jet', vmin=0, vmax=raw_cam_small.max())
                axes[row, 1].set_title(f'Raw CAM (Small)\n{cam_H}x{cam_W}\nMean: {raw_cam_small.mean():.3f}')
                axes[row, 1].axis('off')
                plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
                
                # Column 3: Raw CAM upsampled overlay
                # Upsample raw CAM to match frame resolution
                raw_cam_tensor = self.cam3D[0, class_idx, frame_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                raw_cam_upsampled = F.interpolate(
                    raw_cam_tensor, 
                    size=(frame_H, frame_W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze().cpu().numpy()
                
                # Normalize for visualization
                raw_cam_normalized = (raw_cam_upsampled - raw_cam_upsampled.min()) / (raw_cam_upsampled.max() - raw_cam_upsampled.min() + 1e-8)
                
                # Create heatmap at full resolution
                heatmap = cv2.applyColorMap((raw_cam_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Overlay heatmap on original frame
                overlay_raw = cv2.addWeighted(orig_frame_rgb, 1-alpha, heatmap, alpha, 0)
                
                axes[row, 2].imshow(overlay_raw)
                axes[row, 2].set_title(f'Raw CAM Upsampled\n{frame_H}x{frame_W}\nMean: {raw_cam_upsampled.mean():.3f}')
                axes[row, 2].axis('off')
                
                # Add contours for binary mask (>0.5 threshold on upsampled)
                binary_raw = (raw_cam_upsampled > 0.5).astype(np.uint8)
                if binary_raw.sum() > 0:
                    contours, _ = cv2.findContours(binary_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        axes[row, 2].plot(contour[:, 0, 0], contour[:, 0, 1], 
                                        'w-', linewidth=1, alpha=0.8)
                
                # Column 4: Post-processed mask overlay
                post_mask = post_processed_masks[0, class_idx, frame_idx].cpu().numpy()
                
                # Create binary mask for overlay
                binary_post = (post_mask > 0.5).astype(np.uint8)
                
                # Create colored mask (use tool-specific color if available)
                color = tool_colors[class_idx] if class_idx < len(tool_colors) else (255, 255, 255)
                colored_mask = np.zeros_like(orig_frame_rgb)
                colored_mask[binary_post > 0] = color
                
                # Overlay colored mask on original frame
                overlay_post = cv2.addWeighted(orig_frame_rgb, 1-alpha, colored_mask, alpha, 0)
                
                axes[row, 3].imshow(overlay_post)
                axes[row, 3].set_title(f'Post-processed\nActive: {binary_post.sum()} px\n{post_mask.shape[0]}x{post_mask.shape[1]}')
                axes[row, 3].axis('off')
                
                # Add contours for post-processed mask
                if binary_post.sum() > 0:
                    contours, _ = cv2.findContours(binary_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        axes[row, 3].plot(contour[:, 0, 0], contour[:, 0, 1], 
                                        'w-', linewidth=2, alpha=0.8)
            
            plt.suptitle(f'Mask Visualization - {video_name} - Frame {frame_idx}\n'
                        f'CAM: {cam_H}x{cam_W} -> Frame: {frame_H}x{frame_W}', fontsize=16)
            plt.tight_layout()
            
            # Save the visualization
            output_path = os.path.join(output_dir, 
                                    f"{video_name}_frame{frame_idx:04d}_mask_overlay.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved mask overlay visualization to: {output_path}")
        
        # Also create a resolution comparison visualization
        self.create_resolution_comparison(input_np, video_name, self.cam3D, 
                                        post_processed_masks, frame_indices, 
                                        class_indices, output_dir)

    def create_resolution_comparison(self, input_np, video_name, cam3D, post_masks,
                                frame_indices, class_indices, output_dir):
        """
        Create visualization showing resolution differences between CAM and post-processed masks
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        import torch.nn.functional as F
        
        B, C, D, cam_H, cam_W = cam3D.shape
        frame_H, frame_W = input_np.shape[1], input_np.shape[2]
        
        for frame_idx in frame_indices:
            if frame_idx >= D:
                continue
                
            # Get original frame
            orig_frame = input_np[frame_idx]
            if orig_frame.shape[-1] == 3:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            else:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2RGB)
            
            fig, axes = plt.subplots(len(class_indices), 3, figsize=(15, 5*len(class_indices)))
            if len(class_indices) == 1:
                axes = axes.reshape(1, -1)
            
            for row, class_idx in enumerate(class_indices):
                # Column 1: Raw CAM (small) next to upsampled version
                raw_cam_small = cam3D[0, class_idx, frame_idx].cpu().numpy()
                
                # Create comparison image showing small CAM and upsampled
                comparison_img = np.zeros((frame_H, frame_W, 3), dtype=np.uint8)
                
                # Place small CAM in top-left corner
                small_resized = cv2.resize(raw_cam_small, (frame_W//4, frame_H//4))
                small_normalized = (small_resized - small_resized.min()) / (small_resized.max() - small_resized.min() + 1e-8)
                small_heatmap = cv2.applyColorMap((small_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                small_heatmap = cv2.cvtColor(small_heatmap, cv2.COLOR_BGR2RGB)
                
                comparison_img[10:10+frame_H//4, 10:10+frame_W//4] = small_heatmap
                
                # Draw border around small CAM
                cv2.rectangle(comparison_img, (8, 8), (12+frame_W//4, 12+frame_H//4), (255, 255, 255), 2)
                
                # Add text labels
                cv2.putText(comparison_img, f"CAM: {cam_H}x{cam_W}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(comparison_img, f"Frame: {frame_H}x{frame_W}", (frame_W//2, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                axes[row, 0].imshow(comparison_img)
                axes[row, 0].set_title(f'CAM Resolution Comparison\nClass {class_idx}')
                axes[row, 0].axis('off')
                
                # Column 2: Post-processed mask
                post_mask = post_masks[0, class_idx, frame_idx].cpu().numpy()
                axes[row, 1].imshow(post_mask, cmap='jet', vmin=0, vmax=1)
                axes[row, 1].set_title(f'Post-processed Mask\n{post_mask.shape[0]}x{post_mask.shape[1]}')
                axes[row, 1].axis('off')
                
                # Column 3: Overlay comparison
                # Upsample raw CAM
                raw_cam_tensor = cam3D[0, class_idx, frame_idx].unsqueeze(0).unsqueeze(0)
                raw_cam_upsampled = F.interpolate(
                    raw_cam_tensor, 
                    size=(frame_H, frame_W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze().cpu().numpy()
                
                # Create overlay showing both
                overlay_comparison = orig_frame_rgb.copy()
                
                # Add raw CAM contours (green)
                binary_raw = (raw_cam_upsampled > 0.5).astype(np.uint8)
                if binary_raw.sum() > 0:
                    contours_raw, _ = cv2.findContours(binary_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_comparison, contours_raw, -1, (0, 255, 0), 2)
                
                # Add post-processed contours (red)
                binary_post = (post_mask > 0.5).astype(np.uint8)
                if binary_post.sum() > 0:
                    contours_post, _ = cv2.findContours(binary_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_comparison, contours_post, -1, (255, 0, 0), 1)
                
                axes[row, 2].imshow(overlay_comparison)
                axes[row, 2].set_title(f'Overlay Comparison\nGreen: Raw CAM, Red: Post-processed')
                axes[row, 2].axis('off')
                
                # Add legend
                if row == 0:
                    axes[row, 2].text(0.05, 0.95, f"Raw CAM pixels: {binary_raw.sum()}", 
                                    transform=axes[row, 2].transAxes, color='green',
                                    fontsize=10, verticalalignment='top')
                    axes[row, 2].text(0.05, 0.90, f"Post pixels: {binary_post.sum()}", 
                                    transform=axes[row, 2].transAxes, color='red',
                                    fontsize=10, verticalalignment='top')
                    axes[row, 2].text(0.05, 0.85, f"Upscale factor: {frame_H//cam_H}x", 
                                    transform=axes[row, 2].transAxes, color='white',
                                    fontsize=10, verticalalignment='top')
            
            plt.suptitle(f'Resolution Analysis - {video_name} - Frame {frame_idx}', fontsize=16)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"{video_name}_frame{frame_idx:04d}_resolution_analysis.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved resolution analysis to: {output_path}")
    def create_multi_class_overlay(self, input_np, frame_indices, video_name, 
                                cam3D, post_masks, tool_names, tool_colors, 
                                output_dir, alpha=0.4):
        """
        Create visualization with all classes overlaid on same frame
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        
        B, C, D, H, W = cam3D.shape
        
        for frame_idx in frame_indices:
            if frame_idx >= D:
                continue
                
            # Get original frame
            orig_frame = input_np[frame_idx]
            if orig_frame.shape[-1] == 3:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            else:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2RGB)
            
            # Create figure with 2 columns: Raw CAMs, Post-processed masks
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Start with original frame
            raw_overlay = orig_frame_rgb.copy()
            post_overlay = orig_frame_rgb.copy()
            
            # Add legend text
            legend_text = []
            
            # Process each class
            for class_idx in range(min(C, len(tool_names))):
                class_name = tool_names[class_idx]
                color = tool_colors[class_idx] if class_idx < len(tool_colors) else (255, 255, 255)
                
                # Get masks
                raw_mask = cam3D[0, class_idx, frame_idx].cpu().numpy()
                post_mask = post_masks[0, class_idx, frame_idx].cpu().numpy()
                
                # Resize to original frame size
                raw_mask_resized = cv2.resize(raw_mask, (orig_frame_rgb.shape[1], orig_frame_rgb.shape[0]))
                post_mask_resized = cv2.resize(post_mask, (orig_frame_rgb.shape[1], orig_frame_rgb.shape[0]))
                
                # Create binary masks
                binary_raw = (raw_mask_resized > 0.5).astype(np.uint8)
                binary_post = (post_mask_resized > 0.5).astype(np.uint8)
                
                # Add to overlay (raw - heatmap style)
                if binary_raw.sum() > 0:
                    # Create heatmap
                    raw_mask_normalized = (raw_mask_resized - raw_mask_resized.min()) / (raw_mask_resized.max() - raw_mask_resized.min() + 1e-8)
                    heatmap = cv2.applyColorMap((raw_mask_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Only overlay where mask exists
                    mask_indices = binary_raw > 0
                    raw_overlay[mask_indices] = cv2.addWeighted(
                        raw_overlay[mask_indices], 1-alpha, 
                        heatmap[mask_indices], alpha, 0
                    )
                
                # Add to overlay (post - colored mask style)
                if binary_post.sum() > 0:
                    colored_mask = np.zeros_like(orig_frame_rgb)
                    colored_mask[binary_post > 0] = color
                    post_overlay = cv2.addWeighted(post_overlay, 1-alpha, colored_mask, alpha, 0)
                
                # Add to legend
                raw_active = binary_raw.sum()
                post_active = binary_post.sum()
                legend_text.append(f"{class_name}: Raw={raw_active}, Post={post_active}")
            
            # Display raw CAM overlay
            axes[0].imshow(raw_overlay)
            axes[0].set_title(f'Raw CAMs (All Classes) - Frame {frame_idx}')
            axes[0].axis('off')
            
            # Display post-processed overlay
            axes[1].imshow(post_overlay)
            axes[1].set_title(f'Post-processed Masks (All Classes) - Frame {frame_idx}')
            axes[1].axis('off')
            
            # Add legend as text box
            legend_str = "\n".join(legend_text)
            fig.text(0.02, 0.02, legend_str, fontsize=10, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'Multi-Class Overlay - {video_name} - Frame {frame_idx}', fontsize=16)
            plt.tight_layout()
            
            # Save the visualization
            output_path = os.path.join(output_dir, 
                                    f"{video_name}_frame{frame_idx:04d}_all_classes.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved multi-class overlay to: {output_path}")

    def save_frame_by_frame_comparison(self, input_video, video_name, class_idx, output_dir):
        """
        Save frame-by-frame comparison as video (MP4) showing masks overlaid on frames
        """
        import cv2
        import numpy as np
        from tqdm import tqdm
        
        if not hasattr(self, 'cam3D') or self.cam3D is None:
            return
        
        B, C, D, H, W = self.cam3D.shape
        
        # Prepare input video
        input_np = input_video.cpu().numpy()
        if len(input_np.shape) == 5:  # [B, C, D, H, W]
            input_np = input_np[0]  # Take first batch
            input_np = np.transpose(input_np, (1, 2, 3, 0))  # [D, H, W, C]
        elif len(input_np.shape) == 4:  # [C, D, H, W]
            input_np = np.transpose(input_np, (1, 2, 3, 0))  # [D, H, W, C]
        
        # Normalize input
        if input_np.max() <= 1.0:
            input_np = (input_np * 255).astype(np.uint8)
        else:
            input_np = input_np.astype(np.uint8)
        
        # Generate post-processed masks
        activationLU = nn.ReLU()
        with torch.no_grad():
            post_processed_masks = model_operator.Cam_mask_post_process(
                activationLU(self.cam3D), 
                input_video,
                self.output
            )
        
        # Prepare video writer
        video_path = os.path.join(output_dir, f"{video_name}_class{class_idx}_mask_comparison.mp4")
        fps = 15
        frame_height = input_np.shape[1] * 2  # Two rows
        frame_width = input_np.shape[2] * 2   # Two columns
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        
        # Tool colors
        tool_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128)]
        color = tool_colors[class_idx] if class_idx < len(tool_colors) else (255, 255, 255)
        
        # Process each frame
        for frame_idx in tqdm(range(D), desc=f"Creating video for {video_name}"):
            # Get original frame
            orig_frame = input_np[frame_idx]
            if orig_frame.shape[-1] == 3:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            else:
                orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2RGB)
            
            # Get masks
            raw_mask = self.cam3D[0, class_idx, frame_idx].cpu().numpy()
            post_mask = post_processed_masks[0, class_idx, frame_idx].cpu().numpy()
            
            # Resize masks to original frame size
            raw_mask_resized = cv2.resize(raw_mask, (orig_frame_rgb.shape[1], orig_frame_rgb.shape[0]))
            post_mask_resized = cv2.resize(post_mask, (orig_frame_rgb.shape[1], orig_frame_rgb.shape[0]))
            
            # Create binary masks
            binary_raw = (raw_mask_resized > 0.5).astype(np.uint8)
            binary_post = (post_mask_resized > 0.5).astype(np.uint8)
            
            # Create overlays
            # 1. Raw CAM heatmap overlay
            raw_mask_normalized = (raw_mask_resized - raw_mask_resized.min()) / (raw_mask_resized.max() - raw_mask_resized.min() + 1e-8)
            heatmap = cv2.applyColorMap((raw_mask_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            raw_overlay = cv2.addWeighted(orig_frame_rgb, 0.5, heatmap, 0.5, 0)
            
            # 2. Post-processed colored overlay
            colored_mask = np.zeros_like(orig_frame_rgb)
            colored_mask[binary_post > 0] = color
            post_overlay = cv2.addWeighted(orig_frame_rgb, 0.5, colored_mask, 0.5, 0)
            
            # Create comparison grid
            # Top row: Original and Raw CAM
            top_row = np.hstack([orig_frame_rgb, raw_overlay])
            
            # Bottom row: Binary masks and Post-processed
            # Create binary mask visualization
            binary_raw_viz = np.zeros_like(orig_frame_rgb)
            binary_raw_viz[binary_raw > 0] = (255, 255, 255)
            
            bottom_row = np.hstack([binary_raw_viz, post_overlay])
            
            # Combine rows
            comparison_frame = np.vstack([top_row, bottom_row])
            
            # Add text labels
            cv2.putText(comparison_frame, f"Original", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison_frame, f"Raw CAM", (orig_frame_rgb.shape[1] + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison_frame, f"Binary Raw", (10, orig_frame_rgb.shape[0] + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison_frame, f"Post-processed", (orig_frame_rgb.shape[1] + 10, orig_frame_rgb.shape[0] + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(comparison_frame, f"Frame: {frame_idx}/{D}", 
                    (frame_width - 200, frame_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to BGR for video writing
            comparison_frame_bgr = cv2.cvtColor(comparison_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(comparison_frame_bgr)
        
        video_writer.release()
        print(f"Saved frame-by-frame comparison video to: {video_path}")