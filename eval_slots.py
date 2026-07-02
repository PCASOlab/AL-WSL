from eval import *
from dataset import io
from working_dir_root import Visdom_flag
from sklearn.metrics import adjusted_rand_score
import cv2
from visdom import Visdom
if Visdom_flag:
  viz = Visdom(port=8097)
from model.model_operator import post_process_softmask
from working_dir_root import Display_visdom_figure
from  data_pre_curation. data_ytobj_box_train import apply_mask
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff
def binary_to_multi_channel(binary_mask):
    """
    Convert a binary mask into a multi-channel mask, where each channel represents a distinct object.

    Args:
        binary_mask: A 2D numpy array of shape (H, W) where pixels are either 0 or 1.

    Returns:
        multi_channel_mask: A 3D numpy array of shape (N, H, W) where N is the number of distinct objects.
    """
    # Define the structure for 4-connectivity
    structure = np.array([[0, 1, 0], 
                          [1, 1, 1], 
                          [0, 1, 0]], dtype=np.int8)

    # Apply connected component labeling with 4-connectivity
    labeled_mask, num_features = label(binary_mask, structure=structure)  # Label connected components

    # Create multi-channel mask
    multi_channel_mask = np.zeros((num_features, *binary_mask.shape), dtype=np.float32)

    for i in range(1, num_features + 1):  # Start from 1 to ignore background (0)
        multi_channel_mask[i - 1] = (labeled_mask == i).astype(np.float32)  # Create a binary mask for the current object

    return multi_channel_mask

def convert_label_frame_to_instance_masks(label_frame, min_gap_size=5):
    """
    Convert a binary label frame to instance masks and fill small gaps.

    Args:
        label_frame: A tensor of shape (N, H, W) containing binary masks.
        min_gap_size: The size of small gaps to ignore (in pixels).

    Returns:
        instance_masks: A tensor of shape (num_instances, H, W) representing instance masks.
    """
    N, H, W = label_frame.size()  # Get the dimensions of the label frame
    instance_masks = []

    # Structuring element for morphological closing (filling gaps)
    kernel = np.ones((min_gap_size, min_gap_size), np.uint8)

    # Iterate through each channel (N dimension)
    for channel_idx in range(N):
        binary_mask = label_frame[channel_idx].cpu().numpy()  # Get the binary mask for the channel
        
        # Perform morphological closing to fill small gaps
        processed_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Create multi-channel mask for the processed binary mask
        multi_channel_mask = binary_to_multi_channel(processed_mask)

        # Append the multi-channel masks for this channel to the instance masks
        instance_masks.extend(multi_channel_mask)  # Extend to include all new instance masks

    return torch.tensor(instance_masks)  # Convert the list back to a tensor
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
# import torch.nn.functional as F
# from scipy.spatial.distance import directed_hausdorff
def remove_empty_channels(mask_stack, threshold=10):
    """
    Remove channels that have fewer than the threshold of non-zero pixels from the ground truth mask stack.
    Args:
        mask_stack: Ground truth mask stack, shape [n, L, H, W].
        threshold: Minimum number of non-zero pixels for a channel to be considered non-empty (default is 10).
    Returns:
        filtered_mask_stack: Mask stack with channels having at least 'threshold' non-zero pixels, shape [n_filtered, L, H, W].
    """
    # Count the number of non-zero pixels in each channel
    non_zero_counts = (mask_stack > 0).sum(dim=[1, 2, 3])  # Count non-zero pixels in each channel

    # Get indices of channels with at least 'threshold' non-zero pixels
    non_empty_channels = torch.nonzero(non_zero_counts >= threshold, as_tuple=False).squeeze(1)

    # Select only non-empty channels
    filtered_mask_stack = mask_stack[non_empty_channels, :, :, :]

    return filtered_mask_stack
def remove_empty_channels_frame(mask_stack, threshold=20):
    """
    Remove channels that have fewer than the threshold of non-zero pixels from the ground truth mask stack.
    Args:
        mask_stack: Ground truth mask stack, shape [n, L, H, W].
        threshold: Minimum number of non-zero pixels for a channel to be considered non-empty (default is 10).
    Returns:
        filtered_mask_stack: Mask stack with channels having at least 'threshold' non-zero pixels, shape [n_filtered, L, H, W].
    """
    # Count the number of non-zero pixels in each channel
    if not mask_stack.any():
        print ("no instance ")
        return mask_stack
    non_zero_counts = (mask_stack > 0).sum(dim=[1, 2])  # Count non-zero pixels in each channel

    # Get indices of channels with at least 'threshold' non-zero pixels
    non_empty_channels = torch.nonzero(non_zero_counts >= threshold, as_tuple=False).squeeze(1)

    # Select only non-empty channels
    filtered_mask_stack = mask_stack[non_empty_channels, :, :]

    return filtered_mask_stack
def hungarian_iou(label_mask, predic_mask_3D):
    """
    Calculate the minimal IoU using the Hungarian algorithm between ground truth and predicted masks.
    
    Args:
        label_mask: Ground truth masks (N, L, H, W)
        predic_mask_3D: Predicted masks (M, L, H, W)

    Returns:
        avg_iou: Average minimal IoU for the best matching masks
    """
    N, L, H, W = label_mask.size()  # N = number of ground truth channels
    M, _, _, _ = predic_mask_3D.size()  # M = number of predicted channels
    if torch.isnan(label_mask).any():
        return np.nan,np.nan
    # Initialize IoU matrix
    iou_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            # Calculate IoU between the ith ground truth mask and jth predicted mask
            iou_matrix[i, j] = cal_J(label_mask[i], predic_mask_3D[j]).item()

    # Apply Hungarian algorithm to maximize the IoU matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # We negate the matrix because we want to maximize

    # Compute the average minimal IoU
    avg_iou = iou_matrix[row_ind, col_ind].mean()

    return avg_iou, iou_matrix[row_ind, col_ind]


def hungarian_dice(label_mask, predic_mask_3D):
    """
    Calculate the minimal Dice coefficient using the Hungarian algorithm between ground truth and predicted masks.
    
    Args:
        label_mask: Ground truth masks (N, L, H, W)
        predic_mask_3D: Predicted masks (M, L, H, W)

    Returns:
        avg_dice: Average minimal Dice coefficient for the best matching masks
    """
    N, L, H, W = label_mask.size()
    M, _, _, _ = predic_mask_3D.size()
    if torch.isnan(label_mask).any():
        return np.nan,np.nan
    # Initialize Dice matrix
    dice_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            dice_matrix[i, j] = cal_dice(label_mask[i], predic_mask_3D[j]).item()

    # Apply Hungarian algorithm to maximize the Dice coefficient matching
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    # Compute the average minimal Dice coefficient
    avg_dice = dice_matrix[row_ind, col_ind].mean()

    return avg_dice, dice_matrix[row_ind, col_ind]
def hungarian_iou_per_frame(label_frame, predic_frame):
    """
    Calculate the max matching IoU using the Hungarian algorithm for a single frame.
    
    Args:
        label_frame: Ground truth mask for a single frame, shape (N, H, W).
        predic_frame: Predicted mask for a single frame, shape (M, H, W).

    Returns:
        avg_iou: Average max IoU for the best matching masks for this frame.
    """
    N, H, W = label_frame.size()
    M, _, _ = predic_frame.size()
    if torch.isnan(label_frame).any():
            return np.nan
    # Initialize IoU matrix for this frame
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            # Calculate IoU between the ith ground truth mask and jth predicted mask for the frame
            iou_matrix[i, j] = cal_J(label_frame[i], predic_frame[j]).item()

    # Apply Hungarian algorithm to maximize the IoU matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Compute the average IoU for this frame
    avg_iou = iou_matrix[row_ind, col_ind].mean()

    return avg_iou


def hungarian_dice_per_frame(label_frame, predic_frame):
    """
    Calculate the max matching Dice coefficient using the Hungarian algorithm for a single frame.
    
    Args:
        label_frame: Ground truth mask for a single frame, shape (N, H, W).
        predic_frame: Predicted mask for a single frame, shape (M, H, W).

    Returns:
        avg_dice: Average max Dice coefficient for the best matching masks for this frame.
    """
    N, H, W = label_frame.size()
    M, _, _ = predic_frame.size()
    if torch.isnan(label_frame).any():
            return np.nan
    # Initialize Dice matrix for this frame
    dice_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            dice_matrix[i, j] = cal_dice(label_frame[i], predic_frame[j]).item()

    # Apply Hungarian algorithm to maximize the Dice matching
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    # Compute the average Dice coefficient for this frame
    avg_dice = dice_matrix[row_ind, col_ind].mean()

    return avg_dice

def hungarian_dice_per_frame_instance(label_frame, predic_frame):
    """
    Calculate the max matching Dice coefficient, Hausdorff distance, and return the matched masks using the Hungarian algorithm for a single frame.
    
    Args:
        label_frame: Ground truth mask for a single frame, shape (N, H, W).
        predic_frame: Predicted mask for a single frame, shape (M, H, W).

    Returns:
        avg_dice: Average max Dice coefficient for the best matching masks for this frame.
        avg_hd: Average Hausdorff distance for the best matching masks for this frame.
        matched_gt_masks: Matched ground truth masks, shape (N, H, W) after matching.
        matched_pred_masks: Matched predicted masks, shape (M, H, W) after matching.
    """
    if torch.isnan(label_frame).any():
        return np.nan,np.nan, None, None  
    # Convert ground truth to instance masks
    instance_masks = convert_label_frame_to_instance_masks(label_frame)
    instance_masks = remove_empty_channels_frame(instance_masks)
    
    if not instance_masks.any():
        print("NO GT")
        return np.nan, np.nan, None, None  # No ground truth, return NaN for both Dice and Hausdorff distance

    N, H, W = instance_masks.size()
    instance_masks = instance_masks.to(predic_frame.device)
    M, _, _ = predic_frame.size()

    # Initialize Dice and Hausdorff distance matrices for this frame
    dice_matrix = np.zeros((N, M))
    hausdorff_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            # Calculate Dice coefficient
            dice_matrix[i, j] = cal_dice(instance_masks[i], predic_frame[j]).item()
            
            # Calculate Hausdorff distance
            gt_indices = np.argwhere(instance_masks[i].cpu().numpy() != 0)  # Get coordinates of non-background pixels
            pred_indices = np.argwhere(predic_frame[j].cpu().numpy() != 0)  # Same for predicted mask
            
            if len(gt_indices) > 0 and len(pred_indices) > 0:
                hausdorff_matrix[i, j] = max(directed_hausdorff(gt_indices, pred_indices)[0],
                                             directed_hausdorff(pred_indices, gt_indices)[0])
            else:
                hausdorff_matrix[i, j] = float('inf')  # Set Hausdorff to infinity if one of the masks is empty

    # Apply Hungarian algorithm to maximize the Dice matching
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    # Compute the average Dice coefficient for this frame
    avg_dice = dice_matrix[row_ind, col_ind].mean()

    # Compute the average Hausdorff distance for the matched masks
    avg_hd = hausdorff_matrix[row_ind, col_ind].mean()

    # Create matched ground truth and predicted masks
    matched_gt_masks = torch.stack([instance_masks[i] for i in row_ind], dim=0)  # Select matched GT masks
    matched_pred_masks = torch.stack([predic_frame[j] for j in col_ind], dim=0)  # Select matched predicted masks

    return avg_dice, avg_hd, matched_gt_masks, matched_pred_masks
def calculate_iou(pred, gt):
    """Calculate IoU between two binary masks."""
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 0
    else:
        return intersection / union

def hungarian_matching(pred_masks, gt_masks):
    """
    Perform Hungarian matching based on IoU between predicted and ground truth masks.
    
    Args:
        pred_masks: Predicted binary masks, shape (P, H * W).
        gt_masks: Ground truth binary masks, shape (G, H * W).
        
    Returns:
        matched_pred: Matched predicted masks, reordered based on ground truth.
        matched_gt: Ground truth masks, possibly duplicated.
    """
    P, H_W = pred_masks.shape
    G, _ = gt_masks.shape

    # Compute IoU matrix between all ground truth and predicted masks
    iou_matrix = np.zeros((G, P))
    for g in range(G):
        for p in range(P):
            iou_matrix[g, p] = calculate_iou(gt_masks[g], pred_masks[p])
    
    # Perform Hungarian matching to maximize IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximizing IoU
    
    # Reorder predicted masks according to Hungarian matching
    matched_pred = pred_masks[col_ind]
    matched_gt = gt_masks[row_ind]
    
    return matched_pred, matched_gt

def get_ari_multichannel(prediction_masks, gt_masks, bg_class=0):
    """
    Calculate ARI for multi-channel binary masks after Hungarian matching.
    
    Args:
        prediction_masks: Predicted masks, shape (P, H, W).
        gt_masks: Ground truth masks, shape (G, H, W).
        bg_class: Background class, usually 0.
        
    Returns:
        ari: Adjusted Rand Index for the matched masks.
    """
    # Flatten masks along the spatial dimensions
    prediction_masks_flat = prediction_masks.flatten(start_dim=1).cpu().numpy().astype(int)
    gt_masks_flat = gt_masks.flatten(start_dim=1).cpu().numpy().astype(int)
    
    # Perform Hungarian matching based on IoU
    matched_pred, matched_gt = hungarian_matching(prediction_masks_flat, gt_masks_flat)
    
    # Compute ARI for each frame
    rand_scores = []
    for pred, gt in zip(matched_pred, matched_gt):
        if np.all(gt == bg_class):  # Skip if the ground truth is all background
            continue
        rand_scores.append(adjusted_rand_score(gt, pred))
    
    # Average ARI score across frames
    if len(rand_scores) == 0:
        ari = np.nan
    else:
        ari = sum(rand_scores) / len(rand_scores)
    
    return ari
def overlap_multichannel_gt_pred_separate(frame, mask_gt, mask_pred, alpha=0.5):
    """
    Overlays multi-channel ground truth and predicted masks on top of the video frame separately, 
    assigning a unique color to each channel, ensuring the same color for corresponding channels.
    
    Args:
        frame: The original video frame, shape (3, H, W) or (H, W, 3), assumed to be in RGB.
        mask_gt: The ground truth multi-channel mask, shape (C, H, W), expected to be a tensor.
        mask_pred: The predicted multi-channel mask, shape (C, H, W), expected to be a tensor.
        alpha: Blending factor for the masks. Default is 0.5.
        
    Returns:
        blended_gt_frame: The frame with the ground truth masks overlaid.
        blended_pred_frame: The frame with the predicted masks overlaid.
    """
    # Ensure the frame is in the shape (H, W, 3) and in RGB format
    if frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)  # Convert from (3, H, W) -> (H, W, 3)

    # Move masks from GPU to CPU and convert them to NumPy arrays
    mask_gt = mask_gt.cpu().numpy()
    mask_pred = mask_pred.cpu().numpy()

    # Initialize the blended frames as copies of the original frame
    blended_gt_frame = frame.copy()
    blended_pred_frame = frame.copy()

    # Get the number of channels (should be the same for both ground truth and prediction)
    num_channels_gt = mask_gt.shape[0]
    num_channels_pred = mask_pred.shape[0]

    if num_channels_gt != num_channels_pred:
        raise ValueError("Number of channels in ground truth and predicted masks must be the same.")

    # Generate a consistent color for each channel
    colors = np.random.randint(0, 255, size=(num_channels_gt, 3), dtype=np.uint8)  # Random colors for each channel

    # Initialize empty color masks
    combined_color_mask_gt = np.zeros_like(frame, dtype=np.uint8)
    combined_color_mask_pred = np.zeros_like(frame, dtype=np.uint8)

    # Overlay each ground truth and predicted channel with the same color
    for channel_idx in range(num_channels_gt):
        # Apply the channel's color where the ground truth and predicted masks are non-zero
        combined_color_mask_gt[mask_gt[channel_idx] == 1] = colors[channel_idx]  # GT mask color
        combined_color_mask_pred[mask_pred[channel_idx] == 1] = colors[channel_idx]  # Predicted mask color
    # combined_color_mask_gt = combined_color_mask_gt.astype(np.uint8)
    # combined_color_mask_pred = combined_color_mask_pred.astype(np.uint8)
    # Blend the combined ground truth mask with the original frame
    blended_gt_frame = cv2.addWeighted(blended_gt_frame.astype(np.uint8), 1 - alpha, combined_color_mask_gt, alpha, 0)
    # Blend the combined predicted mask with the original frame
    blended_pred_frame = cv2.addWeighted(blended_pred_frame.astype(np.uint8), 1 - alpha, combined_color_mask_pred, alpha, 0)

    return blended_gt_frame, blended_pred_frame
def cal_all_metrics_slots(read_id, Output_root, label_mask, predic_mask_3D,input_video):
    if not label_mask.any():
        print ("NO GT")
        return
    device = label_mask.device
    predic_mask_3D = predic_mask_3D.to(device)
    ch, D, H, W = label_mask.size()
    predic_mask_3D = F.interpolate(predic_mask_3D, size=(H, W), mode='bilinear', align_corners=False)
    predic_mask_3D = (predic_mask_3D > 0) * predic_mask_3D
    predic_mask_3D = predic_mask_3D - torch.min(predic_mask_3D)
    predic_mask_3D = predic_mask_3D / (torch.max(predic_mask_3D) + 0.0000001) * 1
    predic_mask_3D = predic_mask_3D > 0.5
    predic_mask_3D = torch.clamp(predic_mask_3D, 0, 1)
    filtered_label_mask = remove_empty_channels(label_mask) 

    # Frame-level IoU and Dice calculation
    frame_level_ious = []
    frame_level_dices = []
    frame_level_dices_instance=[]
    frame_level_HD_instance=[]

    frame_level_ari=[]
    video_stack_gt = []
    video_stack_pred = []


    for frame_idx in range(D):
        frame = input_video[:,frame_idx,:,:]  # (3,256,256)

        label_frame = filtered_label_mask[:, frame_idx, :, :]
        predic_frame = predic_mask_3D[:, frame_idx, :, :]

        avg_iou_frame = hungarian_iou_per_frame(label_frame, predic_frame)
        avg_dice_frame = hungarian_dice_per_frame(label_frame, predic_frame)

        avg_dice_frame_instance,avg_HD_frame_instance,match_GT,matched_pred = hungarian_dice_per_frame_instance(label_frame, predic_frame)
        avg_dice_frame_ari = get_ari_multichannel(label_frame, predic_frame)
        

        frame_level_ious.append(avg_iou_frame)
        frame_level_dices.append(avg_dice_frame)
        frame_level_dices_instance.append(avg_dice_frame_instance)
        frame_level_HD_instance.append(avg_HD_frame_instance)
        frame_level_ari.append(avg_dice_frame_ari)

        if match_GT is not None and matched_pred is not None:
            # Overlap ground truth and predicted masks, ensuring consistent colors between them
            blended_gt_frame, blended_pred_frame = overlap_multichannel_gt_pred_separate(
                frame, match_GT, matched_pred, alpha=0.5)
            
            video_stack_gt.append(blended_gt_frame)
            video_stack_pred.append(blended_pred_frame)
    if len(video_stack_gt) >0 and len (video_stack_pred) >0:
        video_stack_gt = np.hstack(video_stack_gt)  # Stack all GT overlays
        video_stack_pred = np.hstack(video_stack_pred)  # Stack all predicted overlays

        combine_stack = np.vstack([video_stack_gt, video_stack_pred])

    # Transpose to put the color channels in the correct position for displaying
        # combine_stack = combine_stack.transpose(1, 2, 0)

        viz.image(np.transpose(combine_stack.astype((np.uint8)), (2, 0, 1)), opts=dict(title=f'{read_id} - stack_color_mask'))

    # cv2.imshow("matched masks overlay", combine_stack.transpose)

    avg_frame_level_iou = np.nanmean(frame_level_ious)
    avg_frame_level_dice = np.nanmean(frame_level_dices)
    avg_frame_level_dice_instance = np.nanmean(frame_level_dices_instance)
    avg_frame_level_HD_instance = np.nanmean(frame_level_HD_instance)

    avg_frame_level_ari = np.nanmean(frame_level_ari)



    # Calculate minimal IoU and Dice using Hungarian Matching over entire video
    avg_iou, matched_ious = hungarian_iou(filtered_label_mask, predic_mask_3D)
    avg_dice, matched_dices = hungarian_dice(filtered_label_mask, predic_mask_3D)

    print(f"Average max IoU (Hungarian): {avg_iou:.4f}")
    print(f"Average max Dice (Hungarian): {avg_dice:.4f}")
    print(f"Frame-level average max IoU (Hungarian): {avg_frame_level_iou:.4f}")
    print(f"Frame-level average max Dice (Hungarian): {avg_frame_level_dice:.4f}")
    print(f"Frame-level average max Dice instance (Hungarian): {avg_frame_level_dice_instance:.4f}")
    print(f"Frame-level average HD instance (Hungarian): {avg_frame_level_HD_instance:.4f}")
    print(f"Frame-level average ARI: {avg_frame_level_ari:.4f}")


    global metrics_video_data
    metrics_video_data.append({
        'read_id': read_id,
        'IoU': avg_iou,
        'Dice_Coefficient': avg_dice,
        'Frame_level_IoU': avg_frame_level_iou,
        'Frame_level_Dice': avg_frame_level_dice,
        'Frame_level_Dice_instance': avg_frame_level_dice_instance,
        'Frame-level average HD instance':avg_frame_level_HD_instance,
        'Frame-level average ARI':avg_frame_level_ari,
        # Add other metrics here if needed
    })

    metrics_video = pd.DataFrame(metrics_video_data)
    metrics_video.to_excel(Output_root + 'metrics_video.xlsx', index=False, float_format='%.4f')