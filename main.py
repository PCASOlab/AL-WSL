# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from time import time
import os
from active_learning_pipeline import ActiveLearningPipeline
os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_cholec'  # Change this to your target mode
import yaml
import random

# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_ytobj'  # Change this to your target mode
print("Current working directory:", os.getcwd())
# print("Current working directory:", os.getcwd())
import shutil
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import  model_experiement, model_infer,model_infer_TC_dinov3
from working_dir_root import Output_root,Linux_computer
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
import scheduler
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
from working_dir_root import Max_epoch, Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
from working_dir_root import Use_Active_Learning,Enable_student,Batch_size,selected_data,sam_feature_OLG_dir2, Display_down_sample, Display_final_SAM
from dataset import io
from data_pre_curation.encord_proc_annots import process_encord_annotations 

# Import from annotation_utils
from model.annotation_utils import (
    binary_mask_to_polygons,
    convert_predictions_to_coco_format,
    create_coco_annotation_json,
    create_encord_annotation_json,
    _get_color_for_category
)
from tqdm import tqdm
Gpu_selection ='1'

Save_batch2mp4 = False
OUT_MP4_DIR = os.path.join(Output_root, "cholec_mp4")
os.makedirs(OUT_MP4_DIR, exist_ok=True)

dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
Output_root = Output_root+ "temporal_consistent/" + dataset_tag +  "DINOv3"+ "/"
io.self_check_path_create(Output_root)

import pickle

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
   
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
if GPU_mode ==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")

 # create the model

if Visdom_flag == True:
    from visual import VisdomLinePlotter

    plotter = VisdomLinePlotter(env_name='path finding training Plots')

def is_external_drive(drive_path):
    # Check if the drive is a removable drive (usually external)
    return os.path.ismount(drive_path) and shutil.disk_usage(drive_path).total > 0

def find_external_drives():
    # List all drives on the system
    drives = [d for d in os.listdir('/') if os.path.isdir(os.path.join('/', d))]

    # Filter out external drives and exclude certain paths
    external_drives = [drive for drive in drives if is_external_drive(os.path.join('/', drive))
                       and not drive.startswith(('media', 'run', 'dev'))]

    return external_drives
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
def add_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = 'module.' + key
        new_state_dict[new_key] = value
    return new_state_dict
     
def weights_init(m):
    classname = m.__class__.__name__
    
    # For standard convolutional layers
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    
    # For standard batch normalization layers
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    
    # For layer normalization layers
    elif classname.find('LayerNorm') != -1 and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    
    # For linear layers
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    
    # For your custom Conv3DBlock_layernorm - initialize its conv submodule
    elif classname == 'Conv3DBlock_layernorm':
        # Initialize the convolutional layer inside the custom block
        if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
            torch.nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
            if hasattr(m.conv, 'bias') and m.conv.bias is not None:
                torch.nn.init.constant_(m.conv.bias.data, 0)
############ for the linux to find the extenral drive
external_drives = find_external_drives()

if external_drives:
    print("External drives found:")
    for drive in external_drives:
        print(drive)
else:
    print("No external drives found.")
############ for the linux to find the extenral drive

Model_infer = model_infer_TC_dinov3._Model_infer(GPU_mode,num_gpus,Enable_teacher=True,gpu_selection=Gpu_selection,pooling="avg",TPC=True)
device = Model_infer.device
if Use_Active_Learning:
    CONFIG_PATH = "config.yaml"   # Change if needed
    with open(CONFIG_PATH, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Override selected variables from YAML
    AL_Folder = yaml_config['dataset']['AL_folder']
    selected_data = yaml_config['dataset']['selected_data']
    sam_feature_OLG_dir = yaml_config['paths']['sam_feature_olg_dir']
    cholec_gt_dir = yaml_config['paths']['cholec_gt_dir']
    original_videos_dir = yaml_config['paths']['original_videos_dir']
    al_dir_name = yaml_config['dataset']['checkpt_dir']
    vid_path=yaml_config['paths']['video_path']
    n_videos = yaml_config['dataset']['num_videos']
    minor_videos = yaml_config['dataset']['min_minor_videos']


    # Build active learning config dict
    ACTIVE_LEARNING_CONFIG = yaml_config['active_learning']

# # Process predictions and create COCO annotations

if Use_Active_Learning:
    active_learning_dir = os.path.join(Output_root, AL_Folder)
    os.makedirs(active_learning_dir, exist_ok=True)

GT_DIR= cholec_gt_dir
dataLoader = myDataloader(
    img_size=img_size,
    Display_loading_video=False,
    Read_from_pkl=True,
    Save_pkl=False,
    Load_flow=Load_flow,
    Device=device,
    gt_mask_dir=cholec_gt_dir
)
use_mask_loss = True


if Continue_flag == False:
    Model_infer.VideoNets.apply(weights_init)
else:
    Model_infer.VideoNets.load_state_dict( torch.load(active_learning_dir + '/outNets_tc' + loadmodel_index ) )
    Model_infer.VideoNets_S.load_state_dict( torch.load(active_learning_dir + '/outNets_st' + loadmodel_index ) )
    Model_infer.backbone.load_state_dict( torch.load(active_learning_dir + '/outNets_vit' + loadmodel_index ) )
    pass

read_id = 0
print(Model_infer)
# print(Model_infer.VideoNets)

iteration_num = 0
##############################training
saver_id =0
displayer = Display(GPU_mode)
epoch = 0
features =None
visdom_id=0
al_pipeline = None
active_learning_completed = False
# Control flag for mask loss
first_al_round_finished = False
if Use_Active_Learning:
    try:
        al_pipeline = ActiveLearningPipeline(
            ssh_private_key_path=ACTIVE_LEARNING_CONFIG['ssh_private_key_path'],
            project_title=ACTIVE_LEARNING_CONFIG['project_title'],
            output_root=Output_root,
            project_hash=ACTIVE_LEARNING_CONFIG.get('project_hash'),
            al_dir=os.path.join(active_learning_dir, al_dir_name)
        )
        print("Active Learning Pipeline initialized successfully!")
        
        # Check current status
        al_status = al_pipeline.get_active_learning_status()
        print(f"Active Learning Status: Round {al_status['current_round']}, Best Loss: {al_status['best_loss']:.4f}")
        
        if al_status['current_round'] > 0:
            first_al_round_finished = True
            use_mask_loss = True
            print("First AL round completed. Mask loss will be used in training.")
        
        if al_status['is_complete']:
            print("Active learning already completed. Switching to normal training.")
            Use_Active_Learning = False
        else:
            print(f"Starting Active Learning Round {al_status['current_round'] + 1}")
            
    except Exception as e:
        print(f"Failed to initialize Active Learning Pipeline: {e}")
        print("Falling back to normal training...")
        Use_Active_Learning = False
last_active_learning_epoch = 0
encord_sent_videos= set()

selected_videos_set = set() 
video_coco_files = {}

def select_videos(all_video_list, encord_sent_videos, 
                                       num_videos, min_minor_videos):
    """
    Select videos ensuring minimum representation of minor categories.
    """
    import pickle
    
    # Minor category indices: Bipolar(1), Scissors(3), Clipper(4), Irrigator(5), SpecimenBag(6)
    MINOR_CATEGORIES = {1, 3, 4, 5, 6}
    
    available_videos = [v for v in all_video_list if v not in encord_sent_videos]
    
    if len(available_videos) < num_videos:
        return set(available_videos)
    
    videos_with_minor = []
    videos_without_minor = []
    
    for video in available_videos:
        filename = video[0]  # e.g., 'clip_XXXXXX.pkl'
        dataset_tag = video[1]  # e.g., 'cholec'
        
        try:
            # Construct the full path based on your dataloader's logic
            # You'll need to adjust this based on where your pickle files are actually stored
            # This might be in a directory like 'cholec_pkl' or similar
            video_path = os.path.join(vid_path, filename)
            
            with open(video_path, 'rb') as f:
                data = pickle.load(f)
                
                # Extract labels based on your pickle structure
                video_labels = data['labels']
                
                # Check if any frame has any minor category present
                has_minor = False
                for frame_labels in video_labels:
                    if any(frame_labels[idx] == 1 for idx in MINOR_CATEGORIES):
                        has_minor = True
                        break
                
                if has_minor:
                    videos_with_minor.append(video)
                else:
                    videos_without_minor.append(video)
                        
        except Exception as e:
            videos_without_minor.append(video)
    
    selected_videos = []
    
    # Select minimum required videos with minor categories
    if len(videos_with_minor) >= min_minor_videos:
        selected_videos.extend(random.sample(videos_with_minor, min_minor_videos))
    else:
        selected_videos.extend(videos_with_minor)
    
    # Fill remaining slots
    remaining_slots = num_videos - len(selected_videos)
    if remaining_slots > 0:
        remaining_pool = [v for v in available_videos if v not in selected_videos]
        if remaining_pool:
            selected_videos.extend(random.sample(remaining_pool, min(remaining_slots, len(remaining_pool))))
    
    return set(selected_videos)

while (epoch<Max_epoch):
    start_time = time()
    # input_videos, labels= dataLoader.read_a_batch()
    batch_data = dataLoader.read_a_batch()
    if len(batch_data) == 3:  # If masks are available
        input_videos, labels, input_masks = batch_data
        
        if input_masks is not None and input_masks.sum() > 0:
            masks_available = True
        else:
            masks_available = False
            input_masks = None
    
    else:  # If no masks available
        input_videos, labels = batch_data
        input_masks = None

    input_videos_GPU = torch.from_numpy(np.float32(input_videos))
    labels_GPU = torch.from_numpy(np.float32(labels))
    input_videos_GPU = input_videos_GPU.to (device)
    labels_GPU = labels_GPU.to (device)
    input_flows = dataLoader.input_flows*1.0/ 255.0
    input_flows_GPU = torch.from_numpy(np.float32(input_flows))  
    input_flows_GPU = input_flows_GPU.to (device)

    if Save_batch2mp4 == True:
        video_label = dataLoader.this_video_label  
        # target_idx = [1, 3, 4, 5, 6]
        target_idx = [1]

        any_hit = any((i < len(video_label)) and (video_label[i] == 1) for i in target_idx)
        if any_hit:
                # Save the first video in this batch as mp4
            try:
                # Build a sensible filename; if your dataloader sets a per-batch name, use it
                # Fallback to epoch_readid
                base_name = dataLoader.this_file_name
                base_name = os.cpath.splitext(base_name)[0]  # strip .pkl if present
                out_mp4_path = os.path.join(OUT_MP4_DIR, f"{base_name}.mp4")

                # Use the numpy batch you already have (shape (B, C, T, H, W))
                io.write_mp4_from_tensor_last_in_batch(input_videos, out_mp4_path, fps=15)

                print(f"[MP4] Saved: {out_mp4_path}")
            except Exception as e:
                print(f"[MP4] Failed to save video for batch {read_id}: {e}")
    if Load_feature ==True:
        features = dataLoader.features 
    # Prepare masks tensor if available
    if masks_available and input_masks is not None:
        input_masks_GPU = torch.from_numpy(np.float32(input_masks)).to(device)
    else:
        input_masks_GPU = None
    video_name = os.path.splitext(dataLoader.this_file_name)[0] if hasattr(dataLoader, 'this_file_name') else f"clip_{read_id:06d}"

    output = Model_infer.forward(
        input_videos_GPU, 
        input_flows_GPU, 
        features, 
        Enable_student=Enable_student, 
        epoch=epoch
    )
    
    if (Use_Active_Learning and epoch >= 11 and  (epoch - last_active_learning_epoch) >= ACTIVE_LEARNING_CONFIG['active_learning_interval']):
        
        # Get video name for this batch
        video_name = os.path.splitext(dataLoader.this_file_name)[0] if hasattr(dataLoader, 'this_file_name') else f"clip_{read_id:06d}"

        print(f"Active Learning: Processing {video_name}")
        # Get all the first elements (filenames) without .pkl extension
        selected_filenames = {video[0].replace('.pkl', '') for video in selected_videos_set}

        if video_name in selected_filenames:
            # ── Check if ground truth already exists in GT_DIR ──────────────────
            gt_path = os.path.join(GT_DIR, f"{video_name}.pkl")
            if os.path.isfile(gt_path):
                # Annotation already present locally – no need to send to Encord
                print(f"[AL] Ground truth already exists for {video_name} in GT_DIR – skipping Encord upload.")
                # Mark as already handled so it is not re-selected in future rounds
                encord_sent_videos.add((video_name + '.pkl', 'cholec'))
                selected_filenames.discard(video_name)
                success = True  # treat as successful so epoch-end logic proceeds
            else:
                # ── Annotation not yet available locally – upload to Encord ────
                coco_file = Model_infer.export_coco_annotations(input_videos_GPU, video_name, active_learning_dir, round_number= al_pipeline.current_round)
                
                if coco_file and al_pipeline is not None:
                    print(f"[AL] Uploading COCO annotations to Encord: {coco_file}")
                    
                    success = al_pipeline.upload_coco_annotations(
                        coco_file_path=coco_file,
                        round_number=al_pipeline.current_round
                    )
                    if success:
                        # After first successful AL round, mark that masks should be used
                        active_learning_completed = True
                        first_al_round_finished = True
                
       

    if Evaluation == False:
        video_in_al_set = (video_name + '.pkl', 'cholec') in encord_sent_videos

        if first_al_round_finished and video_in_al_set and input_masks_GPU is not None and masks_available:
            Model_infer.optimization(labels_GPU, Enable_student, input_masks_GPU) 
        else:
            Model_infer.optimization(labels_GPU, Enable_student, None)

    if  Save_feature_OLG== True and dataLoader.features_exist ==False:
        this_features= Model_infer.f[Batch_size-1].permute(1,0,2,3).half()
        sam_pkl_file_name = dataLoader.this_file_name
        sam_pkl_file_path = os.path.join(sam_feature_OLG_dir, sam_pkl_file_name)

        with open(sam_pkl_file_path, 'wb') as file:
            pickle.dump(this_features, file)
            print("sam Pkl file created:" +sam_pkl_file_name)
    if Save_sam_mask == True:
         
        this_mask= Model_infer.sam_mask.half()
        mask_pkl_file_name = dataLoader.this_file_name
        mask_pkl_file_path = os.path.join(output_folder_sam_masks, mask_pkl_file_name)

        with open(mask_pkl_file_path, 'wb') as file:
            pickle.dump(this_mask, file)
            print("sam Pkl file created:" +mask_pkl_file_name)


    if Display_flag == True and read_id% Display_down_sample ==0:
        displayer.train_display(Model_infer,dataLoader,read_id,Output_root)
         

    if dataLoader.all_read_flag ==1:
        Save_feature_OLG = False        
    
        # Clear the dictionaries for the next epoch
        encord_sent_videos.update(selected_videos_set)
        for i in selected_videos_set:
            print(i[0])
        selected_videos_set.clear()
        video_coco_files.clear()
        if (Use_Active_Learning and epoch >= 11 and (epoch - last_active_learning_epoch) >= ACTIVE_LEARNING_CONFIG['active_learning_interval']):
            if success:
                selected_videos_file = os.path.join(al_pipeline.al_dir, f"round_{al_pipeline.current_round}_selected_videos.txt")
                with open(selected_videos_file, 'w') as f:
                    for vid in sorted(selected_filenames):
                        f.write(f"{vid}\n")
                print(f"Saved selected videos list to {selected_videos_file}")
                
                should_continue = al_pipeline.wait_for_human_correction(al_pipeline.current_round)
                
                if should_continue:
                    print(f"COCO annotations uploaded to Encord for round {al_pipeline.current_round}")
                    last_active_learning_epoch = epoch
                    al_pipeline.current_round += 1
                    
                    # Save checkpoint for this round
                    al_pipeline.save_round_checkpoint(Model_infer, al_pipeline.current_round)
                    annotations_path = al_pipeline.download_annotations(round_number=al_pipeline.current_round, selected_filenames=selected_filenames,download_dir=os.path.join(al_pipeline.al_dir, f"round_{al_pipeline.current_round}_selected") )
                    
                    # Process the downloaded annotations
                    process_encord_annotations(
                        annotation_dir=annotations_path,
                        original_videos_dir=original_videos_dir,
                        output_dir=active_learning_dir,
                        round_number=al_pipeline.current_round-1,
                        gt_dir= GT_DIR
                    )
                    dataLoader.refresh_ground_truth_list()
                    
                    # Update use_mask_loss flag
                    use_mask_loss = True
                    print("New annotated data processed. Masks will be available in next training round!")
    
        # Normal epoch completion code
        print("finished epoch" + str(epoch))

        epoch += 1
        Model_infer.scheduler.step()
        Model_infer.schedulers.step()

        dataLoader.all_read_flag = 0
        read_id = 0
        
        if (Use_Active_Learning and epoch >= 11 and (epoch - last_active_learning_epoch) >= ACTIVE_LEARNING_CONFIG['active_learning_interval']):
            if n_videos==2:
                available_videos = [v for v in dataLoader.all_video_dir_list if v not in encord_sent_videos]
                selected_videos_set = set(random.sample(available_videos, 2))
            else:
                selected_videos_set = select_videos(
                    all_video_list=dataLoader.all_video_dir_list,
                    encord_sent_videos=encord_sent_videos,
                    num_videos=n_videos,
                    min_minor_videos=minor_videos
                    )
            
        if Evaluation:
            break
        if Save_feature_OLG:
            break
        
       
    if Evaluation == False:
        
        if read_id % 100== 0 and Visdom_flag == True  :
            
            plotter.plot('l0', 'l0', 'l0', visdom_id, Model_infer.lossDisplay.cpu().detach().numpy())
            if Enable_student:
                plotter.plot('1ls', '1ls', 'l1s', visdom_id, Model_infer.lossDisplay_s.cpu().detach().numpy())
        if read_id % 1== 0   :
            print(" epoch" + str (epoch) )
            print(" loss" + str (Model_infer.lossDisplay.cpu().detach().numpy()) )
            if Enable_student:
                print(" loss_SS" + str (Model_infer.lossDisplay_s.cpu().detach().numpy()) )

    if (read_id % 1000) == 0  :
        torch.save(Model_infer.state_dict(), active_learning_dir + "/outNets" + str(saver_id) + ".pth")
        torch.save(Model_infer.VideoNets.state_dict(), active_learning_dir + "/outNets_tc" + str(saver_id) + ".pth")
        torch.save(Model_infer.VideoNets_S.state_dict(), active_learning_dir + "/outNets_st" + str(saver_id) + ".pth")
        torch.save(Model_infer.backbone.state_dict(), active_learning_dir + "/outNets_vit" + str(saver_id) + ".pth")


        saver_id +=1
        if saver_id >1:
            saver_id =0

        end_time = time()

        print("time is :" + str(end_time - start_time))

    read_id+=1
    visdom_id+=1
    if epoch>1:
        Load_feature=True
    if epoch>9:
        Enable_student =True