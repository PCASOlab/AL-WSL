# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
from time import time
import os
print("Current working directory:", os.getcwd())
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import  model_experiement, model_infer,model_infer_TC_mix_res_dino
from working_dir_root import Output_root,Linux_computer
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
import scheduler
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
from working_dir_root import Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
from working_dir_root import Enable_student,Batch_size,selected_data,sam_feature_OLG_dir2
from dataset import io

# GPU_mode= True
# Continue_flag = True
# Visdom_flag = False
# Display_flag = False
# loadmodel_index = '3.pth'
Output_root = Output_root+ "temporal_consistent/" +selected_data +"mix2"+'/'
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

# dataroot = "../dataset/CostMatrix/"
# torch.set_num_threads(8)
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
# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
     
############ for the linux to find the extenral drive
external_drives = find_external_drives()

if external_drives:
    print("External drives found:")
    for drive in external_drives:
        print(drive)
else:
    print("No external drives found.")
############ for the linux to find the extenral drive

Model_infer = model_infer_TC_mix_res_dino._Model_infer(GPU_mode,num_gpus,Enable_teacher=True,gpu_selection="1",pooling="rank",TPC=True)
device = Model_infer.device

# if GPU_mode == True:
#     if num_gpus > 1:
#         Model_infer.VideoNets = torch.nn.DataParallel(Model_infer.VideoNets)
#     Model_infer.VideoNets.to(device)

# Model.cuda()
dataLoader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='else',Device=device)
if Continue_flag == False:
    Model_infer.VideoNets.apply(weights_init)
else:
    
    # model_dict = Model_infer.VideoNets.state_dict()
  
    # # 1. filter out unnecessary keys
    # pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict_trim)
    # 3. load the new state dict
    Model_infer.VideoNets.load_state_dict( torch.load(Output_root + 'outNets_tc' + loadmodel_index ) )
    Model_infer.VideoNets_S.load_state_dict( torch.load(Output_root + 'outNets_st' + loadmodel_index ) )
    Model_infer.dino_encoder.load_state_dict( torch.load(Output_root + 'outNets_vit' + loadmodel_index ) )
    Model_infer.resnet.load_state_dict( torch.load(Output_root + 'outNets_res' + loadmodel_index ) )



read_id = 0
print(Model_infer.resnet)
print(Model_infer.VideoNets)

epoch = 0
# transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
# transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num = 0
#################
#############training
saver_id =0
displayer = Display(GPU_mode)
epoch =0
features =None
visdom_id=0
while (1):
    start_time = time()
    input_videos, labels= dataLoader.read_a_batch()
    input_videos_GPU = torch.from_numpy(np.float32(input_videos))
    labels_GPU = torch.from_numpy(np.float32(labels))
    input_videos_GPU = input_videos_GPU.to (device)
    labels_GPU = labels_GPU.to (device)
    input_flows = dataLoader.input_flows*1.0/ 255.0
    input_flows_GPU = torch.from_numpy(np.float32(input_flows))  
    input_flows_GPU = input_flows_GPU.to (device)
    if Load_feature ==True:
        features = dataLoader.features 
    Model_infer.forward(input_videos_GPU,input_flows_GPU,features,Enable_student)

    lr=scheduler.cyclic_learning_rate(current_epoch=epoch,max_lr=Max_lr,min_lr=learningR,cycle_length=4)
    print("learning rate is :" + str(lr))
    if Evaluation == False:
        Model_infer.optimization(labels_GPU,Enable_student) 

    # if  Save_feature_OLG== True:
    #     this_features= Model_infer.f[Batch_size-1].permute(1,0,2,3).half()
    #     sam_pkl_file_name = dataLoader.this_file_name
    #     sam_pkl_file_path = os.path.join(sam_feature_OLG_dir2, sam_pkl_file_name)

    #     with open(sam_pkl_file_path, 'wb') as file:
    #         pickle.dump(this_features, file)
    #         print("sam Pkl file created:" +sam_pkl_file_name)
    if Save_sam_mask == True:
         
        this_mask= Model_infer.sam_mask.half()
        mask_pkl_file_name = dataLoader.this_file_name
        mask_pkl_file_path = os.path.join(output_folder_sam_masks, mask_pkl_file_name)

        with open(mask_pkl_file_path, 'wb') as file:
            pickle.dump(this_mask, file)
            print("sam Pkl file created:" +mask_pkl_file_name)


    if Display_flag == True:
        displayer.train_display(Model_infer,dataLoader,read_id,Output_root)
         

    if dataLoader.all_read_flag ==1:
        Save_feature_OLG = False
        #remove this for none converting mode
        epoch +=1

        print("finished epoch" + str (epoch) )
        dataLoader.all_read_flag = 0
        read_id=0
        if Evaluation:
            break
        if Save_feature_OLG:
            break
        

        # break
    

    if Evaluation == False:
        
        if read_id % 10== 0 and Visdom_flag == True  :
            
            plotter.plot('l0', 'l0', 'l0', visdom_id, Model_infer.lossDisplay.cpu().detach().numpy())
            if Enable_student:
                plotter.plot('1ls', '1ls', 'l1s', visdom_id, Model_infer.lossDisplay_s.cpu().detach().numpy())
        if read_id % 1== 0   :
            print(" epoch" + str (epoch) )
            print(" loss" + str (Model_infer.lossDisplay.cpu().detach().numpy()) )
            if Enable_student:
                print(" loss_SS" + str (Model_infer.lossDisplay_s.cpu().detach().numpy()) )

    if (read_id % 1000) == 0  :
        torch.save(Model_infer.VideoNets.state_dict(), Output_root + "outNets_tc" + str(saver_id) + ".pth")
        torch.save(Model_infer.VideoNets_S.state_dict(), Output_root + "outNets_st" + str(saver_id) + ".pth")
        torch.save(Model_infer.dino_encoder.state_dict(), Output_root + "outNets_vit" + str(saver_id) + ".pth")
        torch.save(Model_infer.resnet.state_dict(), Output_root + "outNets_res" + str(saver_id) + ".pth")


        saver_id +=1
        if saver_id >1:
            saver_id =0

        end_time = time()

        print("time is :" + str(end_time - start_time))

    read_id+=1
    visdom_id+=1
    if epoch>30:
        Enable_student =True
    # print(labels)

    # pass





















