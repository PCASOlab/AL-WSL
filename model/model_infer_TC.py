import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from model.model_3dcnn_linear_TC import _VideoCNN
from model.model_3dcnn_linear_ST_3 import _VideoCNN_S
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_student,Display_final_SAM
# from working_dir_root import Enable_teacher
from dataset.dataset import class_weights
import numpy as np
from image_operator import basic_operator   
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import Enable_student,Random_mask_temporal_feature,Random_mask_patch_feature,Display_fuse_TC_ST
from working_dir_root import Use_max_error_rejection
from model import model_operator
# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset.dataset import label_mask,Mask_out_partial_label
from torch.optim import lr_scheduler
import random

if Evaluation == True:
    learningR=0
    Weight_decay=0
# learningR = 0.0001
def select_gpus(gpu_selection):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of GPUs available:", num_gpus)
        if gpu_selection == "all":
            device = torch.device("cuda" if num_gpus > 0 else "cpu")
            # if num_gpus > 1:
            #     device = torch.device("cuda:0," + ",".join([str(i) for i in range(1, num_gpus)]))
        elif gpu_selection.isdigit():
            gpu_index = int(gpu_selection)
            device = torch.device("cuda:" + gpu_selection if gpu_index < num_gpus else "cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device
class _Model_infer(nn.Module):
    def __init__(self, GPU_mode =True,num_gpus=1,Enable_teacher=True,Using_spatial_conv=True,Student_be_teacher=False,gpu_selection = "all",pooling="rank",TPC=True):
        super(_Model_infer, self).__init__()
        if GPU_mode ==True:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = select_gpus(gpu_selection)
        else:
            device = torch.device("cpu")
        self.device = device
        sam_checkpoint = SAM_pretrain_root+"sam_vit_h_4b8939.pth"
        sam_checkpoint = SAM_pretrain_root+"sam_vit_l_0b3195.pth"
        sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
        self.inter_bz =1
        model_type = "vit_h"
        model_type = "vit_l"
        model_type = "vit_b"
        self.TPC = TPC
        
        # model_type = "vit_t"
        # sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # self.predictor = SamPredictor(self.sam) 
        self.Vit_encoder = sam.image_encoder
        sam_predictor = SamPredictor(sam)
        self.sam_model = sam_predictor.model
        if Student_be_teacher==False:
            self.VideoNets = _VideoCNN_S(inputC=256,pooling=pooling)
        else:
            self.VideoNets =  _VideoCNN_S(Using_spatial_conv=Using_spatial_conv,pooling=pooling)
        self.VideoNets_S = _VideoCNN_S( pooling=pooling)
        self.input_size = 1024
        resnet18 = models.resnet18(pretrained=True)
        self.gradcam = None
        self.Enable_teacher = Enable_teacher
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
                # self.VideoNets.classifier = torch.nn.DataParallel(self.VideoNets.classifier)
                # self.VideoNets.blocks = torch.nn.DataParallel(self.VideoNets.blocks)
                self.VideoNets = torch.nn.DataParallel(self.VideoNets)
                self.VideoNets_S = torch.nn.DataParallel(self.VideoNets_S)


                self.resnet  = torch.nn.DataParallel(self.resnet )
                self.Vit_encoder   = torch.nn.DataParallel(self.Vit_encoder  )
                self.sam_model  = torch.nn.DataParallel(self.sam_model )
        self.VideoNets.to(device)
        self.VideoNets_S.to(device)


        # self.VideoNets.classifier.to(device)
        # self.VideoNets.blocks.to(device)


        self.resnet .to(device)
        self.Vit_encoder.to(device)
        self.sam_model .to (device)
        if Evaluation:
            pass
            #  self.VideoNets.eval()
            #  self.VideoNets_S.eval()
        else:
            self.VideoNets.train(True)
            self.VideoNets_S.train(True)

        
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss(weight=weight_tensor).to(device)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE_S = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE_S = torch.nn.BCEWithLogitsLoss().to(device)
        self.customeBCE =  torch.nn.MSELoss().to(device)
        self.customeBCE_S =  torch.nn.MSELoss().to(device)


        self.customeBCE_mask = torch.nn.MSELoss( ).to(device)

        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)
        
        self.optimizer = torch.optim.Adam([
        {'params': self.VideoNets.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ], weight_decay=Weight_decay)
        # self.optimizer = torch.optim.Adam([
        # {'params': self.VideoNets.parameters(),'lr': learningR}
        # # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        # ])
        # self.optimizer = torch.optim.Adam ([
        # {'params': self.VideoNets.parameters(),'lr': learningR}
        # # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        # ] )
        self.optimizer_s = torch.optim.Adam([
        {'params': self.VideoNets_S.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ] )
        # if GPU_mode ==True:
        #     if num_gpus > 1:
        #         self.optimizer = torch.nn.DataParallel(optself.optimizerimizer)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 10, eta_min=0, last_epoch=-1)  # Optional parameters explained below
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
    def forward(self,input,input_flows, features,Enable_student):
        # self.res_f = self.resnet(input)
        bz, ch, D, H, W = input.size()
        activationLU = nn.ReLU()


        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        # self.
        if Load_feature == False  or features is None:
            flattened_tensor = self.input_resample.permute(0,2,1,3,4)
            flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
            flattened_tensor = (flattened_tensor-124.0)/60.0

            num_chunks = (bz*D + self.inter_bz - 1) // self.inter_bz
        
            # List to store predicted tensors
            predicted_tensors = []
            
            # Chunk input tensor and predict
            with torch.no_grad():
                for i in range(num_chunks):
                    start_idx = i * self.inter_bz
                    end_idx = min((i + 1) * self.inter_bz, bz*D)
                    input_chunk = flattened_tensor[start_idx:end_idx]
                    output_chunk = self.Vit_encoder(input_chunk)
                    predicted_tensors.append(output_chunk)
                    # torch.cuda.empty_cache()
               
        
            # Concatenate predicted tensors along batch dimension
            concatenated_tensor = torch.cat(predicted_tensors, dim=0)

            flattened_tensor = concatenated_tensor
            new_bz, new_ch, new_H, new_W = flattened_tensor.size()
            self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        else:
            with torch.no_grad():
                self.f = features
        flag =random. choice([True, True])
        self.fm =self.f
        if  Random_mask_temporal_feature == True:
            self.fm =   model_operator.random_mask_out_dimension(self.fm, 0.5, dim=2)
        if  Random_mask_patch_feature == True:
            self.fm =   model_operator.hide_patch(self.fm )

        self.output, self.slice_valid, self. cam3D= self.VideoNets(self.fm,flag)
        with torch.no_grad():
            self.slice_hard_label,self.binary_masks= model_operator.CAM_to_slice_hardlabel(activationLU(self.cam3D),self.output)
            self.cam3D_target = self.cam3D.detach().clone()
        if Enable_student:
            self.output_s,self.slice_valid_s,self.cam3D_s = self.VideoNets_S(self.f,flag)
        # stack = self.cam3D_s -torch.min(self.cam3D_s)
        # stack = stack /(torch.max(stack)+0.0000001)
        with torch.no_grad():
            output = self.output.detach().clone()
        if Display_student:
            with torch.no_grad():
                if Display_fuse_TC_ST == True:
                    self.cam3D = (self.cam3D_s.detach().clone() + self.cam3D)/2
                else:
                    self.cam3D = self.cam3D_s.detach().clone()
                
                #output = self.output_s.detach().clone()
        self.raw_cam = self.cam3D.detach().clone()
        if Display_final_SAM:
            with torch.no_grad():
                post_processed_masks=model_operator.Cam_mask_post_process(activationLU(self.cam3D), input,output)
                # self.sam_mask_prompt_decode(activationLU(self.cam3D),self.f,input)

                # post_processed_masks =model_operator.sam_mask_prompt_decode(self.sam_model,post_processed_masks,self.f)
                self.cam3D = post_processed_masks.to(self.device) 
        # self. sam_mask =   F.interpolate(self. sam_mask,  size=(D, 32, 32), mode='trilinear', align_corners=False)
        # self.cam3D = self. sam_mask.to(self.device)  
        # self.cam3D = self.cam3D+stack
                # self.cam3D = post_processed_masks
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
            # loss = F.multilabel_soft_margin_loss(out_logits * label_mask_torch, label * label_mask_torch)
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
    def optimization(self, label,Enable_student):
        # for param_group in  self.optimizer.param_groups:
        #     param_group['lr'] = lr 
        self.optimizer.zero_grad()
        self.optimizer_s.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        # self.set_requires_grad(self.VideoNets, True)

        if Use_max_error_rejection == False:
            self.loss = self.loss_of_one_scale(self.output,label,BCEtype=1)
        else:
            self.loss = self.loss_of_one_scale_with_error_filter(self.output,label,BCEtype=1)


        # self.lossEa.backward(retain_graph=True)
        self.loss.backward( )

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()


        # out_logits_s = self.output_s.view(label.size(0), -1)
        if Enable_student:
            self.set_requires_grad(self.VideoNets_S,True)

            if Use_max_error_rejection == False:
                self.loss_s_v = self.loss_of_one_scale(self.output_s,label,BCEtype=2)  
            else:
                self.loss_s_v = self.loss_of_one_scale_with_error_filter(self.output_s,label,BCEtype=2)  


            bz, ch, D, H, W = self.cam3D_s.size()
            
            label_valid_repeat = label.reshape(bz,ch,1,1,1).repeat(1,1,D,H,W)
            valid_masks_repeated = self.slice_hard_label.repeat(1, 1, 1, H, W)
            # valid_masks_repeated = valid_masks_repeated * label_valid_repeat
            if self.TPC == True:
                predit_mask= self.cam3D_s * valid_masks_repeated
                target_mask= self.cam3D_target *label_valid_repeat * valid_masks_repeated
            else:
                predit_mask= self.cam3D_s  
                target_mask= self.cam3D_target 
            # self.loss_s_pix = self.customeBCE_mask(predit_mask, self.binary_masks * target_mask)
            self.loss_s_pix = self.customeBCE_mask(predit_mask,  target_mask)

            # self.loss_s_pix = self.customeBCE_mask(self.cam3D_s_low  , self.cam3D_target   )
            if self.Enable_teacher :
                self.loss_s = self.loss_s_v  + 0.00001*self.loss_s_pix
            else:
                self.loss_s = self.loss_s_v  
            
            # self.set_requires_grad(self.VideoNets, False)
            self.loss_s.backward()
            self.optimizer_s.step()
            self.lossDisplay_s = self.loss_s. data.mean()

    def optimization_slicevalid(self):

        pass