import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataset.dataset import myDataloader
import model.base_models2 as block_buider
from dataset.dataset import Obj_num, Seperate_LR
import random
from working_dir_root import Evaluation,Fintune,Enable_student
# Seperate_LR = True # seperate left and right
from model import model_operator 
class _VideoCNN(nn.Module):
    # output width=((W-F+2*P )/S)+1

    def __init__(self, inputC=256,base_f=1024,pooling="rank"):
        super(_VideoCNN, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph

        # a side branch predict with original iamge with rectangular kernel
        # 256*256 - 128*256
        # limit=1024
        # self.Random_mask_temporal =False
        self.pooling = pooling
        self.cosine_norm = False

        Drop_out = False
        if Evaluation == True:
            Drop_out = False
            self.Random_mask_temporal = False
        if Enable_student == True:
            Drop_out = False
        self.Random_mask_temporal = False
        self.blocks = nn.ModuleList()

        #
        #
        base_f1= base_f
        self.blocks.append(block_buider.conv_keep_all(inputC, base_f1,k=(1,1,1), s=(1,1,1), p=(0,0,0), resnet= False,dropout = Drop_out))
        # self.blocks.append(nn.AvgPool3d((1,2,2),stride=(1,2,2)))
       
        base_f2= 512

        self.blocks.append(block_buider.conv_keep_all(base_f1, base_f2,k=(1,1,1), s=(1,1,1), p=(0,0,0),resnet = False,dropout = Drop_out))
        # self.blocks.append(block_buider.conv_keep_all(base_f, base_f,resnet = True,dropout=False))
        # self.blocks.append(block_buider.conv_keep_all(base_f, base_f*2,dropout=False))
        # base_f = base_f * 2
        base_f3 = 768
        # # 8*256  - 4*256\
        self.blocks.append(block_buider.conv_keep_all(base_f2, base_f3,k=(1,1,1), s=(1,1,1), p=(0,0,0),resnet = False,dropout = Drop_out))
        # base_f4 =16
        # self.blocks.append(block_buider.conv_keep_all(base_f3, base_f4,k=(1,1,1), s=(1,1,1), p=(0,0,0),resnet = False,dropout = Drop_out))
        # base_f = base_f  
        # self.classifier1 = nn.Conv3d(int(inputC), Obj_num , (1,1,1), (1,1,1), (0,0,0), bias=False)  # 4*256
        self.classifier = nn.Conv3d(int(inputC+base_f1+base_f2+base_f3), Obj_num , (1,1,1), (1,1,1), (0,0,0), bias=False)  # 4*256
        # self.classifier2 = nn.Conv3d(int(base_f2+base_f1), Obj_num , (1,1,1), (1,1,1), (0,0,0), bias=False)  # 4*256
        # self.classifier3 = nn.Conv3d(int(base_f3+base_f2+base_f1), Obj_num , (1,1,1), (1,1,1), (0,0,0), bias=False)  # 4*256



         
        # self.classifier = nn.Conv3d(int(base_f+inputC+base_f/2+base_f/4), Obj_num , (1,1,1), (1,1,1), (0,0,0), bias=False)  # 4*256
    def Top_rank_pooling (self,T,num_selected):
        B, C, D, H, W = T.size()

        # T_flattened = T.view(B, C, -1, H, W) # this is wrong as if some object is selected other object will not

# Step 2: Get the indices of the top k values along the flattened dimension
        result_tensor, indices = torch.topk(T, k=num_selected, dim=2)

        # Step 3: Use the indices to select the corresponding values along the D dimension
        # selected_values = torch.gather(T, dim=2, index=indices)

        # # Step 4: Reshape the resulting tensor to the desired shape (B, C, k, H, W)
        # result_tensor = selected_values.view(B, C, num_selected, H, W)
        Avgpool = nn.AvgPool3d((num_selected,1,1),stride=(1,1,1))
        pooled = Avgpool(result_tensor)
 
# Step 2: Get the indices of the top k values along the flattened dimension
        result_tensor, indices = torch.topk(T, k=num_selected, dim=2)

        # Step 3: Use the indices to select the corresponding values along the D dimension
        # selected_values = torch.gather(T, dim=2, index=indices)

        # # Step 4: Reshape the resulting tensor to the desired shape (B, C, k, H, W)
        # result_tensor = selected_values.view(B, C, num_selected, H, W)
        Avgpool = nn.AvgPool3d((num_selected,1,1),stride=(1,1,1))
        pooled = Avgpool(result_tensor)


         
        return pooled
    def Threshold_pooling(self,T, threshold_range=(0.2, 0.6)):
        B, C, D, H, W = T.size()

        activation = nn.Sigmoid()

        T_norm = activation(T)
        # Reshape the input tensor to (B, C, D, H*W)
        

        # Create a boolean mask based on the threshold range
        threshold_mask = (T_norm >= threshold_range[0]) & (T_norm <= threshold_range[1])
        if not torch.any(threshold_mask):
            threshold_mask= torch.ones((B, C, D, H, W), device=T.device)
        T = T*threshold_mask
        T_avg = T.sum(dim=2, keepdim=True)
        Mask_sum = threshold_mask.sum(dim=2, keepdim=True)
        # Sum along H*W to count the number of selected tensors for each (B, C, D)
        
        # Reshape the resulting tensor to the desired shape (B, C, -1)
        result_tensor =  torch.div(T_avg, Mask_sum)

        # Apply average pooling along the third dimension based on the number of selected tensors
       

        return result_tensor
    def Least_potential_pooling(self,T,num_selected):# the is the contrary of the top Rank pooling and focus on the weeker part
        pass


    def maxpooling(self,input):
       
        # Avg_pool = nn.AvgPool3d((1,2,2),stride=(1,2,2))
        # # input = Avg_pool(input)

        # input = Avg_pool (input)
        bz, ch, D, H, W = input.size()
        # activation = nn.Sigmoid()
        # activation = nn.ReLU()

        # input = Drop( input)
        # Drop = nn.Dropout(0.1)
        # input = Drop(input)
        # input = activation(input)
        # Maxpool_keepD = nn.MaxPool3d((1,H,W),stride=(1,1,1))
        # Maxpool_keepC = nn.MaxPool3d((D,1,1),stride=(1,1,1))
        flag =random. choice([False, False])
        if flag == True: 
            Maxpool_keepD = nn.MaxPool3d((1,H,W),stride=(1,1,1))
            Maxpool_keepC = nn.MaxPool3d((D,1,1),stride=(1,1,1))
        else:
            Maxpool_keepD = nn.AvgPool3d((1,H,W),stride=(1,1,1))
            Maxpool_keepC = nn.MaxPool3d((D,1,1),stride=(1,1,1))
        
        slice_valid = Maxpool_keepD(input)
        # final = Maxpool_keepC(slice_valid)
        if self.pooling == "avg":
            Maxpool_keepC = nn.AvgPool3d((D,1,1),stride=(1,1,1))
            final = Maxpool_keepC(slice_valid)
        elif self.pooling == "max":
            Maxpool_keepC = nn.MaxPool3d((D,1,1),stride=(1,1,1))
            final = Maxpool_keepC(slice_valid)
        else:
            final = self.Top_rank_pooling(slice_valid,7)
        # final = self.Threshold_pooling(slice_valid)

        #Note: how about add a number of object loss here ??
        # activation = nn.Sigmoid()
        # final = activation(final)
        # slice_valid = activation(slice_valid)

        return final, slice_valid
     
    def forward(self, x,down_in):
        bz, ch, D, H, W = x.size()
        Pure_down_pool = nn.AvgPool3d((1,2,2),stride=(1,2,2))
        if Fintune ==False:
            
            if down_in:
                x = Pure_down_pool(x)
 
        # x=F.interpolate(x,  size=(D,int( H/2), int(W/2)), mode='trilinear', align_corners=False)
        bz, ch, D, H, W = x.size()
        # self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)

        
        out = x
        droplayer=nn.Dropout(0.1)
        out = droplayer(out)

        features=[]
        for j, name in enumerate(self.blocks):
            out = self.blocks[j](out)
            if j>=2:

                out = Pure_down_pool(out)

            features.append(out)
        # bz, ch, D, H, W = out.size()
        # downsampled_mask = F.interpolate(input_flows, size=(H, W), mode='nearest')
        # expanded_mask = downsampled_mask.unsqueeze(1)
        # masked_feature = out * expanded_mask
        # cat_feature = torch.cat([out, masked_feature], dim=1)
        # cat_feature = torch.cat([out, out], dim=1)
        features[0]=F.interpolate(features[0],  size=(D,H,W), mode='trilinear', align_corners=False)
        features[1]=F.interpolate(features[1],  size=(D,H,W), mode='trilinear', align_corners=False)
        features[2]=F.interpolate(features[2],  size=(D,H,W), mode='trilinear', align_corners=False)

        # cat_feature = out
        cat_feature = torch.cat([x, features[0],features[1],features[2] ], dim=1)
        # cat_feature1 = features[1]
        # cat_feature2 = torch.cat([features[1],features[2]], dim=1)
        # cat_feature3 = torch.cat([features[1],features[2],features[3]], dim=1)
        if self.Random_mask_temporal == True:
            cat_feature =   model_operator.random_mask_out_dimension(cat_feature, 0.5, 2)

        # all_features = [cat_feature1,cat_feature2,cat_feature3]
        # all_classifers = [self.classifier1,self.classifier2,self.classifier3]
        activation = nn.Sigmoid()
        activationLU = nn.ReLU()
        # pooled, slice_valid = self.maxpooling(out)
        # pooled = pooled.view(out.size(0), -1)
        # Check the size of the final feature map
        # final = self.classifier(pooled)
        flag =random. choice([False, False])
        cam =  self.classifier(cat_feature)
        cam = activationLU(cam)

        if flag== True:
            # bz, ch, D, H, W = out.size()
            final, slice_valid = self.maxpooling(cam)
        else:
            # pooled=[]
            if self.cosine_norm == False:
                pooled, _ = self.maxpooling(cat_feature)
                droplayer=nn.Dropout(0.1)
                pooled = droplayer(pooled)
                final = self.classifier(pooled)

                _, slice_valid = self.maxpooling(cam)
                final = activation(final)
            else:
                pooled, _ = self.maxpooling(cat_feature)
                # pooled_reshaped = pooled.view(pooled.size(0), -1)  # Reshape to (B, C_in)
                weight_vector = self.classifier.weight.view(self.classifier.out_channels, -1)  # Reshape to (C_out, C_in)
                eps = 1e-8
                
                # Compute the dot product between pooled and weight_vector
                # dot_product = self.classifier(pooled)  # Size: (B, C_out)
                dot_product = torch.matmul(pooled.view(pooled.size(0), -1), weight_vector.t())  # Size: (B, C_out)
                dot_product = dot_product.view(dot_product.size(0),dot_product.size(1),1,1,1)
                # Compute the norms of pooled and weight_vector
                # norm_pooled = torch.norm(pooled, dim=1, keepdim=True)  # Size: (B, 1, 1, 1)
                norm_pooled,_ = torch.max(pooled, dim=1, keepdim=True)  # Size: (B, 1, 1, 1)
                norm_pooled = torch.abs(norm_pooled)
                norm_weight= torch.max(weight_vector ) # Size: (C_out, 1)
                norm_weight = torch.abs(norm_weight ) # Size: (C_out, 1)
                # norm_weight = torch.norm(weight_vector ) # Size: (C_out, 1)

                
                # Compute the denominator: |x| * |W| + epsilon
                denominator = norm_pooled * norm_weight + eps  # Size: (B, C_out)
                denominator = denominator.repeat(1,dot_product.size(1),1,1,1)
                # Compute the cosine similarity
                cosine_sim = dot_product / denominator  # Size: (B, C_out)
                # Get the weight vector of the final classifier (W)
                # weight_vector = self.classifier.weight.view(self.classifier.out_channels, -1)  # Reshape to (C_out, C_in)
                # eps=1e-8
                # # Compute the cosine similarity between pooled and weight_vector
                # cosine_sim = self.classifier(pooled)/(abs(max(weight_vector))*abs(max(pooled,keepdim=0))+eps)  # Compute cosine similarity
                
                # Reshape cosine_sim to match the output size of the final classifier
                # cosine_sim = cosine_sim.view(cosine_sim.size(0), cosine_sim.size(1), 1, 1, 1)  # Reshape to (B, C_out, 1, 1, 1)
                cosine_sim = cosine_sim - torch.min( cosine_sim )
                cosine_sim =  cosine_sim /(torch.max( cosine_sim )+eps)

# Normalize the final output using min-max normalization
                # normalized_final = (cosine_sim - min_value) / (max_value - min_value)
                final = cosine_sim  # Assign cosine similarity as the final prediction
                
                _, slice_valid = self.maxpooling(cam)

        # final = activation(final)
        return final, slice_valid, cam