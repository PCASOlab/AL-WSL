import numpy as np

class_name_Cholec_8k={0: 'Black Background',
                    1: 'Abdominal Wall',
                    2: 'Liver',
                    3: 'Gastrointestinal Tract',
                    4: 'Fat',
                    5: 'Grasper',
                    6: 'Connective Tissue',
                    7: 'Blood',
                    8: 'Cystic Duct',
                    9: 'L-hook Electrocautery',
                    10: 'Gallbladder',
                    11: 'Hepatic Vein',
                    12: 'Liver Ligament'}

categories = [
        'Grasper', #0   
        'Bipolar', #1    
        'Hook', #2    
        'Scissors', #3      
        'Clipper',#4       
        'Irrigator',#5    
        'SpecimenBag',#6     
        # 'SelectedFrame'             
    ]
categories_cholec = [
        'Grasper', #0   
        'Bipolar', #1    
        'Hook', #2    
        'Scissors', #3      
        'Clipper',#4       
        'Irrigator',#5    
        'SpecimenBag',#6  
        # 'SelectedFrame'                
]
categories_thoracic = [
    'Lymph node',
    'Vagus nereve',
    'Bronchus',
    'Lung parenchyma',
    'Instruments', 
    ]

categories_endovis =  [
    'Prograsp_Forceps_labels',
    'Large_Needle_Driver_labels',
    'Grasping_Retractor_labels',
    'Bipolar_Forceps_labels',
    'Vessel_Sealer_labels',
    'Monopolar_Curved_Scissors_labels',
    'Other_labels'
]
   
def label_from_endovis(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_endovis)
    new_label = inputlabel>5
    # new_label[:,0,:,:] = inputlabel[:,5,:,:]
    # new_label[:,2,:,:] = inputlabel[:,9,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label

def label_from_seg8k_2_cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,5,:,:] # swap
    new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
# def label_from_full_cholec(inputlabel): #(13,29,256,256)
#     in_ch,in_D,H,W =  inputlabel.shape
#     inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
#     lenth = len(categories_cholec)
#     new_label = np.zeros((in_D,lenth,H,W))
#     # new_label[:,0,:,:] = inputlabel[:,5,:,:] # swap
#     # new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
#     new_label  = inputlabel # swap

#     frame_label=np.sum(new_label,axis=(2,3))
#     frame_label=(frame_label>20)*1.0
#     video_label=np.max(frame_label, axis=0)
#     mask = np.transpose(new_label , (1, 0, 2, 3)) 
#     return mask,frame_label,video_label
def label_from_full_cholec(inputlabel_dict):
    """
    Process labels from cholec PKL file format
    
    Args:
        inputlabel_dict: Dictionary containing 'masks', 'frames', and 'labels' keys
                        where 'masks' has shape (7, 29, H, W)
    
    Returns:
        mask: (7, 29, H, W)
        frame_label: (29, 7) - binary labels per frame per tool
        video_label: (7,) - binary labels per tool for entire video
    """
    # Extract the masks from the dictionary
    if isinstance(inputlabel_dict, dict):
        # Debug: print available keys
        print(f"Dictionary keys: {inputlabel_dict.keys()}")
        
        # Try different possible key names
        if 'masks' in inputlabel_dict:
            inputlabel = inputlabel_dict['masks']
        elif 'mask' in inputlabel_dict:
            inputlabel = inputlabel_dict['mask']
        elif 'labels' in inputlabel_dict:
            # Check if 'labels' contains the mask array
            if isinstance(inputlabel_dict['labels'], np.ndarray):
                inputlabel = inputlabel_dict['labels']
            else:
                raise KeyError(f"Cannot find mask data. Available keys: {inputlabel_dict.keys()}")
        else:
            raise KeyError(f"Cannot find mask data. Available keys: {inputlabel_dict.keys()}")
    else:
        # If it's already an array (backward compatibility)
        inputlabel = inputlabel_dict
    
    in_ch, in_D, H, W = inputlabel.shape  # Expected: (7, 29, H, W)
    
    # Transpose to (29, 7, H, W) for processing
    inputlabel = np.transpose(inputlabel, (1, 0, 2, 3))
    
    lenth = 7  # Number of categories
    new_label = np.zeros((in_D, lenth, H, W))
    
    new_label = inputlabel  # Use the masks as-is
    
    # Create frame-level labels: if more than 20 pixels are present, consider tool present
    frame_label = np.sum(new_label, axis=(2, 3))  # Sum over H, W -> (29, 7)
    frame_label = (frame_label > 20) * 1.0  # Binary threshold
    
    # Create video-level labels: if tool appears in any frame
    video_label = np.max(frame_label, axis=0)  # Max over frames -> (7,)
    
    # Transpose mask back to (7, 29, H, W)
    mask = np.transpose(new_label, (1, 0, 2, 3))
    
    return mask, frame_label, video_label
def label_from_seg8k_full(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_cholec)
    new_label = np.zeros((in_D,lenth,H,W))
    # new_label[:,0,:,:] = inputlabel[:,5,:,:] # swap
    # new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    new_label  = inputlabel # swap
    # new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
def label_from_thoracic(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_thoracic)
    new_label = inputlabel
    # new_label[:,0,:,:] = inputlabel[:,5,:,:]
    # new_label[:,2,:,:] = inputlabel[:,9,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label


def label_from_thoracic2cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,0,:,:]
    new_label[:,1,:,:] = inputlabel[:,1,:,:]
    new_label[:,2,:,:] = inputlabel[:,2,:,:]
    new_label[:,3,:,:] = inputlabel[:,3,:,:]
    new_label[:,4,:,:] = inputlabel[:,4,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label



