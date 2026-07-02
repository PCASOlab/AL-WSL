# Active Learning for Efficient Annotation of Surgical Videos with Weak Supervision

## Description
This is an official implementation of the Active Learning with Weak Supervision and Mask loss model. 
> [Active Learning for Efficient Annotation of Surgical Videos with Weak Supervision] | IPCAI 2025


## Abstract
Precise spatial-temporal annotation of laparoscopic videos is time-consuming and requires expert knowledge. We propose a human-in-the-loop knowledge acquisition framework that combines active learning with dual-loss optimization to significantly reduce the annotation effort needed for automatic localization and segmentation of objects in the surgical field. Our method employs a foundation model to generate temporally consistent class activation maps (CAMs) from video using two complementary training objectives: a weak supervision loss on video-level tool presence labels for unannotated data, and an image-level mask loss on human-corrected annotations obtained through active learning. Rather than requiring dense pixel-level annotation upfront, our pipeline iteratively proposes pseudo-masks that guide the expert annotator to refine the knowledge previously captured by the model. We demonstrate that our framework reduces the effort of surgical video annotation by 50% by the end of training in comparison to fully manual annotation. Through eliminating the need for large, fully annotated datasets from the start, this framework enables scalability to the development of surgical tool segmentation models. This iterative human-in-the-loop refinement supports efficient knowledge acquisition with minimal expert input, providing a practical and deployable strategy for expanding tool segmentation to larger, more diverse datasets and real-world clinical settings.
<p align="center">
  <img src="figure/model_diagram.pdf" width="80%" />
</p>
 Our method leverages a weak supervision model at the video-level that operates on frame representations extracted from the DINOv3 ViT-B/16 (Vision Transformer Base with 16×16 patches) foundation model, pretrained in a self-supervised manner on the LVD-1689M dataset, a collection of around 1.68 billion images drawn from a pool of 17 billion public web images. 

The model consists of a frozen DINOv3 frame‑wise feature extractor followed by a trainable temporal consistency network. Given a raw video clip, each frame is passed independently through DINOv3 to obtain per‑frame spatial feature maps. We chose DINOv3 over other backbones as evidence showing that it can outperform several popular foundation models (e.g., CLIP and SAM) in downstream dense prediction tasks that lie outside the pretraining domain of those foundation models. These features are then aggregated across time by the temporal consistency network, which enforces spatially coherent activations between adjacent frames. The network has two prediction heads: (1) a video‑level classification prediction for weak supervision, and (2) a dense prediction head that produces a class activation map (CAM) of size, localizing tools throughout the entire clip without requiring pixel‑level annotations.

Our method achieved 38% reduction in annotation time compared with fully manual segmentation.It improved in segmentation of minority surgical tool classes, and the combined pipeline showed a better localization score compared to pure Weakly Supervised learning.

## Repository Overview

* `main.py`: Model training and visualization
* `data_cholec_reader_convert.py`: Convert Cholec80 raw data to clips for training.
* `data_cholec_seg8k_convert.py`: Convert Cholec_seg8k with segmentation ground-truth to clips for evaluation
* `config.yaml`: Default parameters and setting 
* `model`: Model code
    * `base_models.py`: code for basic MLP, 3D CNN structures
    * `model_3dcnn_linear_TC_v3.py`: Teacher module based on MLP
    * `model_3dcnn_linear_ST_v3.py`: Student module based on 3DCNN
    * `model_infer_TC_dinov3.py`: Key implementation for the Mask loss implementation for Active Learning with Weakly Supervised Learning
    * `vision_transformer.py`: ViT backbone, refer to [Vit pytorch implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
* `data`: Code for data pre-processing
    * `data.py`: load data to batches, and preprocessing on-the-fly
     
### Dependencies

Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- PIL
- OpenCV

 
### Clone the repository:
   ```bash
   git clone https://github.com/PCASOlab/AL-WSL.git
   cd AL-WSL
   ```

### Preparing Data

* download raw cholec80 data:
Cholec80 dataset: [Download Cholec80](https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz)

* Convert the raw cholec80 data into pkls of video clips in order to run training.

* Separate some part of the data for testing set, and do not run training on those.

* Upload the raw videos to encord, as this is what the model will look at to incorporate model predictions.
* Get an Encord SSH private key to be able to connect to Encord when running AL, and place it in the base folder.

### Config
The config file requires paths defined in the active_learning section for 
* ssh_private key, 
* project title (from Encord), 
* and project hash (from Encord)
and the number defined in 
* max_rounds: the number of AL round to be conducted
* active_learning_interval: the number of epochs in between each AL round. 

Data Paths section requires:
* video_path: path to pkl files of the training data
* cholec_gt_dir: path to ground truth dir, where all annotated videos will be sent after AL rounds
* original_videos_dir: path to original videos
* sam_feature_olg_dir: path to SAM features

Dataset Selection requires:
* selected_data: Cholec dataset is what was used for this study
* AL_folder: folder name of where the selected videos, before encord, and after encord are sent
* checkpt_dir: folder where all AL round checkpoints are sent
* num_videos: number of selected videos wanted to be annotated per round
* min_minor_videos: number of selected videos that must have a minor tool category

Weak Supervised Learning Mask Loss weight:
* mask_loss: the mask loss weight (Study used α=0.1 and α=0.01)

### Train the model 

 * pretrained backbone can be downloaded here: "https://upenn.box.com/s/nsukq51tbdxvlgh6lugnkvufnt42blk1".  paste the dino_deitsmall8_pretrain.pth under the config_root folder

* Training Script: main.py
 
To train the model, 
First, set the encord information, mask loss weight, dataset, and paths to the videos, ground truth folder, sam_features dir, and the original videos
in config.yaml, then 
run:
python main.py

To evaluate the model:
Set the folder to where the testing set is located in file working_para/working_dir_root_eval_cholec_p.py line 18 
Then visualize activation maps, 
run:
python main_eval.py

## Cite

```

```
## License

This project is released under the PENN ACADEMIC SOFTWARE LICENSE AGREEMENT. See the LICENSE file for more details.

## Contact

If you have any questions about the code, please contact Manasa Dendukuri dendukuri.manasa@gmail.com



 
