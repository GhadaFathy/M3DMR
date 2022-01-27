# Implementation of M3DMR 
This implementation includes the published paper:
 https://peerj.com/articles/cs-529/#
  
The repository includes:
* Source code of M3DMR, the sourec code divided to:
  - Learning Techniques Phase for Frame Depth, Camera Pose, and Object Motion (preprocessing, Code, and Evaluation)
  - 3D Model Reconstruction with Point Cloud Fusion (code and Evaluation)
* Paper : A novel no-sensors 3D model reconstruction from monocular video frames for a dynamic environment


The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (BibTeX below).
## Datasets 
there are many Datasets used in M3DMR
 
 * COCO,ImageNet, and ResNet18 Dataset
 
## Requirements
TensorFlow using Python, OpenGL, and open3D, GPU
The operating system is Linux (CentOS).
## Compile and Run

In the first stage [Learning Techniques Phase for Frame Depth, Camera Pose, and Object Motion]
'This code is implemented and supported by Vincent Casser and Anelia Angelova and can be found at
https://sites.google.com/view/struct2depth.'
there are python files for preprocessing, a file for training, a file for online refinement, and files for evaluations. 
The outputs of online refinement are the input for the point-cloud process.

##To run training:
ckpt_dir="your/checkpoint/folder"
data_dir="KITTI_SEQ2_LR/" # Set for KITTI
imagenet_ckpt="resnet_pretrained/model.ckpt"

python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --architecture resnet \
  --imagenet_ckpt $imagenet_ckpt \
  --imagenet_norm true \
  --joint_encoder false

Running depth/ego-motion inference:
input_dir="your/image/folder"
output_dir="your/output/folder"
model_checkpoint="your/model/checkpoint"

python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion true \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint

##Running online-refinement:
prediction_dir="some/prediction/dir"
model_ckpt="checkpoints/checkpoints_baseline/model-199160"
handle_motion="true"
size_constraint_weight="0" 
data_dir="KITTI_SEQ2_LR_EIGEN/"
triplet_list_file="$data_dir/test_files_eigen_triplets.txt"
triplet_list_file_remains="$data_dir/test_files_eigen_triplets_remains.txt"
ft_name="kitti"

python optimize.py \
  --logtostderr \
  --output_dir $prediction_dir \
  --data_dir $data_dir \
  --triplet_list_file $triplet_list_file \
  --triplet_list_file_remains $triplet_list_file_remains \
  --ft_name $ft_name \
  --model_ckpt $model_ckpt \
  --file_extension png \
  --handle_motion $handle_motion \
  --size_constraint_weight $size_constraint_weight

##Running Point-Cloud fusion: 
the secone Stage [3D Model Reconstruction with Point Cloud Fusion].
data_dir="path/ for camera/ intrinsic/ file"
input_dir="/output/from/online refinment/process/"

Point-Cloud Fusion.py --data_dir $data_dir \
 --input_dir $input_dir\




## Citation
Use this BibTeX to cite this repository:
```
Fathy GM, Hassan HA, Sheta W, Omara FA, Nabil E. 2021. 
A novel no-sensors 3D model reconstruction from monocular video frames for a dynamic environment.
PeerJ Computer Science 7:e529 https://doi.org/10.7717/peerj-cs.529

