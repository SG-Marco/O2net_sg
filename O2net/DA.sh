# pre-training the source model
CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/r50_deformable_detr.sh --output_dir exps/source_model --dataset_file city2foggy_source --coco_path /content/drive/MyDrive/Cityscapes_data/data

# training the proposed method
CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/DA_r50_deformable_detr.sh --output_dir exps/ours --transform make_da_transforms --dataset_file city2foggy --checkpoint --coco_path /content/drive/MyDrive/Cityscapes_data/data/l2_norm/checkpoint.pth --coco_path /content/drive/MyDrive/Cityscapes_data/data


