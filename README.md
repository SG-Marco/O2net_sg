```
!pip install cityscapesScripts
!pip install -r requirements.txt
```

```
!mkdir /content/O2net_sg/O2net/datasets/test_output_dir
```

```
# cityscape dataset --> coco style
!python /content/O2net_sg/O2net/dataset_util/city2coco.py --dataset cityscapes_instance_only --outdir /content/O2net_sg/O2net/datasets/test_output_dir --datadir /content/O2net_sg/O2net/datasets/test_data_dir
```
conda install 후에 
```
!conda install -c conda-forge accimage
```

파일에 권한부여
```
!chmod +x DA.sh
!chmod +x ./tools/run_dist_launch.sh
!chmod +x ./configs/r50_deformable_detr.sh
!chmod +x ./configs/DA_r50_deformable_detr.sh
```