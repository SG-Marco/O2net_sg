```
!pip install -r requirements.txt
```
```
# cityscape dataset --> coco style
!python /content/O2net_sg/O2net/dataset_util/city2coco.py --dataset cityscapes_instance_only --outdir 출력될 디렉토리경로 --datadir 입력 데이터 경로
```

epochs, lr 등 조절 어디서?
gpu개수 여러개 하려면 숫자들 어떻게?
데이터들 넣을 방법(구글드라이브?), 데이터 코코스타일 json 만들 때 디렉토리 구조 파악하기.