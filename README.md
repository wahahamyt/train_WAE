# CAT
This repository is based on the paper "Wasserstein distance-based auto-encoder tracking", under reviewing in the journal NEPL.
## Environment
- Anaconda
- Pytorch
- visdom
```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

## Run
```shell
python tracking/run_tracker.py
```
## Evaluation
OTB protocal, LaSOT protocal, TC128 protocal

https://github.com/HengLan/LaSOT_Evaluation_Toolkit

http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html

## TODO
- Weights of Auto-Encoders will be released
- more details will be enriched
