# train Wasserstein distance-based auto-encoder

The WAE is trained under the ImageNet(ILSVRC2015)

## Results
![image](https://github.com/wahahamyt/train_WAE/blob/main/demo.jpg)
Left are input images, right are decoded images.

## Step 1

Convert all images in IMAGENET to 64*64， and stored them in a new location.
```pyton
data_preprocessing.py
```

## Step 2

Strat training
```python
main.py
```


