# train Wasserstein distance-based auto-encoder

The WAE is trained under the ImageNet(ILSVRC2015).

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


