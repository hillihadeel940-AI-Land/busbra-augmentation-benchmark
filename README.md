# busbra-augmentation-benchmark
# BUS-BRA Augmentation Benchmark

Code and results for the paper:

**Systematic Evaluation of Data Augmentation Strategies
for Ordinal BI-RADS Classification in Breast Ultrasound:
A Benchmark on BUS-BRA**

Hadeel Saad Alghamdi, Mufti Mahmud, Abdullah Faisal Al-Battal,
Ammar Alsheghri

King Fahd University of Petroleum and Minerals, Dhahran,
Saudi Arabia

## Requirements

```bash
pip install torch torchvision albumentations scikit-learn
pandas numpy pillow scipy
```

## Usage

```bash
# Run one augmentation condition
python busbra_pipeline_v3.py \
    --data_root /path/to/BUS-BRA \
    --aug modern \
    --backbone efficientnet \
    --epochs 100 \
    --patience 20

# Available augmentations:
# baseline, geometric, intensity, ultrasound, modern, combined

# Available backbones:
# efficientnet, resnet50
```

## Dataset

BUS-BRA is publicly available at:
https://doi.org/10.1002/mp.17427

## Results

Full 5-fold CV results are provided in the `results/` folder.

## Citation

If you use this code, please cite our paper:
[citation will be added after publication]
