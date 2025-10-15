# 🧠 Ghost-DeepResNet — An ultra-lightweight classification model with very low FLOPs and parameter count for identifying strawberry diseases and disorders

This repository accompanies the paper **"Redesigning the Vision Transformer for Parameters and FLOPs Efficiency"**

The project is structured for both **researchers** and **practitioners**, offering a clean, modular, and reproducible codebase.


---

## 📦 Key Features

Interactive training prompts & clear augmentation config

- ✅ Interactive Training Setup — When you start training, the script prompts for:

-- Training data path

-- Validation data path

-- Number of classes

-- Number of epochs

- ✅ Augmentations at a Glance — All data augmentation techniques and their parameters are defined in src/augmentations.py
  
- ✅ Multi-GPU distributed training 

---

## 🚀 Getting Started
### 1. Clone the Repository

```bash
git clone https://github.com/MuhabHariri/EViTiny.git
```
```bash
cd EViTiny
```


---

### 2. Install Requirements

```bash
pip install -r requirements.txt
```



---


### 4. Train the Model 
```bash
python train.py
```
---
