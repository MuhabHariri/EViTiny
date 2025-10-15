# 👁️ EViTiny — Redesigning the Vision Transformer for Parameters and FLOPs Efficiency

This repository accompanies the paper **"Redesigning the Vision Transformer for Parameters and FLOPs Efficiency"**

The project is structured for both **researchers** and **practitioners**, offering a clean, modular, and reproducible codebase.


---

## 📦 Features & Configuration

Interactive training prompts & clear augmentation config

- ✅ Multi-GPU distributed training 
- ✅ Interactive Training Setup — When you start training, the script prompts for:

  - Training data path
  - Validation data path
  - Number of classes
  - Number of epochs

- ✅ All data augmentation techniques and their parameters are defined in src/augmentations.py
  

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


### 3. Train the Model 
```bash
python train.py
```
---
