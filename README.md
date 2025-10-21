# 👁️ EViTiny — Lightweight MLP-Free Vision Transformer with Embedding-Dimension Compression, Nonlinear Projection and Residual Global-Context Channel Gating

This repository accompanies the paper **"Lightweight MLP-Free Vision Transformer with Embedding-Dimension Compression, Nonlinear Projection and Residual Global-Context Channel Gating"**

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


