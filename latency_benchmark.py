#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time, cv2, numpy as np, torch
from ultralytics import YOLO
from pathlib import Path

models_folder = "."
model_name = "best.engine"
source_folder = "Test" # path to images to benchmark; replace with your folder
imgsz = 640
device = 0
warmup_iterations = 10
passes = 5 # number of timed repeats over the dataset

# load model
try:
    model = YOLO(str(Path(models_folder)/model_name), task="detect")
except TypeError:
    model = YOLO(str(Path(models_folder)/model_name))

# preload images to remove disk I/O from timing
image_paths = sorted([p for p in Path(source_folder).glob("*.*") if p.is_file()])
imgs = [cv2.imread(str(p)) for p in image_paths]
assert all(im is not None for im in imgs), "One or more images failed to load."

#warm-up
with torch.inference_mode():
    for _ in range(warmup_iterations):
        _ = model.predict(source=imgs[0], imgsz=imgsz, device=device,
                          save=False, verbose=False, workers=0)
    torch.cuda.synchronize()

# timed runs (E2E: preprocess + TRT + postprocess)
e2e_ms = []
with torch.inference_mode():
    for _ in range(passes):
        for im in imgs:
            t0 = time.perf_counter()
            _ = model.predict(source=im, imgsz=imgsz, device=device,
                              save=False, verbose=False, workers=0)
            torch.cuda.synchronize()
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)

if e2e_ms:
    n = len(e2e_ms)
    print(f"E2E avg: {np.mean(e2e_ms):.2f} ms @ imgsz={imgsz}, batch=1, save=False over {n} images")
else:
    print("No successful inferences.")


