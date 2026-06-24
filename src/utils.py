import os
import cv2
import numpy as np
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
def load_model(model_path, device='cpu'):
    from model import Movenet
    model = Movenet([384, 384]).to(device)
    pretrain = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(pretrain.state_dict(), strict=True)
    model.eval()
    return model

def preprocess_image(img_path, quality, size=(384, 384)):
    img = Image.open(img_path)
    if quality is not None:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert('RGB')
    else:
        img = img.convert('RGB')
    img = img.resize(size)
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose([2, 0, 1])  # CHW
    return torch.from_numpy(np.expand_dims(img, 0))

def preprocess_mask(mask_path, size=(384, 384)):
    gt_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return gt_orig

def compute_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 128).astype(np.uint8)

    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    if TP + FP + FN == 0:
        return 1.0, 1.0  # ảnh không có splicing

    f1  = 2 * TP / (2 * TP + FP + FN)
    iou = TP / (TP + FP + FN)
    return {'F1': f1, 'IoU': iou}

def save_plot(df, x_col, y_cols, title, filename, xlabel=None):
    plt.figure(figsize=(12, 6))
    for y in y_cols:
        sns.lineplot(data=df, x=x_col, y=y, marker='o', label=y)
    plt.title(title)
    plt.xlabel(xlabel or x_col)
    plt.ylabel('Score')
    plt.gca().invert_xaxis() 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()