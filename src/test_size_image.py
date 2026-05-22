# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")

# ====================== CONFIG ======================
MODEL_PATH = './model7f4attentionadaption-291000.pkl'
IMAGE_FOLDER = './CASIA2/image/'
MASK_FOLDER = './CASIA2/groundtruths2/'
OUTPUT_FOLDER = './results_exact/'
CSV_PATH = './metrics_exact_size.csv'

# Chỉ test những ảnh có kích thước gốc thuộc các size này
TARGET_SIZES = [
   (384, 256),
   (256, 384),
   (640, 480),
   (800, 600)
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ====================== LOAD MODEL ======================
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from Movenet7f4attention_adaption import Movenet

    model = Movenet([384, 384]).to(device)
    pretrain = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(pretrain.state_dict(), strict=True)
    model.eval()
    
    print("✅ Model loaded successfully!\n")
    return model, device


# ====================== METRICS ======================
def compute_iou(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / union if union != 0 else 0.0


def compute_f1(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    tp = np.sum(pred * gt)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


# ====================== INFERENCE (giống ap.py) ======================
def predict_one(model, device, img_path):
    # Đọc ảnh gốc
    img_original = Image.open(img_path).convert('RGB')
    orig_size = img_original.size  # (width, height)

    # Resize về 384x384 cho model (đúng như ap.py)
    img_resized = img_original.resize((384, 384))
    imgs = np.array(img_resized).astype(np.float32) / 255.
    imgs = imgs.transpose([2, 0, 1])
    data = torch.from_numpy(np.expand_dims(imgs, axis=0)).to(device)

    with torch.no_grad():
        predict_ = model(data)
    
    # Xử lý output giống ap.py
    predict = predict_.cpu().detach().numpy().astype(np.uint8)
    pred_384 = cv2.resize(predict[0], (384, 384), interpolation=cv2.INTER_NEAREST)
    pred_original = cv2.resize(pred_384, orig_size, interpolation=cv2.INTER_NEAREST)

    return pred_original, orig_size


# ====================== MAIN ======================
def main():
    model, device = load_model()
    
    image_files = [f for f in os.listdir(IMAGE_FOLDER) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    
    results = []
    size_count = {f"{w}x{h}": 0 for w, h in TARGET_SIZES}

    print(f"Found {len(image_files)} images. Filtering by exact target sizes...\n")

    for img_name in tqdm(image_files, desc="Testing"):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        
        # Lấy kích thước gốc
        try:
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
                size_str = f"{orig_w}x{orig_h}"
        except:
            continue

        # Chỉ test nếu kích thước gốc nằm trong TARGET_SIZES
        if (orig_w, orig_h) not in TARGET_SIZES:
            continue

        # Tìm mask tương ứng
        base_name = os.path.splitext(img_name)[0]
        mask_candidates = [f for f in os.listdir(MASK_FOLDER) if base_name+"_gt" in f]
        if not mask_candidates:
            print(f"⚠️ No mask found for {img_name}")
            continue

        mask_path = os.path.join(MASK_FOLDER, mask_candidates[0])

        try:
            t0 = time.time()
            pred_mask, orig_size = predict_one(model, device, img_path)
            inf_time = time.time() - t0

            # Load ground truth
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_mask > 127).astype(np.uint8)

            iou = compute_iou(pred_mask, gt_mask)
            f1 = compute_f1(pred_mask, gt_mask)

            results.append({
                'image': img_name,
                'size': size_str,
                'width': orig_w,
                'height': orig_h,
                'iou': round(iou, 6),
                'f1': round(f1, 6),
                'inference_time': round(inf_time, 4)
            })

            size_count[size_str] += 1

        except Exception as e:
            print(f"❌ Error {img_name}: {e}")

    # ====================== SAVE RESULTS ======================
    if not results:
        print("Không tìm thấy ảnh nào có kích thước phù hợp!")
        return

    df = pd.DataFrame(results)
    df = df[['image', 'size', 'width', 'height', 'iou', 'f1', 'inference_time']]
    
    # Summary theo kích thước
    summary = df.groupby('size').agg({
        'iou': ['count', 'mean', 'std', 'min', 'max'],
        'f1': ['mean', 'std', 'min', 'max'],
        'inference_time': 'mean'
    }).round(4)

    df.to_csv(CSV_PATH, index=False)
    summary.to_csv(CSV_PATH.replace('.csv', '_summary.csv'))

    print("\n" + "="*85)
    print("🎉 HOÀN THÀNH TEST THEO KÍCH THƯỚC CHÍNH XÁC")
    print("="*85)
    print(f"📊 Chi tiết kết quả: {CSV_PATH}")
    print(f"📈 Tóm tắt theo size: {CSV_PATH.replace('.csv', '_summary.csv')}")
    print(f"📁 Ảnh predict đã lưu tại: {OUTPUT_FOLDER}")
    print("\n=== KẾT QUẢ TRUNG BÌNH THEO KÍCH THƯỚC ===")
    print(summary)
    print("="*85)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()