# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
from PIL import Image
import Movenet7f4attention_adaption   # file model của bạn

def compute_metrics(pred, gt):
    """
    Tính F1 và IoU theo công thức trong paper
    pred, gt: numpy array (H, W), giá trị 0/1 hoặc 0/255
    """
    # Chuyển về binary 0/1
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 128).astype(np.uint8)   # threshold 128 cho mask thật

    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    if TP + FP + FN == 0:
        return 1.0, 1.0  # ảnh không có splicing

    f1  = 2 * TP / (2 * TP + FP + FN)
    iou = TP / (TP + FP + FN)
    return f1, iou


def main():
    # ========================== CẤU HÌNH ==========================
    model_path     = r'./model7f4attentionadaption-291000.pkl'   # đường dẫn model
    input_folder   = r'./CASIA2/image'          # Folder chứa ảnh spliced
    gt_folder      = r'./CASIA2/groundtruths2'          # Folder chứa mask thật
    device         = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ============================================================

    print(f"Đang dùng device: {device}")

    # Load model
    model = Movenet7f4attention_adaption.Movenet([384, 384]).to(device)
    pretrain = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(pretrain.state_dict(), strict=True)
    model.eval()

    f1_list = []
    iou_list = []

    # Lấy tất cả file ảnh trong folder1
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', ".tif"))]

    print(f"Tìm thấy {len(image_files)} ảnh để evaluate...\n")
    for idx, img_name in enumerate(image_files, 1):
        base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(input_folder, img_name)

        # Tìm mask tương ứng (ưu tiên .png, sau đó .jpg)
        gt_path = os.path.join(gt_folder, base_name + '_gt.png')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_folder, base_name + '_gt.jpg')
        if not os.path.exists(gt_path):
            print(f"⚠️  Không tìm thấy GT cho {img_name}")
            continue
        
        # === Đọc ảnh gốc ===
        img_original = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_original.size
        
        # Resize về 384x384 cho model
        img_resized = img_original.resize((384, 384))
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)                    # CHW
        data = torch.from_numpy(np.expand_dims(img_np, 0)).to(device)

        # === Inference ===
        with torch.no_grad():
            pred_384 = model(data)                     # shape: (1, 384, 384), giá trị 0/1

        pred_384 = pred_384.cpu().numpy().squeeze(0)   # (384, 384)

        # Resize mask dự đoán về kích thước gốc
        pred_orig = cv2.resize(pred_384, (orig_w, orig_h), 
                               interpolation=cv2.INTER_NEAREST)
        
        # === Đọc GT mask ===
        gt_orig = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        print(pred_orig.size, gt_orig.size)
        if gt_orig is None:
            print(f"❌ Không đọc được GT: {gt_path}")
            continue
        if pred_orig.size != gt_orig.size:
            continue
        # === Tính metric ===
        try:
            f1, iou = compute_metrics(pred_orig, gt_orig)

            f1_list.append(f1)
            iou_list.append(iou)

            print(f"[{idx:3d}/{len(image_files)}] {img_name:30s} → F1: {f1:.4f} | IoU: {iou:.4f}")
        except:
            output_path = img_name+'.jpg'
            cv2.imwrite(output_path, pred_orig * 255)
            print(f"Saved output to: {output_path}")
            continue
    # ======================== KẾT QUẢ CUỐI ========================
    if f1_list:
        print("\n" + "="*60)
        print("📊 KẾT QUẢ TRUNG BÌNH TOÀN DATASET")
        print(f"Average F1-score : {np.mean(f1_list):.4f}")
        print(f"Average IoU      : {np.mean(iou_list):.4f}")
        print("="*60)
    else:
        print("Không có ảnh nào được evaluate!")

if __name__ == '__main__':
    main()