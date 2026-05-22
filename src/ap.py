# -*- coding: utf-8 -*-
import os
import cv2
import time
from PIL import Image
import torch
import Movenet7f4attention_adaption
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.nn.Module.dump_patches = True

def test_one():
    model_path = r'./model7f4attentionadaption-291000.pkl'
    img_fn = './IMD2020/image/4.jpg'  # ảnh 768x1024 của bạn
    
    # Đọc ảnh gốc
    img_original = Image.open(img_fn)
    orig_size = img_original.size  # (width, height) = (1024, 768)
    print(f"Input original size: {orig_size}")
    
    # Resize về 384x384 cho model
    img_resized = img_original.resize((384, 384))
    imgs = np.array(img_resized).astype(np.float32) / 255.
    imgs = imgs.transpose([2, 0, 1])  # (H, W, C) -> (C, H, W)
    data = torch.from_numpy(np.expand_dims(imgs, axis=0))  # (1, C, H, W)
    
    device = "cpu"  # hoặc "cuda" nếu có GPU
    print(f"Using device: {device}")
    
    # Khởi tạo model với kích thước 384x384
    model = Movenet7f4attention_adaption.Movenet([384, 384]).to(device)
    
    # Load pretrained weights
    pretrain = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(pretrain.state_dict(), strict=True)    
    model.eval()
    
    # Inference
    t0 = time.time()
    with torch.no_grad():
        predict_ = model(data.to(device))
    
    inference_time = time.time() - t0
    print(f"Inference time: {inference_time:.5f}s")
    
    predict = predict_.cpu().detach().numpy().astype(np.uint8)
    print(f"Raw output shape (from model): {predict.shape}") 
    
    output_384 = cv2.resize(predict[0], (384, 384), interpolation=cv2.INTER_NEAREST)
    print(f"After first upsample (to 384x384): {output_384.shape}")
    
    output_original = cv2.resize(output_384, orig_size, interpolation=cv2.INTER_NEAREST)
    print(f"After second upsample (to original size): {output_original.shape}")
    
    # Lưu kết quả
    output_path = 'yu_output.jpg'
    cv2.imwrite(output_path, output_original * 255)
    print(f"Saved output to: {output_path}")
    
    return output_original

if __name__ == '__main__':
    torch.cuda.empty_cache()
    result = test_one()
    print("Done!")