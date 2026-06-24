import os
import time
import pandas as pd
from utils import load_model, preprocess_image, preprocess_mask, compute_metrics, save_plot
import torch
from PIL import Image
import io
import cv2
import matplotlib as plt
# os.makedirs('results1/plots', exist_ok=True)

# model_path = './best_model.pkl'
# device = 'cpu'   # đổi thành 'cuda' nếu có GPU
# model = load_model(model_path, device)

# test_dir = './CASIA2/image'
# mask_dir = './CASIA2/groundtruths2'
# image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# configs = [
#     {'name': '100', 'quality': None, 'resize': None},
#     {'name': '95',      'quality': 95,   'resize': None},
#     {'name': '90',      'quality': 90,   'resize': None},
#     {'name': '85',      'quality': 85,   'resize': None},
#     {'name': '80',      'quality': 80,   'resize': None},
#     # {'name': 'resize_256',   'quality': None, 'resize': (256, 256)},
# ]

# results = []

# for cfg in configs:
#     print(f"Đang chạy config: {cfg['name']}")
#     config_results = []
#     i = 0
#     for fname in image_files:
#         base_name = os.path.splitext(fname)[0]
#         img_path = os.path.join(test_dir, fname)
#         mask_path = os.path.join(mask_dir, base_name + '_gt.png')
#         print(f"Processing image: {i}")
#         # if not os.path.exists(mask_path):
#         #     continue
        
#         # Preprocess
#         # if cfg['resize']:
#         #     img = Image.open(img_path).convert('RGB').resize(cfg['resize'])
#         # else:
#         #     img = Image.open(img_path).convert('RGB')
        
#         # # Simulate transmission (JPEG compression)
#         # if cfg['quality'] is not None:
#         #     buffer = io.BytesIO()
#         #     img.save(buffer, format='JPEG', quality=cfg['quality'])
#         #     buffer.seek(0)
#         #     img = Image.open(buffer).convert('RGB')
        
#         # Resize lại về 384x384 cho model
#         # img = img.resize((384, 384))
#         data = preprocess_image(img_path, cfg['quality'])  # vẫn dùng hàm gốc để nhất quán, hoặc preprocess trực tiếp từ img
        
#         gt_mask = preprocess_mask(mask_path)
        
#         t0 = time.time()
#         with torch.no_grad():
#             output = model(data)
#             # Model trả về tuple, lấy output[0] là end (logits)
#         output = output.cpu().numpy().squeeze(0)  # (384, 384)
#         output = cv2.resize(output, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
#         infer_time = time.time() - t0
        
#         metrics = compute_metrics(output, gt_mask)
#         metrics['config'] = cfg['name']
#         metrics['image'] = fname
#         metrics['infer_time_ms'] = infer_time * 1000
#         # Ước lượng volume (KB)
#         # buffer = io.BytesIO()
#         # img.save(buffer, format='JPEG', quality=cfg.get('quality') or 90)
#         # volume_kb = len(buffer.getvalue()) / 1024
#         # metrics['volume_kb'] = volume_kb
#         if  metrics['F1'] > 0 and metrics['IoU'] > 0:
#             results.append(metrics)
#             config_results.append(metrics)
#         i+=1
#         if i == 500:
#             break
    
#     # Lưu trung bình theo config
#     df_cfg = pd.DataFrame(config_results)
#     # print(df_cfg[['F1', 'IoU']].mean())

# # Lưu toàn bộ kết quả
# df = pd.DataFrame(results)
# df.to_csv('results1/config_results.csv', index=False)
df = pd.read_csv('results/config_results.csv')
# df['config'] = pd.to_numeric(df['config'])
# df = df.sort_values(by='config', ascending=False)
print(df)
# Vẽ biểu đồ
# plt.gca().invert_xaxis()
save_plot(df, 'config', ['F1', 'IoU'], 
          'Зависимость качества модели от качества входного изображения', 
          'config_f1_iou1.png', xlabel='Коэффициент качества JPEG')

print("Hoàn thành Phần B! Kết quả lưu tại results/config_results.csv và plots/")