import os
import pandas as pd
from utils import load_model, preprocess_image, preprocess_mask, compute_metrics, save_plot
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs('results/plots', exist_ok=True)

model_path = './best_model.pkl'
device = 'cpu'
model = load_model(model_path, device)

test_dir = 'test_images'
mask_dir = 'test_masks'
image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Định nghĩa các nhóm đặc điểm (bạn có thể chỉnh lại)
def get_feature_group(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Texture complexity (variance of Laplacian)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture = 'high_texture' if lap_var > 500 else 'low_texture'
    
    # Simple content classification (có thể cải tiến bằng CLIP sau)
    h, w = gray.shape
    edge_density = np.sum(cv2.Canny(gray, 100, 200) > 0) / (h * w)
    content = 'portrait' if edge_density < 0.05 else 'landscape'  # heuristic đơn giản
    
    forgery_type = 'splicing'  # nếu bạn có metadata, thay bằng thật
    
    return {
        'texture': texture,
        'content': content,
        'forgery_type': forgery_type
    }

results = []

for fname in image_files:
    img_path = os.path.join(test_dir, fname)
    mask_path = os.path.join(mask_dir, fname)
    if not os.path.exists(mask_path):
        continue
    
    features = get_feature_group(img_path)
    data = preprocess_image(img_path)
    gt_mask = preprocess_mask(mask_path)
    
    with torch.no_grad():
        output = model(data.to(device))
        pred_logits = output[0] if isinstance(output, tuple) else output
        pred = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
    
    metrics = compute_metrics(pred, gt_mask)
    metrics.update(features)
    metrics['image'] = fname
    results.append(metrics)

df = pd.DataFrame(results)
df.to_csv('results/feature_results.csv', index=False)

# Vẽ boxplot theo đặc điểm
plt.figure(figsize=(10, 6))
sns.boxplot(x='texture', y='F1', data=df)
plt.title('F1 theo độ phức tạp texture của dữ liệu gốc')
plt.savefig('results/plots/feature_texture_f1.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='content', y='F1', data=df)
plt.title('F1 theo loại nội dung (portrait vs landscape)')
plt.savefig('results/plots/feature_content_f1.png', dpi=300, bbox_inches='tight')
plt.close()

# Biểu đồ trung bình
mean_df = df.groupby(['texture', 'content'])[['F1', 'IoU']].mean().reset_index()
print(mean_df)

print("Hoàn thành Phần C! Kết quả lưu tại results/feature_results.csv và plots/")