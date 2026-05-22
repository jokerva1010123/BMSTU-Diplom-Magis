import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Movenet7f4attention_adaption import Movenet   # file model của bạn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================== CẤU HÌNH (thay đường dẫn cho phù hợp) ======================
WEIGHT_PATH = "model7f4attentionadaption-291000.pkl"          # ← ĐƯỜNG DẪN FILE .pth CỦA BẠN
INPUT_SIZE  = 384
TEST_IMAGE  = "./CASIA2/image/Tp_D_CNN_M_N_ani00052_ani00054_11130.jpg"      # ← ảnh test
GT_MASK     = "./CASIA2/groundtruths2/Tp_D_CNN_M_N_ani00052_ani00054_11130_gt.png"       # ← mask ground truth

# ====================== LOAD MODEL ======================
def load_model():
    model = Movenet([INPUT_SIZE, INPUT_SIZE]).to(device)
    pretrain = torch.load(WEIGHT_PATH, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(pretrain.state_dict(), strict=True)
    model.eval()
    return model

model = load_model()

# Variant Full (có Attention)
model_full = load_model()

# Variant No Attention (dùng cho phần 1.2)
model_no_att = load_model()
model_no_att.att4 = torch.nn.Identity()
model_no_att.att3 = torch.nn.Identity()
model_no_att.att2 = torch.nn.Identity()
model_no_att.att1 = torch.nn.Identity()

# ====================== LOAD VÀ RESIZE ẢNH ======================
img_bgr = cv2.imread(TEST_IMAGE)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

gt = cv2.imread(GT_MASK, cv2.IMREAD_GRAYSCALE)
gt_resized = cv2.resize(gt, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
gt_bin = (gt_resized > 127).astype(np.uint8)

# ====================== INFERENCE ======================
tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
tensor = tensor.to(device)

with torch.no_grad():
    # Model eval() trả về trực tiếp mask (không phải tuple)
    pred_full   = model_full(tensor).cpu().numpy()[0]      # shape: (H, W)
    pred_no_att = model_no_att(tensor).cpu().numpy()[0]

# ====================== OVERLAY ======================
def overlay(image, mask, alpha=0.6):
    mask_col = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_col[:, :, 0] = 0      # xanh lá = vùng forged
    mask_col[:, :, 2] = 0
    return cv2.addWeighted(image, 1 - alpha, mask_col, alpha, 0)

# ====================== VẼ HÌNH ======================
fig, ax = plt.subplots(1, 4, figsize=(22, 6))

ax[0].imshow(img_resized)
ax[0].set_title("Входное изображение")

ax[1].imshow(gt_bin, cmap='gray')
ax[1].set_title("Маска")

ax[2].imshow( pred_full)
ax[2].set_title("Со слоем внимания\n")

ax[3].imshow( pred_no_att)
ax[3].set_title("Без слоя внимания\n")

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig("figures/mask_comparison_1.2.1.png", dpi=400, bbox_inches='tight')
plt.show()

print("✅ Đã lưu hình thành công tại: figures/mask_comparison_1.2.1.png")