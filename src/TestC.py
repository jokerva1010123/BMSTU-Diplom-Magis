# -*- coding: utf-8 -*-
import os
import cv2
import time
from PIL import Image
import torch
import numpy as np
import warnings
import types
import matplotlib.pyplot as plt

from Movenet7f4attention_adaption import Movenet

warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.nn.Module.dump_patches = True


def create_fal_forward(original_forward, fal_mode):
    def fal_forward(self, data, label=None):
        L1 = self.conv1(data)
        L1b = self.conv1b(L1)
        L1c = self.conv1c(data)
        L11 = self.conv11(data)
        L11b = self.conv11b(L11)
        
        if fal_mode == 'off':
            L11ada = L11b
        elif fal_mode == 'conv':
            L11a = self.conv11ada(L11b)
            L11ada = L11a
        else:
            L11a = self.conv11ada(L11b)
            L11ada = L11a + L11b
        L1b2 = torch.cat([L1b, L11ada], 1)

        L2 = self.conv2(L1b2)
        L2b = self.conv2b(L2)
        L2c = self.conv2c(data)
        L22 = self.conv22(L11b)
        L22b = self.conv22b(L22)
        
        if fal_mode == 'off':
            L22ada = L22b
        elif fal_mode == 'conv':
            L22a = self.conv22ada(L22b)
            L22ada = L22a
        else:
            L22a = self.conv22ada(L22b)
            L22ada = L22a + L22b
        L2b2 = torch.cat([L2b, L22ada], 1)

        L3 = self.conv3(L2b2)
        L3b = self.conv3b(L3)
        L3c = self.conv3c(data)
        L33 = self.conv33(L22b)
        L33b = self.conv33b(L33)
        
        if fal_mode == 'off':
            L33ada = L33b
        elif fal_mode == 'conv':
            L33a = self.conv33ada(L33b)
            L33ada = L33a
        else:
            L33a = self.conv33ada(L33b)
            L33ada = L33a + L33b
        L3b2 = torch.cat([L3b, L33ada], 1)

        L4 = self.conv4(L3b2)
        L4b = self.conv4b(L4)
        L4c = self.conv4c(data)
        L44 = self.conv44(L33b)
        L44b = self.conv44b(L44)
        L44a = self.conv44ada(L44b)
        
        if fal_mode == 'off':
            L44_to_cat = torch.zeros_like(L44a)
        elif fal_mode == 'conv':
            L44_to_cat = L44a
        else:
            L44_to_cat = L44a + L44b
        L4b2 = torch.cat([L4b, L44_to_cat], 1)

        L5 = self.conv5(L4b2)
        L55 = self.conv55(L44b)
        L55a = self.conv55ada(L55)
        
        if fal_mode == 'off':
            L55ada = L55
        elif fal_mode == 'conv':
            L55ada = L55a
        else:
            L55ada = L55a + L55
        L52 = torch.cat([L5, L55ada], 1)

        DL4 = self.deconv4(L52)
        DL4_add = DL4 + L4
        DL44 = self.deconv44(L55)
        DL44_add = DL44 + L44
        DL44b = self.deconv44b(DL44_add)
        
        if fal_mode == 'off':
            DL44bada = DL44_add
        elif fal_mode == 'conv':
            DL44bada = DL44b
        else:
            DL44bada = DL44b + DL44_add
        DL4_cat = torch.cat([DL4_add, L4c, DL44bada], 1)
        DL4_att = self.att4(DL4_cat)
        DL4b = self.deconv4b(DL4_att)

        DL3 = self.deconv3(DL4b)
        DL3_add = DL3 + L3
        DL33 = self.deconv33(DL44b)
        DL33_add = DL33 + L33
        DL33b = self.deconv33b(DL33_add)
        DL33ba = self.deconv33ada(DL33b)
        
        if fal_mode == 'off':
            DL33bada = DL33b
        elif fal_mode == 'conv':
            DL33bada = DL33ba
        else:
            DL33bada = DL33ba + DL33b
        DL3_cat = torch.cat([DL3_add, L3c, DL33bada], 1)
        DL3_att = self.att3(DL3_cat)
        DL3b = self.deconv3b(DL3_att)

        DL2 = self.deconv2(DL3b)
        DL2_add = DL2 + L2
        DL22 = self.deconv22(DL33b)
        DL22_add = DL22 + L22
        DL22b = self.deconv22b(DL22_add)
        DL22ba = self.deconv22ada(DL22b)
        
        if fal_mode == 'off':
            DL22bada = DL22_add
        elif fal_mode == 'conv':
            DL22bada = DL22b
        else:
            DL22bada = DL22b + DL22_add
        DL2_cat = torch.cat([DL2_add, L2c, DL22bada], 1)
        DL2_att = self.att2(DL2_cat)
        DL2b = self.deconv2b(DL2_att)

        DL1 = self.deconv1(DL2b)
        DL1_add = DL1 + L1
        DL11 = self.deconv11(DL22b)
        DL11_add = DL11 + L11
        DL11b = self.deconv11b(DL11_add)
        DL11ba = self.deconv11ada(DL11b)
        
        if fal_mode == 'off':
            DL11bada = DL11b
        elif fal_mode == 'conv':
            DL11bada = DL11ba
        else:
            DL11bada = DL11ba + DL11b
        DL1_cat = torch.cat([DL1_add, L1c, DL11bada], 1)
        DL1_att = self.att1(DL1_cat)
        DL1b = self.deconv1b(DL1_att)

        end = self.conv_end(DL1b)
        output = torch.argmax(end, dim=1)
        if not self.training:
            return output
        return end, None, None, output, None, None

    return fal_forward


def run_inference_no_edit(model_path, img_fn, device, fal_mode):
    img_original = Image.open(img_fn)
    orig_size = img_original.size

    img_resized = img_original.resize((384, 384))
    imgs = np.array(img_resized).astype(np.float32) / 255.
    imgs = imgs.transpose([2, 0, 1])
    data = torch.from_numpy(np.expand_dims(imgs, axis=0))

    model = Movenet([384, 384]).to(device)
    pretrain = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(pretrain.state_dict(), strict=True)
    model.eval()

    original_forward = model.forward
    model.forward = types.MethodType(create_fal_forward(original_forward, fal_mode), model)

    # t0 = time.time()
    with torch.no_grad():
        predict_ = model(data.to(device))
    # print(f"[{fal_mode.upper()}] Inference time: {time.time() - t0:.5f}s")

    predict = predict_.cpu().detach().numpy().astype(np.uint8)[0]
    output_384 = cv2.resize(predict, (384, 384), interpolation=cv2.INTER_NEAREST)
    # output_original = cv2.resize(output_384, orig_size, interpolation=cv2.INTER_NEAREST)
    
    return (output_384 * 255).astype(np.uint8)


def overlay(image, mask, alpha=0.6):
    mask_col = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_col[:, :, 0] = 0      # xanh lá = vùng forged
    mask_col[:, :, 2] = 0
    return cv2.addWeighted(image, 1 - alpha, mask_col, alpha, 0)


def create_comparison_image_no_edit(img_fn, gt_mask_fn, model_path, device="cpu"):
    print("=== ТЕСТ 3 РЕЖИМОВ FAL (без изменения оригинального кода) ===")
    img_bgr = cv2.imread(img_fn)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (384, 384), interpolation=cv2.INTER_LINEAR)

    gt = cv2.imread(gt_mask_fn, cv2.IMREAD_GRAYSCALE)
    gt_resized = cv2.resize(gt, (384, 384), interpolation=cv2.INTER_NEAREST)
    gt_bin = (gt_resized > 127).astype(np.uint8)
    fig, ax = plt.subplots(1, 5, figsize=(25, 6))

    ax[0].imshow(img_resized)
    ax[0].set_title("Входное изображение")

    ax[1].imshow(gt_bin, cmap='gray')
    ax[1].set_title("Маска")

    modes = ['off', 'conv', 'resblock']
    fal_masks = {}
    for mode in modes:
        fal_masks[mode] = run_inference_no_edit(model_path, img_fn, device, mode)
    ax[2].imshow(fal_masks['conv'])
    ax[2].set_title("Без слоев адаптации признаков")

    ax[3].imshow(fal_masks['off'])
    ax[3].set_title("Слои адаптации признаков \nкак блоки свертки")

    ax[4].imshow(fal_masks['resblock'])
    ax[4].set_title("Слои адаптации признаков \nкак остаточные блоки")

    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.savefig("figures/fal_test1.png", dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    model_path = r'./model7f4attentionadaption-291000.pkl'
    # img_fn     = "./CASIA2/image/Tp_D_CNN_M_B_nat10139_nat00059_11949.jpg"
    # gt_mask_fn = "./CASIA2/groundtruths2/Tp_D_CNN_M_B_nat10139_nat00059_11949_gt.png"  
    img_fn     = "./CASIA2/image/Tp_D_CNN_M_N_nat00013_cha00042_11093.jpg"
    gt_mask_fn = "./CASIA2/groundtruths2/Tp_D_CNN_M_N_nat00013_cha00042_11093_gt.png"  
    # img_fn     = './IMD2020/image/4.jpg'
    # gt_mask_fn = './IMD2020/mask/4.png'   # ← sửa nếu đường dẫn khác
    
    device = "cpu"   # đổi thành "cuda" nếu có GPU
    
    create_comparison_image_no_edit(img_fn, gt_mask_fn, model_path, device)