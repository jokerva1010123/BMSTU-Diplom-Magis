# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import time
import io

# Import model
import Movenet7f4attention_adaption

st.set_page_config(page_title="Image Forgery Localization", layout="wide")
st.title("**Метод обнаружения фальсифицированного фрагмента на изображении**")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model_path = "./model7f4attentionadaption-291000.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Movenet7f4attention_adaption.Movenet([384, 384]).to(device)
    
    pretrain = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(pretrain.state_dict(), strict=True)
    model.eval()
    return model, device

model, device = load_model()

# ====================== SIDEBAR ======================
st.sidebar.header("Настройка")
alpha = st.sidebar.slider("Интенсивность цвета тепловой карты", 
                          min_value=0.1, max_value=0.9, 
                          value=0.40, step=0.05)

# ====================== UPLOAD ======================
uploaded_file = st.file_uploader("Загрузить изображение для проверки", 
                                 type=["jpg", "jpeg", "png", "tiff", "tif"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file).convert("RGB")
    orig_size = image.size  # (width, height)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Загруженное изображение")
        st.image(image, width = "stretch")

    # Нút phân tích
    if st.button("Обнаружить фальсификации", type="primary", use_container_width=True):
        with st.spinner("Анализ изображения..."):
            start_time = time.time()
            
            # Preprocess
            img_resized = image.resize((384, 384))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = img_array.transpose([2, 0, 1])  # HWC -> CHW
            data = torch.from_numpy(np.expand_dims(img_array, axis=0)).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(data)
            
            inference_time = time.time() - start_time
            
            # Xử lý output
            mask = output.cpu().detach().numpy().astype(np.uint8)[0]   # Binary mask
            
            # Resize về kích thước gốc
            mask_resized = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
            
            # ==================== РАСЧЁТ ПРОЦЕНТА ====================
            total_pixels = mask_resized.shape[0] * mask_resized.shape[1]
            forged_pixels = np.sum(mask_resized)
            percentage = (forged_pixels / total_pixels) * 100
            
            # Tạo Heatmap
            heatmap_gray = (mask_resized * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Tạo Overlay
            orig_np = np.array(image)
            overlay = cv2.addWeighted(orig_np, 1 - alpha, heatmap, alpha, 0)
            
            # ====================== ВЫВОД РЕЗУЛЬТАТА ======================
            if percentage < 0.01:  # Почти 0%
                st.success("**Не обнаружено фрагментов подделки**")
                st.info("Модель не выявила подозрительных областей на изображении.")
            else:
                st.success(f"Обнаружение завершено за **{inference_time:.3f} сек**")
                st.metric(
                    label="Площадь фальсификации",
                    value=f"{percentage:.2f}%",
                    delta=f"{forged_pixels:,} пикселей"
                )
            
            # ====================== HIỂN THỊ KẾT QUẢ ======================
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Загруженное изображение")
                st.image(image, width = "stretch")
            
            with col2:
                st.subheader("Тепловая карта")
                st.image(heatmap, width = "stretch")
            
            with col3:
                st.subheader("Наложение тепловой карты")
                st.image(overlay, width = "stretch")
            
            # ====================== DOWNLOAD ======================
            st.subheader("Загрузить результат")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            # Mask
            mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8))
            buf1 = io.BytesIO()
            mask_pil.save(buf1, format="PNG")
            buf1.seek(0)
            
            col_dl1.download_button(
                label="Маска фальсификации",
                data=buf1,
                file_name="forgery_mask.png",
                mime="image/png"
            )
            
            # Heatmap
            heatmap_pil = Image.fromarray(heatmap)
            buf2 = io.BytesIO()
            heatmap_pil.save(buf2, format="PNG")
            buf2.seek(0)
            
            col_dl2.download_button(
                label="Тепловая карта",
                data=buf2,
                file_name="forgery_heatmap.png",
                mime="image/png"
            )
            
            # Overlay
            overlay_pil = Image.fromarray(overlay)
            buf3 = io.BytesIO()
            overlay_pil.save(buf3, format="PNG")
            buf3.seek(0)
            
            col_dl3.download_button(
                label="Наложение тепловой карты",
                data=buf3,
                file_name="forgery_overlay.png",
                mime="image/png"
            )

else:
    st.info("Пожалуйста, загрузите изображение для начала обнаружения фальсификаций.")

# st.caption("Model: Movenet7f4attention_adaption | Input: 384×384")