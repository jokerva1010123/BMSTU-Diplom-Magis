# -*- coding: utf-8 -*-
import os
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

def count_images_by_size(folder_path):
    """
    Đếm số lượng ảnh theo kích thước (width x height)
    """
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục không tồn tại: {folder_path}")
        return

    size_count = defaultdict(int)
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    print(f"Đang quét {len(image_files)} file ảnh trong thư mục: {folder_path}\n")
    
    for filename in tqdm(image_files, desc="Đang xử lý"):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                size_str = f"{width} x {height}"
                size_count[size_str] += 1
        except Exception as e:
            print(f"⚠️  Không đọc được {filename}: {e}")
            continue

    # Sắp xếp theo số lượng ảnh giảm dần
    sorted_sizes = sorted(size_count.items(), key=lambda x: x[1], reverse=True)

    # In kết quả
    print("\n" + "="*60)
    print("📊 THỐNG KÊ KÍCH THƯỚC ẢNH")
    print("="*60)
    print(f"{'Kích thước':<18} {'Số lượng':<10} {'Tỷ lệ'}")
    print("-" * 60)
    
    total_images = len(image_files)
    
    for size, count in sorted_sizes:
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"{size:<18} {count:<10} {percentage:6.2f}%")
    
    print("-" * 60)
    print(f"Tổng số ảnh: {total_images}")
    print("="*60)

    # Nếu muốn xem top 10 kích thước phổ biến nhất
    print(f"\n🔝 Top 10 kích thước phổ biến nhất:")
    for size, count in sorted_sizes[:10]:
        print(f"   • {size}: {count} ảnh")


if __name__ == '__main__':
    # ====================== CONFIG ======================
    IMAGE_FOLDER = './CASIA2/image/'   # Thay đổi nếu cần
    
    count_images_by_size(IMAGE_FOLDER)