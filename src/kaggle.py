# -*- coding: utf-8 -*-
"""
Full training script for Movenet with attention
Includes: Model definition, training loop, metrics logging (IoU, F1, Precision, Recall, Accuracy)
Loads images and masks from folders, automatically splits train/val
Saves best model based on IoU
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# DATASET CLASS
# ============================================================================

class SegmentationDataset(Dataset):
    """
    Dataset for segmentation with images and masks from folders
    Folder structure:
        images_folder/
            image1.jpg
            image2.png
            ...
        masks_folder/
            mask1.png
            mask2.png
            ...
    """
    def __init__(self, image_paths, mask_paths, input_size=(384, 384)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = input_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = image.resize(self.input_size, Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        
        # Load mask (grayscale, binary)
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask = mask.resize(self.input_size, Image.NEAREST)
        mask = np.array(mask).astype(np.uint8)
        
        # Convert to binary (0/1) if needed
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        
        # Create edge map from mask (simple edge detection)
        mask_edge = cv2.Canny(mask * 255, 50, 150) > 0
        mask_edge = mask_edge.astype(np.float32)
        
        # Create data edge (edge from image) - using simple edge detection
        img_gray = cv2.cvtColor((image.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        data_edge = cv2.Canny(img_gray, 50, 150).astype(np.float32) / 255.0
        data_edge = np.expand_dims(data_edge, axis=0)
        
        return (torch.from_numpy(image),
                torch.from_numpy(data_edge),
                torch.from_numpy(mask),
                torch.from_numpy(mask_edge))
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable size data"""
        images = torch.stack([item[0] for item in batch])
        edges = torch.stack([item[1] for item in batch])
        masks = torch.stack([item[2] for item in batch])
        mask_edges = torch.stack([item[3] for item in batch])
        return images, edges, masks, mask_edges


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Movenet(nn.Module):
    def __init__(self, data_size):
        super(Movenet, self).__init__()

        bh = data_size[0] // 8
        bw = data_size[1] // 8
        self.original_size = data_size

        bn = 32  # base output number
        
        # -------------- conv -------------- #
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv1c = nn.Conv2d(3, bn, kernel_size=1, padding=0, stride=1)
        self.conv11 = nn.Sequential(
            nn.Conv2d(3, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv11b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv11ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * bn, 2 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * bn),
            nn.ReLU(True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(2 * bn, 2 * bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(2 * bn),
            nn.ReLU(True)
        )
        self.conv2c = nn.Conv2d(3, bn, kernel_size=2, padding=0, stride=2)
        self.conv22 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv22b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv22ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3 * bn, 4 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * bn),
            nn.ReLU(True)
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(4 * bn, 4 * bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(4 * bn),
            nn.ReLU(True)
        )
        self.conv3c = nn.Conv2d(3, bn, kernel_size=4, padding=0, stride=4)
        self.conv33 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv33b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv33ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(5 * bn, 8 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )
        self.conv4b = nn.Sequential(
            nn.Conv2d(8 * bn, 8 * bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )
        self.conv4c = nn.Conv2d(3, bn, kernel_size=8, padding=0, stride=8)
        self.conv44 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv44b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv44ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )

        # -------------- middle -------------- #
        self.conv5 = nn.Sequential(
            nn.Conv2d(9 * bn, 8 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )
        self.conv55 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv55ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )

        # -------------- deconv -------------- #
        self.deconv4 = nn.Sequential(
            nn.Upsample(size=(bw, bh), mode='bilinear'),
            nn.Conv2d(9 * bn, 8 * bn, kernel_size=1),
        )
        self.att4 = SELayer(channel=10*bn, reduction=16)
        self.deconv4b = nn.Sequential(
            nn.Conv2d(10 * bn, 8 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )
        self.deconv44 = nn.Sequential(
            nn.Upsample(size=(bw, bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv44b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv44ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(size=(2 * bw, 2 * bh), mode='bilinear'),
            nn.Conv2d(8 * bn, 4 * bn, kernel_size=1),
        )
        self.att3 = SELayer(channel=6 * bn, reduction=16)
        self.deconv3b = nn.Sequential(
            nn.Conv2d(6 * bn, 4 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * bn),
            nn.ReLU(True)
        )
        self.deconv33 = nn.Sequential(
            nn.Upsample(size=(2 * bw, 2 * bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv33b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv33ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(size=(4 * bw, 4 * bh), mode='bilinear'),
            nn.Conv2d(4 * bn, 2 * bn, kernel_size=1),
        )
        self.att2 = SELayer(channel=4 * bn, reduction=16)
        self.deconv2b = nn.Sequential(
            nn.Conv2d(4 * bn, 2 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * bn),
            nn.ReLU(True)
        )
        self.deconv22 = nn.Sequential(
            nn.Upsample(size=(4 * bw, 4 * bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv22b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv22ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.Upsample(size=(8 * bw, 8 * bh), mode='bilinear'),
            nn.Conv2d(2 * bn, bn, kernel_size=1),
        )
        self.att1 = SELayer(channel=3 * bn, reduction=16)
        self.deconv1b = nn.Sequential(
            nn.Conv2d(3 * bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv11 = nn.Sequential(
            nn.Upsample(size=(8 * bw, 8 * bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv11b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv11ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.ReLU(True)
        )
        
        # -------------- end -------------- #
        self.conv_end = nn.Conv2d(bn, 2, kernel_size=3, padding=1, bias=False)
        self.conv_end_label_edge = nn.Conv2d(bn, 2, kernel_size=3, padding=1, bias=False)
        self.conv_end_img_edge = nn.Conv2d(bn, 1, kernel_size=3, padding=1, bias=False)
        
        # Final upsample to match input size
        self.final_upsample = nn.Upsample(size=(data_size[0], data_size[1]), mode='bilinear')

    def forward(self, data, label=None):
        L1 = self.conv1(data)
        L1b = self.conv1b(L1)
        L1c = self.conv1c(data)
        L11 = self.conv11(data)
        L11b = self.conv11b(L11)
        L11a = self.conv11ada(L11b)
        L11ada = L11a + L11b
        L1b2 = torch.cat([L1b, L11ada], 1)

        L2 = self.conv2(L1b2)
        L2b = self.conv2b(L2)
        L2c = self.conv2c(data)
        L22 = self.conv22(L11b)
        L22b = self.conv22b(L22)
        L22a = self.conv22ada(L22b)
        L22ada = L22a + L22b
        L2b2 = torch.cat([L2b, L22ada], 1)

        L3 = self.conv3(L2b2)
        L3b = self.conv3b(L3)
        L3c = self.conv3c(data)
        L33 = self.conv33(L22b)
        L33b = self.conv33b(L33)
        L33a = self.conv33ada(L33b)
        L33ada = L33a + L33b
        L3b2 = torch.cat([L3b, L33ada], 1)

        L4 = self.conv4(L3b2)
        L4b = self.conv4b(L4)
        L4c = self.conv4c(data)
        L44 = self.conv44(L33b)
        L44b = self.conv44b(L44)
        L44a = self.conv44ada(L44b)
        L44ada = L44a + L44b
        L4b2 = torch.cat([L4b, L44a], 1)

        L5 = self.conv5(L4b2)
        L55 = self.conv55(L44b)
        L55a = self.conv55ada(L55)
        L55ada = L55a + L55
        L52 = torch.cat([L5, L55ada], 1)

        DL4 = self.deconv4(L52)
        DL4_add = DL4 + L4
        DL44 = self.deconv44(L55)
        DL44_add = DL44 + L44
        DL44b = self.deconv44b(DL44_add)
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
        DL22bada = DL22ba + DL22_add
        DL2_cat = torch.cat([DL2_add, L2c, DL22bada], 1)
        DL2_att = self.att2(DL2_cat)
        DL2b = self.deconv2b(DL2_att)

        DL1 = self.deconv1(DL2b)
        DL1_add = DL1 + L1
        DL11 = self.deconv11(DL22b)
        DL11_add = DL11 + L11
        DL11b = self.deconv11b(DL11_add)
        DL11ba = self.deconv11ada(DL11b)
        DL11bada = DL11ba + DL11b
        DL1_cat = torch.cat([DL1_add, L1c, DL11bada], 1)
        DL1_att = self.att1(DL1_cat)
        DL1b = self.deconv1b(DL1_att)

        end = self.conv_end(DL1b)
        end_label_edge = self.conv_end_label_edge(DL1b)
        end_img_edge = self.conv_end_img_edge(DL11b)
        
        # Apply final upsample
        end = self.final_upsample(end)
        end_label_edge = self.final_upsample(end_label_edge)
        end_img_edge = self.final_upsample(end_img_edge)

        output = torch.argmax(end, dim=1)
        output2 = torch.argmax(end_label_edge, dim=1)
        output3 = end_img_edge
        
        if not self.training:
            return output
        return end, end_label_edge, end_img_edge, output, output2, output3


# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def calculate_metrics(predictions, targets):
    """
    Calculate all metrics: IoU, F1, Precision, Recall, Accuracy
    """
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Calculate confusion matrix elements
    intersection = np.logical_and(pred_flat, target_flat).sum()
    union = np.logical_or(pred_flat, target_flat).sum()
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    # Precision, Recall, F1, Accuracy using sklearn
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(target_flat, pred_flat)
    
    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    all_metrics = {'iou': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (data, data_edge0, label0, label_edge) in enumerate(dataloader):
            data = data.to(device)
            label = label0.to(device).long()
            
            # Forward pass - model returns different things based on training mode
            outputs = model(data)
            
            # Handle the two possible return types
            if isinstance(outputs, tuple) and len(outputs) == 6:
                # Training mode output (during eval? shouldn't happen, but handle it)
                end1, end2, end3, predict1_, predict2_, predict3_ = outputs
                # Calculate loss using logits
                loss1 = F.cross_entropy(end1, label)
                loss = loss1.item()
            else:
                # Evaluation mode output - just the segmentation mask
                predict1_ = outputs
                # Can't calculate loss without logits, so set loss to 0
                loss = 0
            
            total_loss += loss
            num_batches += 1
            
            # Convert to numpy for metrics
            predictions = predict1_.cpu().detach().numpy().astype(np.uint8)
            targets = label0.numpy().astype(np.uint8)
            
            # Calculate metrics for each image in batch
            for i in range(predictions.shape[0]):
                metrics = calculate_metrics(predictions[i], targets[i])
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0
    
    return avg_metrics


# ============================================================================
# LOAD DATA FROM FOLDERS
# ============================================================================

def load_data_from_folders(images_folder, masks_folder, val_split=0.2, random_seed=42):
    """
    Load image and mask paths from folders and split into train/val
    
    Args:
        images_folder: path to folder containing images
        masks_folder: path to folder containing masks (same filenames as images)
        val_split: validation split ratio (default 0.2 = 20% for validation)
        random_seed: random seed for reproducibility
    
    Returns:
        train_image_paths, train_mask_paths, val_image_paths, val_mask_paths
    """
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    # Build full paths
    image_paths = [os.path.join(images_folder, f) for f in image_files]
    
    # Find corresponding masks (same filename, different extension)
    mask_paths = []
    valid_image_paths = []
    
    for img_path, img_file in zip(image_paths, image_files):
        # Get filename without extension
        name_without_ext = os.path.splitext(img_file)[0]
        
        # Try to find mask with same name (png, jpg, etc.)
        mask_found = None
        for mask_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            mask_candidate = os.path.join(masks_folder, name_without_ext +'_gt'+ mask_ext)
            if os.path.exists(mask_candidate):
                mask_found = mask_candidate
                break
        
        # Also try with '_mask' suffix
        if mask_found is None:
            for mask_ext in ['.png', '.jpg', '.jpeg']:
                mask_candidate = os.path.join(masks_folder, name_without_ext + '_mask' + mask_ext)
                if os.path.exists(mask_candidate):
                    mask_found = mask_candidate
                    break
        
        if mask_found is not None:
            mask_paths.append(mask_found)
            valid_image_paths.append(img_path)
        else:
            print(f"Warning: No mask found for image {img_file}, skipping...")
    
    print(f"Found {len(valid_image_paths)} valid image-mask pairs")
    
    # Split into train and validation
    train_img, val_img, train_mask, val_mask = train_test_split(
        valid_image_paths, mask_paths, 
        test_size=val_split, 
        random_state=random_seed,
        stratify=None
    )
    
    print(f"Train samples: {len(train_img)}")
    print(f"Validation samples: {len(val_img)}")
    
    return train_img, train_mask, val_img, val_mask


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    # ==================== CONFIGURATION ====================
    # Paths to your data folders
    images_folder = '/kaggle/input/datasets/vietanhaaaa/casiav2/image'
    masks_folder = '/kaggle/input/datasets/vietanhaaaa/casiav2/groundtruths2'
    val_split = 0.2
    random_seed = 42
    
    # Training hyperparameters
    lr = 0.00005
    epochs = 60
    batch_size = 8
    display_interval = 75
    steps = [150000, 40000000]
    
    # Model parameters
    input_size = (384, 384)
    
    # Save paths
    save_model_path = './snapshot_fullres_best'
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
    
    # CSV log file
    csv_path = './training_log.csv'
    
    # ==================== DEVICE ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ==================== LOAD DATA ====================
    print("\n" + "="*80)
    print("Loading data from folders...")
    print("="*80)
    
    # Check if folders exist
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    if not os.path.exists(masks_folder):
        raise FileNotFoundError(f"Masks folder not found: {masks_folder}")
    
    # Load and split data
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = load_data_from_folders(
        images_folder, masks_folder, val_split, random_seed
    )
    
    # Create datasets
    train_dataset = SegmentationDataset(train_img_paths, train_mask_paths, input_size)
    val_dataset = SegmentationDataset(val_img_paths, val_mask_paths, input_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True, 
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ==================== MODEL ====================
    model = Movenet(input_size).to(device)
    print("\nModel initialized with input/output size 384x384")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load pretrained weights if available
    pretrain_path = './all-50w.pkl'
    if os.path.exists(pretrain_path):
        print(f"\nLoading pretrained weights from {pretrain_path}")
        pretrain = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(pretrain.state_dict(), strict=False)
        print("Pretrained weights loaded (strict=False)")
    else:
        print("\nNo pretrained weights found, training from scratch")
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ==================== OPTIMIZER & SCHEDULER ====================
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.1, last_epoch=-1)
    
    # ==================== TRAINING LOOP ====================
    best_val_iou = 0.0
    best_epoch = 0
    
    # Initialize CSV log
    columns = ['epoch', 'train_loss', 'train_iou', 'train_f1', 'train_precision', 
               'train_recall', 'train_accuracy', 'val_loss', 'val_iou', 'val_f1', 
               'val_precision', 'val_recall', 'val_accuracy', 'lr']
    
    # Create new CSV file
    df_log = pd.DataFrame(columns=columns)
    df_log.to_csv(csv_path, index=False)
    print(f"\nCreated log file: {csv_path}")
    
    t0 = time.time()
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    for epoch in range(0, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_metrics = {'iou': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
        
        # Training loop over batches
        for batch_idx, (data, data_edge0, label0, label_edge) in enumerate(train_loader):
            # Move to device
            data = data.to(device)
            data_edge = data_edge0.to(device)
            label = label0.to(device).long()
            label_edge = label_edge.to(device).long()
            
            # Forward pass
            optimizer.zero_grad()
            end1, end2, end3, predict1_, predict2_, predict3_ = model(data)
            
            # Calculate loss
            loss1 = F.cross_entropy(end1, label)
            loss2 = F.cross_entropy(end2, label_edge)
            loss3 = F.mse_loss(end3, data_edge)
            
            w1, w2 = 0.2, 0.8
            loss = loss1 + w1 * loss2 + w2 * loss3
            
            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Calculate metrics for this batch
            predictions = predict1_.cpu().detach().numpy().astype(np.uint8)
            targets = label0.numpy().astype(np.uint8)
            
            for i in range(predictions.shape[0]):
                metrics = calculate_metrics(predictions[i], targets[i])
                for key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key])
            
            # Display progress within epoch
            if batch_idx % display_interval == 0:
                lr_current = optimizer.param_groups[0]['lr']
                avg_loss = np.mean(epoch_losses[-display_interval:])
                avg_iou = np.mean(epoch_metrics['iou'][-display_interval*batch_size:]) if epoch_metrics['iou'] else 0
                
                print(f'Epoch: {epoch:3d}/{epochs} | Batch: {batch_idx:4d}/{len(train_loader)} | '
                      f'LR: {lr_current:.6f} | Loss: {avg_loss:.4f} | Train IoU: {avg_iou:.4f} | '
                      f'Time: {time.time()-t0:.1f}s')
                t0 = time.time()
        
        # ==================== END OF EPOCH - VALIDATION ====================
        print(f"\n--- End of Epoch {epoch} ---")
        
        # Calculate average training metrics for the epoch
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_train_iou = np.mean(epoch_metrics['iou']) if epoch_metrics['iou'] else 0
        avg_train_f1 = np.mean(epoch_metrics['f1']) if epoch_metrics['f1'] else 0
        avg_train_precision = np.mean(epoch_metrics['precision']) if epoch_metrics['precision'] else 0
        avg_train_recall = np.mean(epoch_metrics['recall']) if epoch_metrics['recall'] else 0
        avg_train_accuracy = np.mean(epoch_metrics['accuracy']) if epoch_metrics['accuracy'] else 0
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        lr_current = optimizer.param_groups[0]['lr']
        
        # Print validation results
        print(f"Train - Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}, F1: {avg_train_f1:.4f}, "
              f"Precision: {avg_train_precision:.4f}, Recall: {avg_train_recall:.4f}, Acc: {avg_train_accuracy:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Save to CSV
        new_row = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_iou': avg_train_iou,
            'train_f1': avg_train_f1,
            'train_precision': avg_train_precision,
            'train_recall': avg_train_recall,
            'train_accuracy': avg_train_accuracy,
            'val_loss': val_metrics['loss'],
            'val_iou': val_metrics['iou'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_accuracy': val_metrics['accuracy'],
            'lr': lr_current
        }
        
        df_log = pd.read_csv(csv_path)
        df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
        df_log.to_csv(csv_path, index=False)
        
        # Save best model based on validation IoU
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch
            
            # Save best model
            if isinstance(model, nn.DataParallel):
                torch.save(model.module, f'{save_model_path}/best_model_iou.pkl')
                torch.save(model.module.state_dict(), f'{save_model_path}/best_model_state_dict.pkl')
            else:
                torch.save(model, f'{save_model_path}/best_model_iou.pkl')
                torch.save(model.state_dict(), f'{save_model_path}/best_model_state_dict.pkl')
            
            print(f"*** New best model saved! Val IoU: {best_val_iou:.4f} at epoch {epoch} ***")
        
        print(f"Best Val IoU so far: {best_val_iou:.4f} (epoch {best_epoch})")
        print("-" * 80 + "\n")
    
    # Final save
    final_path = f'{save_model_path}/final_model.pkl'
    if isinstance(model, nn.DataParallel):
        torch.save(model.module, final_path)
    else:
        torch.save(model, final_path)
    
    print("\n" + "="*80)
    print(f"Training Completed!")
    print(f"Best Validation IoU: {best_val_iou:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {save_model_path}")
    print(f"Training log saved to: {csv_path}")
    print("="*80)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()