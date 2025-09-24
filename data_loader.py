import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import cv2
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import WeightedRandomSampler
import glob
from typing import List, Tuple, Optional, Dict
import random

class LungSegmentationDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transform: Optional[transforms.Compose] = None, 
                 augment: bool = False, apply_clahe: bool = True):
        """
        Dataset for lung segmentation
        
        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files
            transform: Optional transform to be applied
            augment: Whether to apply data augmentation
            apply_clahe: Whether to apply CLAHE filter
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        self.apply_clahe = apply_clahe
        
        # Validate that we have matching images and masks
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')
        
        # Convert to numpy arrays for processing
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Resize COVID-19_Radiography_Dataset images from 299×299 to 256×256
        if image_np.shape != (256, 256):
            image_pil = Image.fromarray(image_np)
            image_pil = image_pil.resize((256, 256), Image.BILINEAR)
            image_np = np.array(image_pil)
            
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.resize((256, 256), Image.NEAREST)
            mask_np = np.array(mask_pil)
        
        # Apply CLAHE if enabled - create CLAHE object locally to avoid pickling issues
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_np = clahe.apply(image_np)
        
        # Convert back to PIL for torchvision transforms
        image = Image.fromarray(image_np)
        mask = Image.fromarray(mask_np)
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            # Default convert to tensor
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
        
        # Data augmentation
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask
    
    def _apply_augmentation(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations to image and mask"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        
        # Random rotation (-10 to 10 degrees)
        angle = random.uniform(-10, 10)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        
        # Add Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)
        
        return image, mask

def get_all_image_mask_pairs(base_dir: str = '.') -> Tuple[List[str], List[str]]:
    """
    Find all image-mask pairs across all datasets
    
    Returns:
        Tuple of (image_paths, mask_paths)
    """
    image_paths = []
    mask_paths = []
    
    # COVID-19_Radiography_Dataset
    covid_dirs = [
        'COVID-19_Radiography_Dataset/COVID',
        'COVID-19_Radiography_Dataset/Lung_Opacity', 
        'COVID-19_Radiography_Dataset/Normal',
        'COVID-19_Radiography_Dataset/Viral Pneumonia'
    ]
    
    for dir_path in covid_dirs:
        image_dir = os.path.join(base_dir, dir_path, 'images')
        mask_dir = os.path.join(base_dir, dir_path, 'masks')
        
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
            
            image_paths.extend(images)
            mask_paths.extend(masks)
    
    # Infection Segmentation Data - only use lung masks, ignore infection masks
    infection_splits = ['Train', 'Val', 'Test']
    infection_classes = ['COVID-19', 'Non-COVID', 'Normal']
    
    for split in infection_splits:
        for cls in infection_classes:
            image_dir = os.path.join(base_dir, 'Infection Segmentation Data', 'Infection Segmentation Data', 
                                   split, cls, 'images')
            mask_dir = os.path.join(base_dir, 'Infection Segmentation Data', 'Infection Segmentation Data', 
                                  split, cls, 'lung masks')
            
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
                
                image_paths.extend(images)
                mask_paths.extend(masks)
    
    # Lung Segmentation Data
    lung_splits = ['Train', 'Val', 'Test']
    lung_classes = ['COVID-19', 'Non-COVID', 'Normal']
    
    for split in lung_splits:
        for cls in lung_classes:
            image_dir = os.path.join(base_dir, 'Lung Segmentation Data', 'Lung Segmentation Data', 
                                   split, cls, 'images')
            mask_dir = os.path.join(base_dir, 'Lung Segmentation Data', 'Lung Segmentation Data', 
                                  split, cls, 'lung masks')
            
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
                
                image_paths.extend(images)
                mask_paths.extend(masks)
    
    return image_paths, mask_paths

def get_dataset_statistics(image_paths: List[str], mask_paths: List[str]) -> Dict:
    """Get statistics about the dataset"""
    stats = {
        'total_images': len(image_paths),
        'dataset_breakdown': {},
        'size_breakdown': {}
    }
    
    # Count by dataset source and size
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Determine dataset source
        if 'COVID-19_Radiography_Dataset' in img_path:
            dataset_key = 'COVID-19_Radiography'
        elif 'Infection Segmentation Data' in img_path:
            dataset_key = 'Infection_Segmentation'
        elif 'Lung Segmentation Data' in img_path:
            dataset_key = 'Lung_Segmentation'
        else:
            dataset_key = 'Unknown'
        
        stats['dataset_breakdown'][dataset_key] = stats['dataset_breakdown'].get(dataset_key, 0) + 1
        
        # Check image and mask sizes
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        img_size = img.size
        mask_size = mask.size
        
        size_key = f"img_{img_size}_mask_{mask_size}"
        stats['size_breakdown'][size_key] = stats['size_breakdown'].get(size_key, 0) + 1
    
    return stats

def get_loaders(
    data_dir: str = '.',
    batch_size: int = 16,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    apply_clahe: bool = True,
    augment_train: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders for lung segmentation
    
    Args:
        data_dir: Root directory containing the datasets
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of workers for data loading
        use_weighted_sampler: Whether to use weighted sampler for class imbalance
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
        persistent_workers: Whether to persist workers
        apply_clahe: Whether to apply CLAHE filter
        augment_train: Whether to augment training data
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get all image-mask pairs
    image_paths, mask_paths = get_all_image_mask_pairs(data_dir)
    
    # Get dataset statistics
    stats = get_dataset_statistics(image_paths, mask_paths)
    print(f"Dataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print("\nDataset Breakdown:")
    for dataset, count in stats['dataset_breakdown'].items():
        print(f"  {dataset}: {count} images")
    
    print("\nSize Breakdown:")
    for size, count in stats['size_breakdown'].items():
        print(f"  {size}: {count} images")
    
    # Create full dataset
    full_dataset = LungSegmentationDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        augment=False,  # We'll handle augmentation in the train loader
        apply_clahe=apply_clahe
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Enable augmentation for training dataset
    # We need to access the underlying dataset through the subset
    train_dataset.dataset.augment = augment_train
    val_dataset.dataset.augment = False
    test_dataset.dataset.augment = False
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader, test_loader

# For testing and demonstration
if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader, test_loader = get_loaders(
        data_dir='.',
        batch_size=4,
        num_workers=0
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get one batch to check
    for images, masks in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Masks unique values: {torch.unique(masks)}")
        break