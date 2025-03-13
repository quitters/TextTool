"""
Image Utility Functions
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Any


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio
    
    Args:
        image: OpenCV image as numpy array
        target_size: Target (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate target aspect ratio and image aspect ratio
    target_ratio = target_size[0] / target_size[1]
    image_ratio = w / h
    
    # Determine new dimensions
    if image_ratio > target_ratio:
        # Image is wider than target
        new_w = target_size[0]
        new_h = int(new_w / image_ratio)
    else:
        # Image is taller than target
        new_h = target_size[1]
        new_w = int(new_h * image_ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


def create_thumbnail(image_path: str, max_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
    """
    Create a thumbnail of an image
    
    Args:
        image_path: Path to the input image
        max_size: Maximum dimensions of the thumbnail (width, height)
        
    Returns:
        Thumbnail image as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize to create thumbnail
    return resize_image(image, max_size)


def convert_cv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV image to PIL Image
    
    Args:
        cv_image: OpenCV image (BGR format)
        
    Returns:
        PIL Image (RGB format)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Create PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image


def convert_pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to OpenCV image
    
    Args:
        pil_image: PIL Image (RGB format)
        
    Returns:
        OpenCV image (BGR format)
    """
    # Convert to numpy array
    rgb_image = np.array(pil_image)
    
    # Convert RGB to BGR
    cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    return cv_image


def calculate_image_quality(image_path: str) -> float:
    """
    Calculate an image quality score
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Quality score (0.0 to 1.0)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian) / 10000  # Normalize
    
    # Calculate contrast
    min_val, max_val = np.percentile(gray, [5, 95])
    contrast = (max_val - min_val) / 255
    
    # Combine metrics
    quality = (sharpness + contrast) / 2
    
    # Clamp to 0-1 range
    return max(0.0, min(1.0, quality))
