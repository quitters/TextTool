"""
Image Analysis Module for Intelligent Text Placement
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import slic
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TextRegion:
    """Represents a candidate region for text placement"""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    score: float  # Suitability score (0-1)
    rotation: float = 0.0  # Suggested rotation in degrees
    
    @property
    def coordinates(self) -> Tuple[int, int, int, int]:
        """Return coordinates as (x, y, width, height)"""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return the center point of the region"""
        return (self.x + self.width // 2, self.y + self.height // 2)


class ImageAnalyzer:
    """
    Analyzes images to identify optimal text placement regions
    using computer vision techniques.
    """
    
    def __init__(self):
        self.edge_weight = 0.3
        self.saliency_weight = 0.4
        self.contrast_weight = 0.3
        self.debug_callback = None
        
    def set_debug_callback(self, callback):
        """
        Set a callback function to receive debug messages
        
        Args:
            callback: Function that takes a string message as parameter
        """
        self.debug_callback = callback
        
    def log(self, message):
        """
        Log a debug message if callback is set
        
        Args:
            message: Debug message to log
        """
        if self.debug_callback:
            self.debug_callback(message)
        
    def fast_analyze(self, image_path: str, min_region_size: int = 100, progress_callback=None) -> List[TextRegion]:
        """
        Fast analysis of the image to find the largest solid areas for text placement
        using posterization and simplified processing with template detection.
        
        Args:
            image_path: Path to the input image
            min_region_size: Minimum size (pixels) for a candidate region
            progress_callback: Optional callback function to receive progress updates
            
        Returns:
            List of TextRegion objects representing candidate regions
        """
        import time
        import os
        start_time = time.time()
        
        # Send structured progress update
        self._send_progress("Starting fast analysis", "start", 0, start_time)
        self.log(f"Starting fast analysis of image: {os.path.basename(image_path)}")
        
        # Also send progress through the callback if provided
        if progress_callback:
            progress_callback({
                "message": "Starting image analysis",
                "percent": 0,
                "step": "analysis_start"
            })
        
        # Load image
        try:
            # Send progress update
            self._send_progress("Loading image", "load", 5, start_time)
            if progress_callback:
                progress_callback({
                    "message": "Loading image data",
                    "percent": 5,
                    "step": "load_image"
                })
                
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            # Get image dimensions
            original_height, original_width = image.shape[:2]
            self.log(f"Image dimensions: {original_width}x{original_height}")
            
            # Send progress update after loading
            if progress_callback:
                progress_callback({
                    "message": f"Image loaded ({original_width}x{original_height})",
                    "percent": 10,
                    "step": "preprocessing"
                })
        except Exception as e:
            self.log(f"Error loading image: {str(e)}")
            raise
        
        # Preprocess the image
        self._send_progress("Preprocessing image", "preprocess", 10, start_time)
        if progress_callback:
            progress_callback({
                "message": "Preprocessing image",
                "percent": 20,
                "step": "preprocessing"
            })
        
        # Store original dimensions for later reference
        # Resize image for faster processing if it's large
        resize_start = time.time()
        self._send_progress("Resizing image", "resize", 15, start_time)
        
        max_dimension = 600  # Reduced maximum dimension for even faster processing
        h, w = image.shape[:2]
        scale = 1.0
        
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_width = int(w * scale)
            new_height = int(h * scale)
            image = cv2.resize(image, (new_width, new_height))
            self.log(f"Resized image to {new_width}x{new_height} for faster processing")
        
        resize_time = time.time() - resize_start
        self.log(f"Image resizing completed in {resize_time:.2f} seconds")
        
        # First, try to detect if this is a template image with a clear frame/border
        template_start = time.time()
        self._send_progress("Checking for template patterns", "template_check", 20, start_time)
        
        # Convert to grayscale for template detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check if this is a template with a clear white/light center area
        template_regions = self._detect_template_regions(gray, image)
        
        if template_regions:
            self.log("Template pattern detected - using optimized processing")
            # Scale template regions back to original size
            for region in template_regions:
                if scale < 1.0:
                    region.x = int(region.x / scale)
                    region.y = int(region.y / scale)
                    region.width = int(region.width / scale)
                    region.height = int(region.height / scale)
            
            template_time = time.time() - template_start
            self.log(f"Template detection completed in {template_time:.2f} seconds")
            self._send_progress(f"Template analysis complete: {len(template_regions)} regions found", "complete", 100, start_time)
            
            total_time = time.time() - start_time
            self.log(f"Total fast analysis time: {total_time:.2f} seconds")
            
            # Final callback before returning
            if progress_callback:
                progress_callback({
                    "message": f"Analysis completed in {total_time:.2f}s",
                    "percent": 100,
                    "step": "complete"
                })
                
            return template_regions
        
        self.log("No clear template pattern detected - using standard analysis")
        template_time = time.time() - template_start
        self.log(f"Template check completed in {template_time:.2f} seconds")
        
        # Proceed with adaptive posterization
        posterize_start = time.time()
        self._send_progress("Adaptive posterizing", "posterize", 30, start_time)
        if progress_callback:
            progress_callback({
                "message": "Simplifying image colors",
                "percent": 30,
                "step": "analyzing"
            })
        
        # Determine image complexity to set appropriate posterization levels
        # More complex images need more levels to preserve important details
        std_dev = np.std(gray)
        levels = max(4, min(12, int(std_dev / 10)))  # Adaptive levels based on image complexity
        self.log(f"Image complexity: {std_dev:.2f}, using {levels} posterization levels")
        
        # Convert to LAB color space for better color segmentation
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(lab_image, 9, 75, 75)
        
        # Posterize by quantizing colors with adaptive levels
        posterized = self._posterize_image(filtered, levels=levels)
        
        # Convert back to BGR for display and further processing
        posterized_bgr = cv2.cvtColor(posterized, cv2.COLOR_LAB2BGR)
        
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(posterized_bgr, cv2.COLOR_BGR2GRAY)
        
        posterize_time = time.time() - posterize_start
        self.log(f"Adaptive posterization completed in {posterize_time:.2f} seconds")
        
        # Find contours in the posterized image
        contour_start = time.time()
        self._send_progress("Finding contours", "contours", 50, start_time)
        if progress_callback:
            progress_callback({
                "message": "Identifying potential text regions",
                "percent": 50,
                "step": "analyzing"
            })
        
        # Apply adaptive thresholding with improved parameters
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 3  # Adjusted parameters for better region detection
        )
        
        # Apply morphological operations to improve region detection
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        # Close small holes
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with hierarchy to better handle nested regions
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_time = time.time() - contour_start
        self.log(f"Enhanced contour detection completed in {contour_time:.2f} seconds")
        self.log(f"Found {len(contours)} contours in the processed image")
        
        # Generate regions from contours with improved scoring
        region_start = time.time()
        self._send_progress("Generating regions", "regions", 70, start_time)
        if progress_callback:
            progress_callback({
                "message": "Evaluating region quality",
                "percent": 70,
                "step": "analyzing"
            })
        
        regions = []
        min_area = min_region_size * min_region_size
        
        # Get image center for centrality scoring
        img_center_x = image.shape[1] / 2
        img_center_y = image.shape[0] / 2
        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
        
        # Process each contour to create regions with improved scoring
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < min_area:
                continue
            
            # Get bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Calculate enhanced score based on multiple factors
            # 1. Area relative to image size
            area_score = area / (image.shape[0] * image.shape[1])
            
            # 2. Solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # 3. Centrality (how close to image center)
            region_center_x = x + w_rect/2
            region_center_y = y + h_rect/2
            distance_to_center = np.sqrt((region_center_x - img_center_x)**2 + 
                                         (region_center_y - img_center_y)**2)
            centrality = 1.0 - (distance_to_center / max_distance)
            
            # 4. Aspect ratio suitability (prefer wider regions for text)
            aspect = w_rect / h_rect if h_rect > 0 else 0
            aspect_score = min(1.0, aspect / 3.0) if aspect < 3.0 else min(1.0, 5.0 / aspect)
            
            # 5. Color uniformity (check variance within region)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            non_zero = roi[roi != 0]
            uniformity = 1.0 - min(1.0, np.std(non_zero) / 128.0) if len(non_zero) > 0 else 0
            
            # Combine scores with appropriate weights
            score = (0.35 * area_score + 
                     0.20 * solidity + 
                     0.20 * centrality + 
                     0.10 * aspect_score + 
                     0.15 * uniformity)
            
            # Create the text region, adjusting coordinates for the original image size
            region = TextRegion(
                x=int(x / scale) if scale < 1.0 else x,
                y=int(y / scale) if scale < 1.0 else y,
                width=int(w_rect / scale) if scale < 1.0 else w_rect,
                height=int(h_rect / scale) if scale < 1.0 else h_rect,
                score=score
            )
            
            regions.append(region)
            
            # Log progress less frequently to reduce overhead
            if i % 20 == 0 and self.debug_callback:
                self._send_progress(f"Processing region {i+1}/{len(contours)}", "region_progress", 
                                   70 + (i / len(contours)) * 20, start_time)
        
        region_time = time.time() - region_start
        self.log(f"Enhanced region generation completed in {region_time:.2f} seconds")
        self.log(f"Generated {len(regions)} candidate regions")
        
        # Sort regions by score (descending) and apply refinement
        sort_start = time.time()
        self._send_progress("Sorting and refining regions", "sort", 95, start_time)
        if progress_callback:
            progress_callback({
                "message": "Finalizing region selection",
                "percent": 90,
                "step": "finalizing"
            })
        
        regions.sort(key=lambda r: r.score, reverse=True)
        
        # Apply edge refinement to top regions
        refined_regions = self._refine_region_edges(regions[:15], original_width, original_height, scale)
        
        # Limit to top regions after refinement
        regions = refined_regions[:10]
        
        sort_time = time.time() - sort_start
        self.log(f"Region sorting and refinement completed in {sort_time:.2f} seconds")
        
        total_time = time.time() - start_time
        self._send_progress(f"Fast analysis complete: {len(regions)} regions found", "complete", 100, start_time)
        
        # Final callback before returning
        if progress_callback:
            progress_callback({
                "message": f"Found {len(regions)} suitable text regions",
                "percent": 100,
                "step": "complete"
            })
            
        self.log(f"Total fast analysis time: {total_time:.2f} seconds")
        self.log(f"Found {len(regions)} candidate regions for text placement")
        
        return regions
        
    def _detect_template_regions(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[TextRegion]:
        """
        Detect if the image is a template with clear regions for text placement
        
        Args:
            gray_image: Grayscale version of the image
            color_image: Color version of the image
            
        Returns:
            List of TextRegion objects if template is detected, empty list otherwise
        """
        # Check for billboard/frame template (Image 1 type)
        # Look for rectangular white/light areas surrounded by darker frame
        
        # Threshold to find light areas
        _, light_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of light areas
        light_contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if we have large light areas that could be billboards/frames
        template_regions = []
        img_area = gray_image.shape[0] * gray_image.shape[1]
        
        for contour in light_contours:
            area = cv2.contourArea(contour)
            # Check if area is significant (at least 10% of image)
            if area > 0.1 * img_area:
                # Check if shape is approximately rectangular
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # If it has 4 vertices, it's likely a rectangle
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's not just the entire image
                    if x > 5 and y > 5 and x + w < gray_image.shape[1] - 5 and y + h < gray_image.shape[0] - 5:
                        # Calculate score based on size and centrality
                        center_x = x + w/2
                        center_y = y + h/2
                        img_center_x = gray_image.shape[1] / 2
                        img_center_y = gray_image.shape[0] / 2
                        
                        # Distance from center (normalized)
                        distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
                        centrality = 1.0 - (distance / max_distance)
                        
                        # Score combines size and centrality
                        score = 0.7 * (area / img_area) + 0.3 * centrality
                        
                        region = TextRegion(x=x, y=y, width=w, height=h, score=score)
                        template_regions.append(region)
        
        # Check for floral/decorative frame template (Image 2 type)
        if not template_regions:
            # Use color information to detect decorative frames
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # Look for central white/light area
            center_x = gray_image.shape[1] // 2
            center_y = gray_image.shape[0] // 2
            
            # Check center region brightness
            center_region = gray_image[center_y-20:center_y+20, center_x-20:center_x+20]
            if np.mean(center_region) > 200:  # Center is bright
                # Check for color variation around edges (decorative elements)
                edge_regions = [
                    hsv[0:20, :],                    # Top
                    hsv[-20:, :],                   # Bottom
                    hsv[:, 0:20],                   # Left
                    hsv[:, -20:]                    # Right
                ]
                
                edge_std = np.mean([np.std(region[:,:,0]) for region in edge_regions])
                
                # If edges have color variation (decorative elements)
                if edge_std > 20:
                    # Find the central white area
                    _, center_mask = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
                    
                    # Erode to remove small white spots
                    kernel = np.ones((5,5), np.uint8)
                    eroded = cv2.erode(center_mask, kernel, iterations=1)
                    
                    # Find contours of the central area
                    center_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if center_contours:
                        # Get the largest contour
                        largest_contour = max(center_contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Create a region with high score
                        score = 0.95  # High score for template-matched regions
                        region = TextRegion(x=x, y=y, width=w, height=h, score=score)
                        template_regions.append(region)
        
        return template_regions
        
    def _refine_region_edges(self, regions: List[TextRegion], original_width: int, 
                             original_height: int, scale: float) -> List[TextRegion]:
        """
        Refine the edges of detected regions for more precise boundaries
        
        Args:
            regions: List of detected regions to refine
            original_width: Width of the original image
            original_height: Height of the original image
            scale: Scale factor used for resizing
            
        Returns:
            List of refined regions
        """
        refined = []
        
        for region in regions:
            # Create a copy to avoid modifying the original
            refined_region = TextRegion(
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                score=region.score,
                rotation=region.rotation
            )
            
            # Ensure region doesn't extend beyond image boundaries
            if refined_region.x < 0:
                refined_region.width += refined_region.x
                refined_region.x = 0
                
            if refined_region.y < 0:
                refined_region.height += refined_region.y
                refined_region.y = 0
                
            if refined_region.x + refined_region.width > original_width:
                refined_region.width = original_width - refined_region.x
                
            if refined_region.y + refined_region.height > original_height:
                refined_region.height = original_height - refined_region.y
            
            # Apply small margin for aesthetics (5% of region size)
            margin_x = int(refined_region.width * 0.05)
            margin_y = int(refined_region.height * 0.05)
            
            refined_region.x += margin_x
            refined_region.y += margin_y
            refined_region.width -= 2 * margin_x
            refined_region.height -= 2 * margin_y
            
            # Ensure minimum dimensions
            if refined_region.width < 20:
                refined_region.width = 20
            if refined_region.height < 20:
                refined_region.height = 20
                
            refined.append(refined_region)
            
        return refined
        
    def _posterize_image(self, image: np.ndarray, levels: int = 8) -> np.ndarray:
        """
        Posterize an image by reducing the number of colors
        
        Args:
            image: Input image
            levels: Number of levels per channel
            
        Returns:
            Posterized image
        """
        # Create a copy of the image
        posterized = image.copy()
        
        # Calculate the bin size for each level
        bin_size = 256 // levels
        
        # Apply posterization to each channel
        for i in range(image.shape[2]):
            channel = posterized[:, :, i]
            # Integer division to quantize values
            channel = (channel // bin_size) * bin_size
            posterized[:, :, i] = channel
            
        return posterized
        
    def analyze(self, image_path: str, min_region_size: int = 100) -> List[TextRegion]:
        """
        Analyze the image and return candidate regions for text placement.
        Now uses the fast_analyze method by default for better performance.
        
        Args:
            image_path: Path to the input image
            min_region_size: Minimum size (pixels) for a candidate region
            
        Returns:
            List of TextRegion objects representing candidate regions
        """
        # Use the fast analysis method for better performance
        return self.fast_analyze(image_path, min_region_size)
        
    def _send_progress(self, message, step, percent, start_time):
        """
        Send a structured progress update
        
        Args:
            message: Description of the current step
            step: Identifier for the current step
            percent: Percentage of completion (0-100)
            start_time: Time when the analysis started
        """
        if self.debug_callback:
            import time
            elapsed = time.time() - start_time
            progress = {
                "status": "progress",
                "message": f"{message}... ({elapsed:.1f}s elapsed)",
                "step": step,
                "percent": percent,
                "elapsed": elapsed
            }
            self.debug_callback(progress)
    
    def _detect_edges(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image
        
        Returns:
            Edge map where high values represent strong edges
        """
        # Use Sobel filter for edge detection
        edges = sobel(image_gray)
        
        # Normalize to 0-1 range
        edges = (edges - np.min(edges)) / (np.max(edges) - np.min(edges))
        
        return edges
    
    def _generate_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a saliency map where high values indicate visually important areas
        
        Returns:
            Saliency map normalized to 0-1 range
        """
        # Use a faster approximation of saliency
        # Convert to grayscale if not already
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize to a smaller size for faster processing
        # For a 512x512 image, this will be 128x128
        height, width = gray.shape[:2]
        small_gray = cv2.resize(gray, (width//4, height//4))
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(small_gray, (9, 9), 0)
        
        # Calculate the difference of Gaussians (DoG)
        # This is a simple but effective approximation of saliency
        dog = cv2.GaussianBlur(small_gray, (5, 5), 0) - blur
        
        # Resize back to original size
        saliency_map = cv2.resize(dog, (width, height))
        
        # Normalize to 0-1 range
        if np.max(saliency_map) > np.min(saliency_map):
            saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
        else:
            saliency_map = np.zeros_like(saliency_map, dtype=np.float32)
            
        return saliency_map
    
    def _calculate_contrast(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Calculate local contrast in the image
        
        Returns:
            Contrast map where high values indicate high contrast areas
        """
        # Use a simpler and faster method for contrast calculation
        # Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        
        # Calculate absolute difference between original and blurred image
        # This highlights areas with high local contrast
        diff = cv2.absdiff(image_gray, blurred)
        
        # Normalize to 0-1 range
        if np.max(diff) > np.min(diff):
            contrast_map = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
        else:
            contrast_map = np.zeros_like(diff)
            
        return contrast_map
    
    def _generate_regions(self, score_map: np.ndarray, image_shape: Tuple[int, int, int], 
                          min_region_size: int) -> List[TextRegion]:
        """
        Generate candidate regions based on the score map
        
        Args:
            score_map: Combined score map where high values indicate suitable areas
            image_shape: Shape of the original image
            min_region_size: Minimum size for a candidate region
            
        Returns:
            List of TextRegion objects
        """
        regions = []
        h, w = score_map.shape[:2]
        
        # Skip SLIC which is computationally expensive
        # Instead, use simple thresholding to find good regions
        
        # Convert score map to 8-bit for thresholding
        score_8bit = (score_map * 255).astype(np.uint8)
        
        # Apply threshold to find good areas (top 25% of scores)
        threshold_value = np.percentile(score_8bit, 75)
        _, thresh = cv2.threshold(score_8bit, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < min_region_size * min_region_size / 4:
                continue
                
            # Get bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Ensure minimum size
            width = max(w_rect, min_region_size)
            height = max(h_rect, min_region_size)
            
            # Calculate score for this region
            region_mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], 0, 1, -1)
            avg_score = np.mean(score_map[region_mask > 0])
            
            # Create the text region
            region = TextRegion(
                x=x,
                y=y,
                width=width,
                height=height,
                score=float(avg_score)
            )
            
            regions.append(region)
        
        # Always add rule-of-thirds regions for reliable placement options
        third_w, third_h = w // 3, h // 3
        
        # Add the four rule-of-thirds intersection points with regions around them
        for x, y in [(third_w, third_h), (2*third_w, third_h), 
                     (third_w, 2*third_h), (2*third_w, 2*third_h)]:
            region_size = min(w // 4, h // 4)
            
            # Create region centered at the intersection point
            x_start = max(0, x - region_size // 2)
            y_start = max(0, y - region_size // 2)
            
            # Ensure the region stays within the image
            width = min(region_size, w - x_start)
            height = min(region_size, h - y_start)
            
            # Calculate the score for this region
            region_score = np.mean(score_map[y_start:y_start+height, x_start:x_start+width])
            
            region = TextRegion(
                x=x_start,
                y=y_start,
                width=width,
                height=height,
                score=float(region_score) * 1.2  # Boost score for rule-of-thirds regions
            )
            
            regions.append(region)
        
        return regions
    
    def _suggest_rotations(self, regions: List[TextRegion], edge_map: np.ndarray) -> List[TextRegion]:
        """
        Suggest rotation angles for text regions based on nearby edges
        
        Args:
            regions: List of candidate regions
            edge_map: Edge detection map
            
        Returns:
            Updated list of regions with rotation suggestions
        """
        # Limit the number of regions to analyze for rotation to improve performance
        # Only process the top scoring regions
        top_regions = sorted(regions, key=lambda r: r.score, reverse=True)[:min(len(regions), 5)]
        
        # For each region, analyze the nearby edges to suggest text rotation
        for region in top_regions:
            # Extract the region from the edge map
            x, y, w, h = region.coordinates
            
            # If the region is too small, skip rotation analysis
            if w < 20 or h < 20:
                continue
                
            # Downsample the region for faster processing if it's large
            scale_factor = 1.0
            if w > 100 or h > 100:
                scale_factor = 100 / max(w, h)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                region_edges = cv2.resize(edge_map[y:y+h, x:x+w], (new_w, new_h))
            else:
                region_edges = edge_map[y:y+h, x:x+w]
            
            # If no edges, skip rotation analysis
            if np.max(region_edges) < 0.1:
                continue
                
            # Use a faster approach with fewer angle bins
            region_edges_uint8 = (region_edges * 255).astype(np.uint8)
            lines = cv2.HoughLines(region_edges_uint8, 1, np.pi/18, threshold=max(min(w, h)//4, 10))
            
            if lines is not None and len(lines) > 0:
                # Calculate the dominant angle from a limited number of lines
                angles = []
                for line in lines[:min(len(lines), 10)]:
                    rho, theta = line[0]
                    angle_deg = np.degrees(theta) % 180
                    # Convert to -90 to 90 range
                    if angle_deg > 90:
                        angle_deg -= 180
                    angles.append(angle_deg)
                
                # Use the median angle as the dominant direction
                if angles:
                    dominant_angle = np.median(angles)
                    # Only suggest non-trivial rotations
                    if abs(dominant_angle) > 5 and abs(dominant_angle) < 45:
                        region.rotation = float(dominant_angle)
        
        return regions
