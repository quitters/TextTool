"""
Region Proposal Module for Intelligent Text Placement
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from .image_analyzer import TextRegion


class RegionProposer:
    """
    Generates and refines candidate regions for text placement
    based on design principles and image content.
    """
    
    def __init__(self):
        # Configuration parameters
        self.min_aspect_ratio = 1.2  # Minimum width-to-height ratio for horizontal text
        self.max_aspect_ratio = 5.0  # Maximum width-to-height ratio
        self.margin_factor = 0.1     # Margin as a fraction of image dimensions
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
        
    def fast_propose_regions(self, image_path: str, 
                           regions: List[TextRegion], 
                           text_length: int = 20) -> List[Dict[str, Any]]:
        """
        Fast version of region proposal that focuses only on finding the best solid areas
        for text placement without complex styling calculations.
        Optimized for template detection and improved performance.
        
        Args:
            image_path: Path to the input image
            regions: List of candidate TextRegion objects
            text_length: Approximate length of text to be placed (in characters)
            
        Returns:
            List of dictionaries containing region proposals with basic styling suggestions
        """
        import time
        import os
        start_time = time.time()
        
        # Check if we have any regions at all
        if not regions:
            self.log({"status": "warning", "message": "No regions provided for proposal generation", "step": "fast_proposal_empty"})
            return []
            
        # Check if this is likely a template image (high-scoring regions)
        is_template = any(region.score > 0.9 for region in regions[:3])
        if is_template:
            self.log(f"Template image detected - using optimized template processing")
        
        self.log({"status": "progress", "message": "Starting fast region proposal", "step": "fast_proposal_start", "percent": 70})
        self.log(f"Processing {len(regions)} candidate regions for text placement")
        
        # Load image to get dimensions - with early return on failure to avoid hanging
        load_start = time.time()
        self.log({"status": "progress", "message": "Loading image for region proposal", "step": "proposal_load", "percent": 75})
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.log({"status": "error", "message": f"Could not load image from {image_path}", "step": "proposal_load_error"})
                return []
                
            img_height, img_width = image.shape[:2]
            load_time = time.time() - load_start
            self.log(f"Image loaded in {load_time:.2f} seconds")
        except Exception as e:
            self.log({"status": "error", "message": f"Error loading image: {str(e)}", "step": "proposal_load_error"})
            return []
        
        # Determine how many regions to process based on template detection
        max_regions = 10 if is_template else 5
        early_stop_count = 5 if is_template else 3
        
        # Skip complex refinement and just do basic adjustments for text length
        adjust_start = time.time()
        self.log({"status": "progress", "message": "Analyzing regions for text placement", "step": "proposal_adjust", "percent": 80})
        
        # Convert to multiple color spaces for better analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        proposals = []
        min_margin_x = int(img_width * 0.05)  # 5% of image width
        min_margin_y = int(img_height * 0.05)  # 5% of image height
        
        # Process each region directly with early stopping for non-templates
        for idx, region in enumerate(regions):
            # Skip processing if we already have enough good regions
            if not is_template and len(proposals) >= early_stop_count and idx > early_stop_count:
                self.log(f"Already found {len(proposals)} good regions, stopping early")
                break
                
            # Stop after processing max_regions
            if idx >= max_regions:
                self.log(f"Reached maximum number of regions to process ({max_regions})")
                break
                
            # Basic adjustment for text length
            adjusted = self._adjust_region_size(region, img_width, img_height, text_length, min_margin_x, min_margin_y)
            
            # Quick check if region is suitable
            if adjusted.width < text_length * 8:  # Rough estimate of minimum width needed
                continue
                
            # Extract region from image for enhanced color analysis
            try:
                # Use try-except with boundary checks to avoid index errors
                y1 = max(0, min(adjusted.y, img_height-1))
                y2 = max(0, min(adjusted.y + adjusted.height, img_height))
                x1 = max(0, min(adjusted.x, img_width-1))
                x2 = max(0, min(adjusted.x + adjusted.width, img_width))
                
                # Skip if region is invalid after boundary adjustment
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                roi = image[y1:y2, x1:x2]
                roi_gray = gray[y1:y2, x1:x2]
                roi_hsv = hsv[y1:y2, x1:x2]
                
                if roi.size == 0:  # Skip empty regions
                    continue
                    
                # Enhanced color analysis
                avg_luminance = np.mean(roi_gray)
                color_std = np.std(roi_gray)  # Check color uniformity
                is_uniform = color_std < 30  # Lower threshold for template images
                
                # Determine text color with improved contrast
                if avg_luminance > 180:  # Very light background
                    text_color = "#000000"  # Black text
                elif avg_luminance < 60:  # Very dark background
                    text_color = "#FFFFFF"  # White text
                else:  # Medium luminance - use higher contrast
                    # Choose black or white based on which gives better contrast
                    text_color = "#000000" if avg_luminance > 128 else "#FFFFFF"
                
                # Calculate font size based on region dimensions and text length
                font_size = self._suggest_font_size(adjusted.width, text_length)
                
                # For template images, adjust position for better aesthetics
                if is_template and is_uniform:
                    # Center text horizontally
                    center_x = adjusted.x + adjusted.width // 2
                    # Adjust vertical position based on region height
                    if adjusted.height > 100:
                        # For tall regions, place text in upper third
                        adjusted.y = adjusted.y + adjusted.height // 4
                        adjusted.height = adjusted.height // 2
                    
                    # Increase font size for templates with uniform backgrounds
                    font_size = int(font_size * 1.2)
                
                # Create proposal with enhanced styling
                proposal = {
                    "id": idx + 1,
                    "region": {
                        "x": adjusted.x,
                        "y": adjusted.y,
                        "width": adjusted.width,
                        "height": adjusted.height,
                        "rotation": adjusted.rotation
                    },
                    "score": adjusted.score,
                    "styling": {
                        "color": text_color,
                        "suggested_font_size": font_size,
                        "is_uniform": is_uniform,
                        "is_template": is_template
                    }
                }
                
                proposals.append(proposal)
                
                # Log progress less frequently to reduce overhead
                if idx % 3 == 0:
                    self.log({"status": "progress", "message": f"Processed {idx+1}/{min(len(regions), max_regions)} regions", 
                              "step": "proposal_progress", "percent": 80 + (idx / min(len(regions), max_regions)) * 10})
                
            except Exception as e:
                self.log(f"Error processing region {idx}: {str(e)}")
                continue
                
        adjust_time = time.time() - adjust_start
        self.log(f"Region analysis completed in {adjust_time:.2f} seconds")
        self.log(f"Generated {len(proposals)} region proposals")
        
        # Sort by score and limit to top proposals
        proposals.sort(key=lambda p: p["score"], reverse=True)
        proposals = proposals[:5]  # Limit to top 5 proposals
        
        total_time = time.time() - start_time
        self.log({"status": "progress", "message": f"Fast proposal generation complete: {len(proposals)} proposals", "step": "proposal_complete", "percent": 95})
        self.log(f"Total proposal generation time: {total_time:.2f} seconds")
        
        return proposals
        
    def _suggest_font_size(self, region_width: int, text_length: int) -> int:
        """
        Suggest an appropriate font size based on region width and text length
        
        Args:
            region_width: Width of the region in pixels
            text_length: Approximate length of text in characters
            
        Returns:
            Suggested font size in pixels
        """
        if text_length <= 0:
            text_length = 10  # Default if no text length provided
            
        # Estimate characters per line based on region width
        # Assume average character width is about 0.6 times the font size
        char_width_ratio = 0.6
        
        # Calculate font size that would allow the text to fit the region width
        # with some margin (80% of region width)
        available_width = region_width * 0.8
        font_size = available_width / (text_length * char_width_ratio)
        
        # Cap font size to reasonable limits
        font_size = max(12, min(72, font_size))
        
        return int(font_size)
        
    def propose_regions(self, image_path: str, 
                        regions: List[TextRegion], 
                        text_length: int = 20) -> List[Dict[str, Any]]:
        """
        Refine and propose optimal regions for text placement.
        Now uses the fast_propose_regions method by default for better performance.
        
        Args:
            image_path: Path to the input image
            regions: List of candidate TextRegion objects
            text_length: Approximate length of text to be placed (in characters)
            
        Returns:
            List of dictionaries containing region proposals with styling suggestions
        """
        # Use the fast proposal method for better performance
        return self.fast_propose_regions(image_path, regions, text_length)
        
    def _legacy_propose_regions(self, image_path: str, 
                        regions: List[TextRegion], 
                        text_length: int = 20) -> List[Dict[str, Any]]:
        """
        Original implementation of region proposal with full styling calculations.
        Kept for reference and comparison.
        
        Args:
            image_path: Path to the input image
            regions: List of candidate TextRegion objects
            text_length: Approximate length of text to be placed (in characters)
            
        Returns:
            List of dictionaries containing region proposals with styling suggestions
        """
        import time
        import os
        start_time = time.time()
        
        self.log({"status": "progress", "message": "Starting region proposal generation", "step": "proposal_start", "percent": 70})
        self.log(f"Processing {len(regions)} candidate regions for text placement")
        
        # Load image to get dimensions
        load_start = time.time()
        self.log({"status": "progress", "message": "Loading image for region proposal", "step": "proposal_load", "percent": 75})
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            img_height, img_width = image.shape[:2]
            load_time = time.time() - load_start
            self.log(f"Image loaded in {load_time:.2f} seconds")
        except Exception as e:
            self.log(f"Error loading image: {str(e)}")
            raise
        
        # Apply design heuristics to refine regions
        refine_start = time.time()
        self.log({"status": "progress", "message": "Refining regions based on design principles", "step": "proposal_refine", "percent": 80})
        
        refined_regions = self._refine_regions(regions, img_width, img_height, text_length)
        refine_time = time.time() - refine_start
        self.log(f"Region refinement completed in {refine_time:.2f} seconds")
        self.log(f"Refined to {len(refined_regions)} suitable regions")
        
        # Generate detailed proposals with styling suggestions
        styling_start = time.time()
        self.log({"status": "progress", "message": "Generating styling suggestions", "step": "proposal_style", "percent": 90})
        
        proposals = []
        
        for idx, region in enumerate(refined_regions):
            # Calculate region luminance to suggest text color
            roi = image[region.y:region.y+region.height, region.x:region.x+region.width]
            avg_luminance = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            
            # Determine text color based on background luminance
            text_color = "dark" if avg_luminance > 128 else "light"
            
            # Determine if the region has strong edges
            edge_intensity = self._calculate_edge_intensity(roi)
            has_strong_edges = edge_intensity > 0.2
            
            # Create proposal with styling suggestions
            proposal = {
                "id": idx,
                "region": {
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                    "rotation": region.rotation
                },
                "score": region.score,
                "styling": {
                    "color": text_color,
                    "outline": has_strong_edges,
                    "shadow": avg_luminance < 200 and avg_luminance > 50,
                    "suggested_font_size": self._suggest_font_size(region.width, text_length)
                }
            }
            
            proposals.append(proposal)
        
        styling_time = time.time() - styling_start
        self.log(f"Styling suggestion completed in {styling_time:.2f} seconds")
        
        # Sort proposals by score (descending)
        sort_start = time.time()
        self.log({"status": "progress", "message": "Sorting and finalizing proposals", "step": "proposal_sort", "percent": 95})
        
        proposals.sort(key=lambda p: p["score"], reverse=True)
        sort_time = time.time() - sort_start
        self.log(f"Proposal sorting completed in {sort_time:.2f} seconds")
        
        total_time = time.time() - start_time
        self.log({"status": "progress", "message": f"Region proposal complete: {len(proposals)} proposals generated", "step": "proposal_complete", "percent": 98})
        self.log(f"Total proposal generation time: {total_time:.2f} seconds")
        
        return proposals
    
    def _refine_regions(self, regions: List[TextRegion], img_width: int, img_height: int, 
                       text_length: int) -> List[TextRegion]:
        """
        Refine candidate regions based on design principles
        
        Args:
            regions: List of candidate TextRegion objects
            img_width: Width of the image
            img_height: Height of the image
            text_length: Approximate length of text to be placed (in characters)
            
        Returns:
            List of refined TextRegion objects
        """
        import time
        start_time = time.time()
        
        self.log(f"Refining {len(regions)} candidate regions")
        
        refined_regions = []
        min_margin_x = int(img_width * self.margin_factor)
        min_margin_y = int(img_height * self.margin_factor)
        
        # Track statistics for logging
        too_small_count = 0
        too_large_count = 0
        accepted_count = 0
        
        for i, region in enumerate(regions):
            # Periodically log progress for large region sets
            if i > 0 and i % 50 == 0:
                self.log(f"Processed {i}/{len(regions)} regions...")
                self.log({"status": "progress", 
                          "message": f"Refining regions: {i}/{len(regions)}", 
                          "step": "refine_progress", 
                          "percent": 80 + int(10 * i / len(regions))})
            
            # Skip regions that are too small
            if region.width < 50 or region.height < 20:
                too_small_count += 1
                continue
                
            # Skip regions that are too large
            if region.width > img_width * 0.9 or region.height > img_height * 0.9:
                too_large_count += 1
                continue
                
            # Adjust region size based on aspect ratio and text length
            adjusted_region = self._adjust_region_size(
                region, img_width, img_height, text_length, min_margin_x, min_margin_y
            )
            
            refined_regions.append(adjusted_region)
            accepted_count += 1
            
            # If we have enough good regions, stop
            if len(refined_regions) >= 5:
                self.log(f"Found {len(refined_regions)} good regions, stopping early")
                break
        
        # Log statistics
        elapsed_time = time.time() - start_time
        self.log(f"Region refinement statistics:")
        self.log(f"  - Total regions processed: {len(regions)}")
        self.log(f"  - Regions too small: {too_small_count}")
        self.log(f"  - Regions too large: {too_large_count}")
        self.log(f"  - Regions accepted: {accepted_count}")
        self.log(f"  - Processing time: {elapsed_time:.2f} seconds")
                
        return refined_regions
    
    def _adjust_region_size(self, region: TextRegion, img_width: int, img_height: int,
                           text_length: int, min_margin_x: int, min_margin_y: int) -> TextRegion:
        """
        Adjust region size based on text length and design principles
        
        Args:
            region: Original TextRegion object
            img_width: Width of the image
            img_height: Height of the image
            text_length: Approximate length of text to be placed (in characters)
            min_margin_x: Minimum x margin
            min_margin_y: Minimum y margin
            
        Returns:
            Adjusted TextRegion object
        """
        # Estimate width needed for text (very rough approximation)
        estimated_width = text_length * 15  # Assuming average character width of 15px
        
        # Create a copy of the region to adjust
        adjusted = TextRegion(
            x=region.x,
            y=region.y,
            width=region.width,
            height=region.height,
            score=region.score,
            rotation=region.rotation
        )
        
        # Adjust width based on text length, but keep within bounds
        if adjusted.width < estimated_width:
            # Try to expand width
            extra_width = estimated_width - adjusted.width
            
            # Check if we can expand to the right
            right_expansion = min(extra_width, img_width - (adjusted.x + adjusted.width) - min_margin_x)
            
            # If we can't expand enough to the right, try to expand to the left
            left_expansion = 0
            if right_expansion < extra_width:
                left_expansion = min(extra_width - right_expansion, adjusted.x - min_margin_x)
            
            # Apply expansions
            adjusted.x -= left_expansion
            adjusted.width += (left_expansion + right_expansion)
        
        # Ensure the aspect ratio is within bounds
        aspect_ratio = adjusted.width / adjusted.height
        
        if aspect_ratio < self.min_aspect_ratio:
            # Too tall, increase width or decrease height
            new_height = int(adjusted.width / self.min_aspect_ratio)
            # Center the new height within the original
            height_diff = adjusted.height - new_height
            adjusted.y += height_diff // 2
            adjusted.height = new_height
            
        elif aspect_ratio > self.max_aspect_ratio:
            # Too wide, increase height or decrease width
            new_width = int(adjusted.height * self.max_aspect_ratio)
            # Center the new width within the original
            width_diff = adjusted.width - new_width
            adjusted.x += width_diff // 2
            adjusted.width = new_width
        
        # Ensure the region stays within image bounds with margins
        adjusted.x = max(min_margin_x, min(adjusted.x, img_width - adjusted.width - min_margin_x))
        adjusted.y = max(min_margin_y, min(adjusted.y, img_height - adjusted.height - min_margin_y))
        
        return adjusted
    
    def _calculate_edge_intensity(self, image_roi: np.ndarray) -> float:
        """
        Calculate the edge intensity in a region of interest
        
        Args:
            image_roi: Region of interest from the image
            
        Returns:
            Edge intensity as a float between 0 and 1
        """
        import time
        start_time = time.time()
        
        # Skip processing if ROI is too small
        if image_roi.shape[0] < 5 or image_roi.shape[1] < 5:
            self.log(f"ROI too small for edge detection: {image_roi.shape}")
            return 0.0
        
        try:
            # Convert to grayscale if needed
            if len(image_roi.shape) > 2 and image_roi.shape[2] > 1:
                gray_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = image_roi
                
            # Calculate edges using Sobel
            sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate the gradient magnitude
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize and calculate average intensity
            if np.max(magnitude) > 0:
                normalized = magnitude / np.max(magnitude)
                edge_intensity = float(np.mean(normalized))
                
                # Log edge intensity for significant edges
                if edge_intensity > 0.2:
                    self.log(f"Strong edge detected in region: {edge_intensity:.3f}")
                
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.1:  # Only log if it took significant time
                    self.log(f"Edge detection completed in {elapsed_time:.3f} seconds")
                    
                return edge_intensity
            
            return 0.0
            
        except Exception as e:
            self.log({"status": "error", "message": f"Error in edge detection: {str(e)}", "step": "edge_detection"})
            return 0.0
    
    def _suggest_font_size(self, region_width: int, text_length: int) -> int:
        """
        Suggest an appropriate font size based on region width and text length
        
        Args:
            region_width: Width of the region in pixels
            text_length: Approximate length of text to be placed (in characters)
            
        Returns:
            Suggested font size in points
        """
        # Ensure we don't divide by zero
        if text_length <= 0:
            text_length = 1
            self.log("Warning: Text length was zero or negative, using default length of 1")
        
        # Rough approximation of space needed per character
        # Average character width varies by font, but we'll use a reasonable approximation
        # Assuming a standard font where each character takes approximately 0.6 times the point size in pixels
        space_per_char = region_width / text_length
        
        # Convert to approximate point size
        # For readable text, we want each character to have enough space
        # The factor 1.7 is derived from typical character width to point size ratios
        point_size = int(space_per_char / 0.6)
        
        # Reasonable bounds for font size
        min_size = 10  # Minimum readable size
        max_size = 72  # Maximum practical size
        
        # Apply bounds
        bounded_size = max(min_size, min(max_size, point_size))
        
        # Log the font size calculation
        self.log(f"Font size calculation: region width={region_width}px, text length={text_length} chars")
        self.log(f"Space per character: {space_per_char:.1f}px, calculated point size: {point_size}pt")
        
        if bounded_size != point_size:
            self.log(f"Font size adjusted to {bounded_size}pt to stay within bounds ({min_size}-{max_size})")
        
        return bounded_size
