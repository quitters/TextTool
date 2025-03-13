"""
Text Rendering Module for Intelligent Text Placement
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
import math
from typing import Tuple, Dict, Any, Optional, List


class TextRenderer:
    """
    High-quality vector-based text renderer with support for transformations,
    effects, and styling to ensure professional output suitable for print.
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialize the text renderer
        
        Args:
            dpi: Resolution in dots per inch (default: 300 for print quality)
        """
        self.dpi = dpi
        self.default_font = "Arial"
        
        # Scale factor for high-resolution rendering
        # We render at a higher resolution and then scale down for better quality
        self.scale_factor = max(1, self.dpi // 72)
    
    def render_text(self, image_path: str, text: str, region: Dict[str, Any], 
                   styling: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Render text onto an image with specified transformations and styling
        
        Args:
            image_path: Path to the image
            text: Text to render
            region: Dictionary with region information (x, y, width, height, rotation)
            styling: Dictionary with styling information
            
        Returns:
            PIL Image with the rendered text
        """
        # Load image with PIL for better text rendering
        image = Image.open(image_path).convert("RGBA")
        
        # Create a default styling if none provided
        if styling is None:
            styling = self._get_default_styling()
        else:
            # Merge with defaults for any missing values
            styling = {**self._get_default_styling(), **styling}
        
        # Create a transparent layer for text
        text_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        
        # Process and render the text
        rendered_text = self._process_text(text_layer, text, region, styling)
        
        # Composite the rendered text onto the original image
        result = Image.alpha_composite(image, rendered_text)
        
        return result
    
    def _get_default_styling(self) -> Dict[str, Any]:
        """
        Get default styling parameters
        
        Returns:
            Dictionary with default styling values
        """
        return {
            "font": self.default_font,
            "font_size": 24,
            "color": "black",
            "outline_color": None,  # No outline by default
            "outline_width": 1,
            "shadow": False,
            "shadow_color": (0, 0, 0, 128),  # Semi-transparent black
            "shadow_offset": (3, 3),
            "shadow_blur": 5,
            "opacity": 1.0,
            "alignment": "center",  # One of: left, center, right
            "warp": None,  # No warping by default
            "blend_mode": "normal"  # One of: normal, multiply, screen, overlay
        }
    
    def _process_text(self, text_layer: Image.Image, text: str, 
                     region: Dict[str, Any], styling: Dict[str, Any]) -> Image.Image:
        """
        Process and render text with transformations and effects
        
        Args:
            text_layer: Transparent PIL Image for rendering text
            text: Text to render
            region: Dictionary with region information
            styling: Dictionary with styling information
            
        Returns:
            PIL Image with rendered text
        """
        # Extract region parameters
        x, y = region["x"], region["y"]
        width, height = region["width"], region["height"]
        rotation = region.get("rotation", 0)
        
        # Scale for high-resolution rendering
        sf = self.scale_factor
        x_scaled, y_scaled = x * sf, y * sf
        width_scaled, height_scaled = width * sf, height * sf
        
        # Create high-resolution text layer
        hi_res_size = (text_layer.width * sf, text_layer.height * sf)
        hi_res_text_layer = Image.new("RGBA", hi_res_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(hi_res_text_layer)
        
        # Prepare font
        font_size = styling["font_size"] * sf
        try:
            font = ImageFont.truetype(styling["font"], size=int(font_size))
        except (IOError, OSError):
            # Fallback to default font
            font = ImageFont.load_default()
            if hasattr(font, "size"):
                font = font.font_variant(size=int(font_size))
        
        # Get text dimensions for positioning
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Determine text position based on alignment
        if styling["alignment"] == "left":
            text_x = x_scaled
        elif styling["alignment"] == "center":
            text_x = x_scaled + (width_scaled - text_width) / 2
        else:  # right
            text_x = x_scaled + width_scaled - text_width
            
        # Center vertically
        text_y = y_scaled + (height_scaled - text_height) / 2
        
        # Prepare text color
        text_color = styling["color"]
        if isinstance(text_color, str):
            text_color = ImageColor.getrgb(text_color)
            
        # Add alpha channel if needed
        if len(text_color) == 3:
            opacity = int(255 * styling["opacity"])
            text_color = text_color + (opacity,)
        
        # Render shadow if enabled
        if styling["shadow"]:
            shadow_x = text_x + styling["shadow_offset"][0] * sf
            shadow_y = text_y + styling["shadow_offset"][1] * sf
            
            # Draw shadow text
            draw.text((shadow_x, shadow_y), text, font=font, fill=styling["shadow_color"])
            
            # Apply blur to the entire layer for the shadow
            hi_res_text_layer = hi_res_text_layer.filter(
                ImageFilter.GaussianBlur(radius=styling["shadow_blur"] * sf / 3)
            )
            
            # Create a new draw object after the filter
            draw = ImageDraw.Draw(hi_res_text_layer)
        
        # Draw outline if specified
        if styling["outline_color"]:
            outline_color = styling["outline_color"]
            if isinstance(outline_color, str):
                outline_color = ImageColor.getrgb(outline_color)
                
            # Add alpha channel if needed
            if len(outline_color) == 3:
                opacity = int(255 * styling["opacity"])
                outline_color = outline_color + (opacity,)
                
            # Draw text multiple times for outline effect
            outline_width = max(1, int(styling["outline_width"] * sf))
            for offset_x in range(-outline_width, outline_width + 1, max(1, outline_width // 2)):
                for offset_y in range(-outline_width, outline_width + 1, max(1, outline_width // 2)):
                    if offset_x == 0 and offset_y == 0:
                        continue  # Skip the center position
                    draw.text((text_x + offset_x, text_y + offset_y), text, font=font, fill=outline_color)
        
        # Draw the main text
        draw.text((text_x, text_y), text, font=font, fill=text_color)
        
        # Apply warping if specified
        if styling["warp"]:
            hi_res_text_layer = self._apply_warp(hi_res_text_layer, styling["warp"])
        
        # Apply rotation if specified
        if rotation != 0:
            # Rotate around the center of the text
            center_x = text_x + text_width / 2
            center_y = text_y + text_height / 2
            
            # Create a larger canvas to accommodate rotation
            rotation_layer = Image.new("RGBA", hi_res_text_layer.size, (0, 0, 0, 0))
            
            # Paste the text layer (this step is necessary to handle alpha correctly during rotation)
            rotation_layer.paste(hi_res_text_layer, (0, 0), hi_res_text_layer)
            
            # Rotate and re-center
            hi_res_text_layer = rotation_layer.rotate(
                -rotation,  # Negative because PIL rotates counterclockwise
                center=(center_x, center_y),
                resample=Image.BICUBIC,
                expand=False
            )
        
        # Scale back down to original resolution
        final_text_layer = hi_res_text_layer.resize(
            (text_layer.width, text_layer.height),
            Image.LANCZOS
        )
        
        # Apply blend mode
        if styling["blend_mode"] != "normal":
            final_text_layer = self._apply_blend_mode(text_layer, final_text_layer, styling["blend_mode"])
        
        return final_text_layer
    
    def _apply_warp(self, image: Image.Image, warp_params: Dict[str, Any]) -> Image.Image:
        """
        Apply warping transformation to the text
        
        Args:
            image: PIL Image to transform
            warp_params: Dictionary with warping parameters
            
        Returns:
            Transformed PIL Image
        """
        warp_type = warp_params.get("type", "perspective")
        strength = warp_params.get("strength", 0.2)
        
        width, height = image.size
        
        if warp_type == "perspective":
            # Create perspective transform
            src_points = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            
            # Calculate destination points based on strength
            offset_x = int(width * strength)
            offset_y = int(height * strength * 0.5)
            
            dst_points = [
                (offset_x, 0),
                (width - offset_x, 0),
                (width, height),
                (0, height)
            ]
            
            # Calculate perspective transform matrix
            coeffs = self._find_coeffs(dst_points, src_points)
            
            # Apply perspective transform
            return image.transform(
                image.size, 
                Image.PERSPECTIVE, 
                coeffs,
                Image.BICUBIC
            )
            
        elif warp_type == "arc":
            # Arc/curve warping
            # Convert PIL image to numpy array for easier pixel manipulation
            img_np = np.array(image)
            
            # Create output array
            result = np.zeros_like(img_np)
            
            # Calculate arc parameters
            amplitude = height * strength
            
            # Apply arc transformation
            for y in range(height):
                for x in range(width):
                    # Calculate offset based on arc equation
                    x_offset = int(amplitude * math.sin(math.pi * x / width))
                    
                    new_y = y + x_offset
                    
                    # Ensure we stay within bounds
                    if 0 <= new_y < height:
                        result[new_y, x] = img_np[y, x]
            
            # Convert back to PIL image
            return Image.fromarray(result)
        
        # Default: no warping
        return image
    
    def _find_coeffs(self, pa: List[Tuple[int, int]], pb: List[Tuple[int, int]]) -> List[float]:
        """
        Find coefficients for perspective transformation
        
        Args:
            pa: Destination points
            pb: Source points
            
        Returns:
            Coefficients for the perspective transform
        """
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8).tolist()
    
    def _apply_blend_mode(self, background: Image.Image, foreground: Image.Image, 
                         blend_mode: str) -> Image.Image:
        """
        Apply blend mode between foreground and background
        
        Args:
            background: Background PIL Image
            foreground: Foreground PIL Image
            blend_mode: String specifying the blend mode
            
        Returns:
            Blended PIL Image
        """
        # Convert to numpy arrays for easier manipulation
        bg = np.array(background).astype(float) / 255
        fg = np.array(foreground).astype(float) / 255
        
        # Extract alpha channel
        if bg.shape[2] == 4:
            bg_alpha = bg[:, :, 3:4]
            bg_rgb = bg[:, :, :3]
        else:
            bg_alpha = np.ones((*bg.shape[:2], 1))
            bg_rgb = bg
            
        if fg.shape[2] == 4:
            fg_alpha = fg[:, :, 3:4]
            fg_rgb = fg[:, :, :3]
        else:
            fg_alpha = np.ones((*fg.shape[:2], 1))
            fg_rgb = fg
        
        # Apply blend mode
        if blend_mode == "multiply":
            blended_rgb = bg_rgb * fg_rgb
        elif blend_mode == "screen":
            blended_rgb = 1 - (1 - bg_rgb) * (1 - fg_rgb)
        elif blend_mode == "overlay":
            mask = bg_rgb <= 0.5
            blended_rgb = np.zeros_like(bg_rgb)
            blended_rgb[mask] = 2 * bg_rgb[mask] * fg_rgb[mask]
            blended_rgb[~mask] = 1 - 2 * (1 - bg_rgb[~mask]) * (1 - fg_rgb[~mask])
        else:  # normal or unknown mode
            blended_rgb = fg_rgb
            
        # Combine alpha using Porter-Duff "over" operation
        blended_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
        
        # Combine RGB with alpha
        blended = np.concatenate([
            blended_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha),
            blended_alpha
        ], axis=2)
        
        # Convert back to PIL Image
        blended_uint8 = (blended * 255).astype(np.uint8)
        return Image.fromarray(blended_uint8)
    
    def save_output(self, image: Image.Image, output_path: str, quality: int = 95) -> None:
        """
        Save the rendered image at high resolution
        
        Args:
            image: PIL Image to save
            output_path: Path where to save the output
            quality: JPEG quality (0-100)
        """
        # Determine format based on extension
        extension = output_path.split('.')[-1].lower()
        
        if extension == 'jpg' or extension == 'jpeg':
            # Convert to RGB for JPEG (no alpha channel)
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # Use alpha as mask
                image = background
                
            image.save(output_path, quality=quality, dpi=(self.dpi, self.dpi))
        elif extension == 'png':
            image.save(output_path, dpi=(self.dpi, self.dpi))
        elif extension == 'pdf':
            # For PDF, maintain vector quality
            image.save(output_path, resolution=self.dpi)
        else:
            # Default save
            image.save(output_path)
