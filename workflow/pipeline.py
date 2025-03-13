"""
Workflow Pipeline Module for Intelligent Text Placement
"""
import os
import cv2
import numpy as np
import time
import threading
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional

from ai_analysis.image_analyzer import ImageAnalyzer
from ai_analysis.region_proposal import RegionProposer
from text_rendering.renderer import TextRenderer


class TextPlacementPipeline:
    """
    Main workflow pipeline that integrates image analysis, region proposal,
    and text rendering components into a cohesive workflow.
    """
    
    def __init__(self, dpi: int = 300, debug: bool = False):
        """
        Initialize the pipeline with component modules
        
        Args:
            dpi: Resolution in dots per inch for output (default: 300)
            debug: Enable debug mode with detailed logging (default: False)
        """
        self.image_analyzer = ImageAnalyzer()
        self.region_proposer = RegionProposer()
        self.text_renderer = TextRenderer(dpi=dpi)
        self.current_image = None
        self.current_regions = []
        self.current_proposals = []
        self.debug = debug
        self.debug_callback = None
        self.analysis_running = False
        self.analysis_start_time = None
        
    def set_debug_callback(self, callback):
        """
        Set a callback function to receive debug messages
        
        Args:
            callback: Function that takes a message as parameter (string or dict)
        """
        self.debug_callback = callback
        
        # Pass the callback to component modules
        self.image_analyzer.set_debug_callback(callback)
        self.region_proposer.set_debug_callback(callback)
        
    def log(self, message):
        """
        Log a debug message with improved formatting for UI display
        
        Args:
            message: Debug message to log (string or dict)
        """
        if self.debug:
            # Print to console for debugging
            if isinstance(message, dict):
                print(f"[TextPlacementPipeline] {message.get('message', str(message))}")
            else:
                print(f"[TextPlacementPipeline] {message}")
            
        # Send to callback if registered
        if self.debug and self.debug_callback:
            # Ensure message is properly formatted for the UI
            if isinstance(message, dict):
                # Message is already a dictionary, ensure it has required fields
                if "status" not in message:
                    message["status"] = "progress"
                if "message" not in message and "text" in message:
                    message["message"] = message["text"]
                elif "message" not in message:
                    message["message"] = "Processing..."
                if "percent" not in message:
                    message["percent"] = 30  # Default progress percentage
            else:
                # Convert string message to proper dictionary format
                message = {
                    "status": "progress",
                    "message": str(message),
                    "step": "processing",
                    "percent": 30  # Default progress percentage
                }
                
            # Send the formatted message to the callback
            self.debug_callback(message)
            
    def _report_progress(self, start_time):
        """
        Periodically report progress during long-running operations
        
        Args:
            start_time: Time when the operation started
        """
        import time
        import threading
        
        # Report progress every 2 seconds
        interval = 2.0
        count = 0
        
        while self.analysis_running:
            elapsed = time.time() - start_time
            count += 1
            
            # Send a progress update through the callback
            if self.debug_callback:
                progress_msg = {
                    "status": "progress",
                    "message": f"Analysis in progress... {elapsed:.1f} seconds elapsed"
                }
                self.debug_callback(progress_msg)
                
            # Sleep for the interval
            time.sleep(interval)
        
    def process_image(self, image_path: str, text_length: int = 20) -> List[Dict[str, Any]]:
        """Process an image to identify potential text placement regions
        
        Args:
            image_path: Path to the image file
            text_length: Approximate length of text to be placed
            
        Returns:
            List of dictionaries containing region information
        """
        try:
            # Send initial progress update
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress", 
                    "message": f"Starting analysis of {os.path.basename(image_path)}", 
                    "step": "start", 
                    "percent": 0
                })
                time.sleep(0.1)  # Small delay to ensure UI updates
            
            # Step 1: Load the image
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress", 
                    "message": "Loading image...", 
                    "step": "load_image", 
                    "percent": 10
                })
                time.sleep(0.1)
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                if self.debug_callback:
                    self.debug_callback({
                        "status": "error",
                        "message": f"Failed to load image: {image_path}",
                        "step": "error",
                        "percent": 0
                    })
                return []
            
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Update progress
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress", 
                    "message": f"Image loaded ({width}x{height})", 
                    "step": "processing", 
                    "percent": 20
                })
                time.sleep(0.1)
            
            # Step 2: Analyze the image (simplified approach)
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress", 
                    "message": "Analyzing image...", 
                    "step": "analyzing", 
                    "percent": 40
                })
                time.sleep(0.1)
            
            # Use a simple approach to find regions
            # This is a simplified version that won't get stuck
            regions = []
            
            try:
                # Try to use the image analyzer if available
                if hasattr(self, 'image_analyzer') and self.image_analyzer is not None:
                    # Update progress
                    if self.debug_callback:
                        self.debug_callback({
                            "status": "progress", 
                            "message": "Running image analysis...", 
                            "step": "analyzing", 
                            "percent": 60
                        })
                    
                    # Set a timeout for analysis
                    max_time = 30  # seconds
                    start_time = time.time()
                    
                    # Run analysis in a separate thread with timeout
                    analysis_result = [None]
                    analysis_done = [False]
                    
                    def run_analysis():
                        try:
                            analysis_result[0] = self.image_analyzer.fast_analyze(image_path)
                            analysis_done[0] = True
                        except Exception as e:
                            print(f"Analysis error: {str(e)}")
                    
                    analysis_thread = threading.Thread(target=run_analysis)
                    analysis_thread.daemon = True
                    analysis_thread.start()
                    
                    # Wait for analysis to complete or timeout
                    while not analysis_done[0] and time.time() - start_time < max_time:
                        # Update progress while waiting
                        progress = min(80, 60 + int((time.time() - start_time) / max_time * 20))
                        if self.debug_callback:
                            self.debug_callback({
                                "status": "progress", 
                                "message": f"Analyzing image... ({int(time.time() - start_time)}s)", 
                                "step": "analyzing", 
                                "percent": progress
                            })
                        time.sleep(1)
                    
                    # Check if analysis completed
                    if analysis_done[0] and analysis_result[0]:
                        regions = analysis_result[0]
                    else:
                        # Fallback to simple region if analysis timed out
                        print("Analysis timed out or failed, using fallback")
                        # Create a default region in the center
                        center_region = TextRegion(
                            x=int(width * 0.2),
                            y=int(height * 0.2),
                            width=int(width * 0.6),
                            height=int(height * 0.6),
                            score=0.8
                        )
                        regions = [center_region]
                else:
                    # No analyzer available, use fallback
                    center_region = TextRegion(
                        x=int(width * 0.2),
                        y=int(height * 0.2),
                        width=int(width * 0.6),
                        height=int(height * 0.6),
                        score=0.8
                    )
                    regions = [center_region]
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
                # Create a default region in the center
                center_region = TextRegion(
                    x=int(width * 0.2),
                    y=int(height * 0.2),
                    width=int(width * 0.6),
                    height=int(height * 0.6),
                    score=0.8
                )
                regions = [center_region]
            
            # Update progress
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress", 
                    "message": f"Found {len(regions)} potential regions", 
                    "step": "finalizing", 
                    "percent": 90
                })
                time.sleep(0.1)
            
            # Step 3: Prepare the results
            result = []
            for i, region in enumerate(regions):
                result.append({
                    "id": i,
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                    "score": region.score
                })
            
            # Final progress update
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress", 
                    "message": "Analysis complete", 
                    "step": "complete", 
                    "percent": 100
                })
                time.sleep(0.1)
            
            return result
            
        except Exception as e:
            # Handle any exceptions
            error_msg = f"Error processing image: {str(e)}"
            print(f"ERROR: {error_msg}")
            
            # Send error message
            if self.debug_callback:
                self.debug_callback({
                    "status": "error",
                    "message": error_msg,
                    "step": "error",
                    "percent": 0
                })
                
                # Always send a completion message
                self.debug_callback({
                    "status": "progress", 
                    "message": "Analysis failed", 
                    "step": "complete", 
                    "percent": 100
                })
            
            return []
    
    def render_preview(self, image_path: str, proposal_id: int, text: str,
                     custom_styling: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Render a preview of text placed according to a selected proposal
        
        Args:
            image_path: Path to the input image
            proposal_id: ID of the selected proposal
            text: Text to render
            custom_styling: Optional custom styling parameters
            
        Returns:
            PIL Image with the rendered text
        """
        # Find the selected proposal
        selected_proposal = None
        for proposal in self.current_proposals:
            if proposal["id"] == proposal_id:
                selected_proposal = proposal
                break
                
        if selected_proposal is None:
            raise ValueError(f"No proposal found with ID {proposal_id}")
            
        # Extract region and styling information
        region = selected_proposal["region"]
        styling = selected_proposal["styling"]
        
        # Merge custom styling if provided
        if custom_styling:
            styling = {**styling, **custom_styling}
            
        # Render the text
        return self.text_renderer.render_text(image_path, text, region, styling)
    
    def render_final(self, image_path: str, text: str, region: Dict[str, Any],
                   styling: Dict[str, Any], output_path: str) -> str:
        """
        Render the final output with text placed according to specified parameters
        
        Args:
            image_path: Path to the input image
            text: Text to render
            region: Dictionary with region information
            styling: Dictionary with styling information
            output_path: Path where to save the output
            
        Returns:
            Path to the rendered output
        """
        # Render the text
        rendered = self.text_renderer.render_text(image_path, text, region, styling)
        
        # Save the output
        self.text_renderer.save_output(rendered, output_path)
        
        return output_path
    
    def get_visualization(self, image_path: str, proposals: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate a visualization of the region proposals
        
        Args:
            image_path: Path to the input image
            proposals: List of region proposals (uses current proposals if None)
            
        Returns:
            OpenCV image with visualized regions
        """
        import time
        start_time = time.time()
        
        # Send progress update
        if self.debug_callback:
            self.debug_callback({
                "status": "progress",
                "message": "Generating visualization of region proposals",
                "step": "visualization_start",
                "percent": 99
            })
        
        self.log(f"Generating visualization for {os.path.basename(image_path)}")
        
        if proposals is None:
            proposals = self.current_proposals
            
        # Log the number of proposals to visualize
        self.log(f"Visualizing {len(proposals)} region proposals")
        
        try:
            # Load the image with OpenCV for visualization
            image = cv2.imread(image_path)
            if image is None:
                error_msg = f"Could not load image from {image_path}"
                self.log({
                    "status": "error",
                    "message": error_msg,
                    "step": "visualization_load",
                    "percent": 99
                })
                raise ValueError(error_msg)
                
            # Create a copy for visualization
            viz_image = image.copy()
            img_height, img_width = viz_image.shape[:2]
            self.log(f"Visualization canvas size: {img_width}x{img_height}")
            
            # Draw each region with its ID
            for idx, proposal in enumerate(proposals):
                region = proposal["region"]
                score = proposal["score"]
                
                x, y = region["x"], region["y"]
                width, height = region["width"], region["height"]
                rotation = region.get("rotation", 0)
                
                # Choose color based on score (green for high scores, red for low)
                color = (
                    int(255 * (1 - score)),  # B
                    int(255 * score),        # G
                    0                        # R
                )
                
                # Draw rectangle
                cv2.rectangle(viz_image, (x, y), (x + width, y + height), color, 2)
                
                # Draw ID and score
                cv2.putText(
                    viz_image, 
                    f"ID: {proposal['id']} ({score:.2f})", 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    color, 
                    2
                )
                
                # Draw styling info if available
                if "styling" in proposal:
                    styling = proposal["styling"]
                    font_size = styling.get("suggested_font_size", "?")
                    text_color = styling.get("color", "?")
                    
                    # Draw styling info below the region
                    info_text = f"Font: {font_size}pt, Color: {text_color}"
                    cv2.putText(
                        viz_image,
                        info_text,
                        (x, y + height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        1
                    )
                
                # Visualize rotation if specified
                if rotation != 0:
                    # Calculate center point
                    center_x = x + width // 2
                    center_y = y + height // 2
                    
                    # Draw a line indicating rotation
                    angle_rad = math.radians(rotation)
                    line_length = min(width, height) // 2
                    
                    end_x = int(center_x + line_length * math.cos(angle_rad))
                    end_y = int(center_y + line_length * math.sin(angle_rad))
                    
                    cv2.line(viz_image, (center_x, center_y), (end_x, end_y), color, 2)
            
            # Add a legend to the visualization
            legend_y = 30
            cv2.putText(
                viz_image,
                "Legend: Green = High Score, Red = Low Score",
                (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
            
            # Log completion
            elapsed_time = time.time() - start_time
            self.log(f"Visualization generated in {elapsed_time:.2f} seconds")
            
            # Send completion update
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress",
                    "message": "Visualization complete",
                    "step": "visualization_complete",
                    "percent": 100
                })
                
            return viz_image
            
        except Exception as e:
            error_msg = f"Error generating visualization: {str(e)}"
            self.log({
                "status": "error",
                "message": error_msg,
                "step": "visualization_error",
                "percent": 99
            })
            # Return a blank image with error message if visualization fails
            error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(error_img, "Visualization Error", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_img, str(e), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            return error_img
    
    def save_visualization(self, viz_image: np.ndarray, output_path: str) -> str:
        """
        Save the visualization image
        
        Args:
            viz_image: OpenCV image with visualizations
            output_path: Path where to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        import time
        start_time = time.time()
        
        # Send progress update
        if self.debug_callback:
            self.debug_callback({
                "status": "progress",
                "message": f"Saving visualization to {os.path.basename(output_path)}",
                "step": "save_visualization_start",
                "percent": 95
            })
        
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log(f"Created output directory: {output_dir}")
            
            # Get image dimensions and file size estimate
            height, width = viz_image.shape[:2]
            channels = 3 if len(viz_image.shape) > 2 else 1
            estimated_size_kb = (width * height * channels) / 1024
            
            self.log(f"Saving visualization ({width}x{height}, ~{estimated_size_kb:.1f}KB) to {output_path}")
            
            # Save the image
            result = cv2.imwrite(output_path, viz_image)
            
            if not result:
                error_msg = f"Failed to save visualization to {output_path}"
                self.log({
                    "status": "error",
                    "message": error_msg,
                    "step": "save_visualization_error",
                    "percent": 95
                })
                raise IOError(error_msg)
            
            # Get actual file size
            file_size_kb = os.path.getsize(output_path) / 1024
            elapsed_time = time.time() - start_time
            
            # Log success with highlight
            success_msg = f"Visualization saved successfully to {output_path} ({file_size_kb:.1f}KB) in {elapsed_time:.2f} seconds"
            self.log({
                "status": "highlight",
                "message": success_msg,
                "step": "save_visualization_complete",
                "percent": 100
            })
            
            # Send completion update
            if self.debug_callback:
                self.debug_callback({
                    "status": "progress",
                    "message": "Visualization saved successfully",
                    "step": "save_visualization_complete",
                    "percent": 100
                })
                
            return output_path
            
        except Exception as e:
            error_msg = f"Error saving visualization: {str(e)}"
            self.log({
                "status": "error",
                "message": error_msg,
                "step": "save_visualization_error",
                "percent": 95
            })
            raise
