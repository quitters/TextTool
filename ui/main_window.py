"""
Main Window UI Module for Intelligent Text Placement (Tkinter version)
"""
import sys
import os
import time
import queue
import threading
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, scrolledtext
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
from workflow.pipeline import TextPlacementPipeline
from tkinter import colorchooser

class BackgroundTask(threading.Thread):
    """Base class for background tasks"""
    
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.daemon = True
        
    def run(self):
        raise NotImplementedError("Subclasses must implement run()")
        
    def get_result(self):
        """Get the result of the task if available"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


class ImageAnalysisTask(BackgroundTask):
    """Thread for image analysis"""
    
    def __init__(self, pipeline: TextPlacementPipeline, image_path: str, text_length: int = 20):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path
        self.text_length = text_length
        self.progress_updates = []
        self.last_update_time = time.time()
        self.has_completed = False
    
    def run(self):
        """Execute the task in a separate thread"""
        try:
            # Simple debug callback to track progress
            def debug_callback(data):
                if isinstance(data, dict) and "status" in data:
                    # Put progress updates in the queue
                    self.queue.put((False, data))
                    self.progress_updates.append(data)
                    self.last_update_time = time.time()
            
            # Set the debug callback in the pipeline
            self.pipeline.set_debug_callback(debug_callback)
            
            # Send initial progress update
            debug_callback({
                "status": "progress", 
                "message": f"Processing image {os.path.basename(self.image_path)}", 
                "step": "start", 
                "percent": 5
            })
            
            # Process the image
            try:
                # Simplified approach - just call process_image and let it handle progress updates
                result = self.pipeline.process_image(self.image_path, self.text_length)
                
                # Format the result as a dictionary with proposals
                if isinstance(result, list):
                    formatted_result = {
                        "proposals": result,
                        "log": self.progress_updates.copy()
                    }
                else:
                    formatted_result = result
                
                # Send completion update
                debug_callback({
                    "status": "progress", 
                    "message": "Analysis complete", 
                    "step": "complete", 
                    "percent": 100
                })
                
                # Return the result
                self.queue.put((True, formatted_result))
                self.has_completed = True
                
            except Exception as e:
                # Log the error
                error_msg = f"Error processing image: {str(e)}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                
                # Send error message to UI
                debug_callback({
                    "status": "error",
                    "message": error_msg,
                    "step": "error",
                    "percent": 0
                })
                
                # Always send a completion message to ensure UI updates
                debug_callback({
                    "status": "progress", 
                    "message": "Analysis failed", 
                    "step": "complete", 
                    "percent": 100
                })
                
                # Return empty result
                self.queue.put((True, {"proposals": [], "error": str(e), "log": self.progress_updates.copy()}))
                self.has_completed = True
                
        except Exception as e:
            # Catch any other exceptions
            error_msg = f"Unexpected error in analysis task: {str(e)}"
            print(f"CRITICAL ERROR: {error_msg}")
            traceback.print_exc()
            self.queue.put((True, {"proposals": [], "error": str(e)}))
            self.has_completed = True
    
    def get_result(self):
        """Get the latest result or progress update"""
        try:
            if not self.queue.empty():
                return self.queue.get_nowait()
            
            # If there are progress updates but nothing in the queue,
            # return the latest progress update
            if self.progress_updates and not self.has_completed:
                return (False, self.progress_updates[-1])
            
            return None
        except Exception as e:
            print(f"Error getting result: {e}")
            return None


class RenderPreviewTask(BackgroundTask):
    """Thread for rendering text previews"""
    
    def __init__(self, pipeline: TextPlacementPipeline, image_path: str, 
                proposal_id: int, text: str, styling: dict):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path
        self.proposal_id = proposal_id
        self.text = text
        self.styling = styling
        
    def run(self):
        """Render the text preview"""
        try:
            rendered = self.pipeline.render_preview(
                self.image_path,
                self.proposal_id,
                self.text,
                self.styling
            )
            self.queue.put((True, rendered))
        except Exception as e:
            self.queue.put((False, str(e)))


class ProposalItem:
    """Class to represent a region proposal in the UI"""
    
    def __init__(self, proposal):
        """Initialize with a proposal dictionary"""
        self.id = proposal.get("id", "unknown")
        self.x = proposal.get("x", 0)
        self.y = proposal.get("y", 0)
        self.width = proposal.get("width", 0)
        self.height = proposal.get("height", 0)
        self.score = proposal.get("score", 0.0)
        
    def __str__(self):
        """String representation for display in listbox"""
        return f"{self.id} (Score: {self.score:.2f}) - [{self.x},{self.y},{self.width}x{self.height}]"


class MainWindow(tk.Frame):
    """Main application window for Intelligent Text Placement (Tkinter version)"""
    
    def __init__(self, master):
        super().__init__(master)
        
        # Store reference to master
        self.master = master
        
        # Initialize the pipeline with debug mode enabled
        self.pipeline = TextPlacementPipeline(dpi=300, debug=True)
        
        # Initialize state variables
        self.current_image_path = None
        self.current_proposals = []
        self.selected_proposal = None
        self.selected_proposal_obj = None
        self.preview_image = None
        self.text_color = "#000000"  # Black
        self.outline_color = "#FFFFFF"  # White
        self.current_task = None
        self.tk_image = None  # For storing PhotoImage references
        
        # Setup UI
        self.pack(fill=tk.BOTH, expand=True)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Configure main window
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Create main frame with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        # Create left panel (controls)
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        # Create right panel (image preview)
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        # Configure panels
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
        # Create status bar with debug button
        status_frame = ttk.Frame(self)
        status_frame.grid(column=0, row=1, sticky=(tk.W, tk.E))
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add a debug log button
        self.debug_btn = ttk.Button(status_frame, text="Show Debug Log", command=self.show_last_debug_log)
        self.debug_btn.pack(side=tk.RIGHT, padx=5)
        self.debug_btn["state"] = tk.DISABLED
        
        # Store debug logs
        self.last_debug_log = []
        
    def setup_left_panel(self, parent):
        """Setup the left panel with controls"""
        # Configure parent
        parent.columnconfigure(0, weight=1)
        
        # Input section
        input_frame = ttk.LabelFrame(parent, text="Input")
        input_frame.grid(column=0, row=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        # Image selection
        image_frame = ttk.Frame(input_frame)
        image_frame.grid(column=0, row=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        image_frame.columnconfigure(0, weight=1)
        
        self.image_path_var = tk.StringVar(value="No image selected")
        image_label = ttk.Label(image_frame, textvariable=self.image_path_var, wraplength=200)
        image_label.grid(column=0, row=0, sticky=(tk.W, tk.E))
        
        select_btn = ttk.Button(image_frame, text="Select Image", command=self.on_select_image)
        select_btn.grid(column=1, row=0, padx=5)
        
        # Text input
        text_frame = ttk.Frame(input_frame)
        text_frame.grid(column=0, row=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        text_frame.columnconfigure(0, weight=1)
        
        ttk.Label(text_frame, text="Text to Place:").grid(column=0, row=0, sticky=tk.W)
        
        self.text_input = scrolledtext.ScrolledText(text_frame, height=3, wrap=tk.WORD)
        self.text_input.grid(column=0, row=1, sticky=(tk.W, tk.E))
        self.text_input.bind("<KeyRelease>", self.on_text_changed)
        
        # Analysis button
        self.analyze_btn = ttk.Button(input_frame, text="Analyze Image for Text Placement", 
                                    command=self.on_analyze_image, state=tk.DISABLED)
        self.analyze_btn.grid(column=0, row=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Region proposals section
        proposals_frame = ttk.LabelFrame(parent, text="Region Proposals")
        proposals_frame.grid(column=0, row=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        proposals_frame.columnconfigure(0, weight=1)
        proposals_frame.rowconfigure(0, weight=1)
        
        # Create a listbox with scrollbar
        proposals_frame_inner = ttk.Frame(proposals_frame)
        proposals_frame_inner.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        proposals_frame_inner.columnconfigure(0, weight=1)
        proposals_frame_inner.rowconfigure(0, weight=1)
        
        self.proposals_list = tk.Listbox(proposals_frame_inner, height=6)
        self.proposals_list.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.proposals_list.bind("<<ListboxSelect>>", self.on_proposal_selected)
        
        scrollbar = ttk.Scrollbar(proposals_frame_inner, orient=tk.VERTICAL, command=self.proposals_list.yview)
        scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S))
        self.proposals_list['yscrollcommand'] = scrollbar.set
        
        # Styling section
        styling_frame = ttk.LabelFrame(parent, text="Text Styling")
        styling_frame.grid(column=0, row=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        styling_frame.columnconfigure(0, weight=1)
        styling_frame.columnconfigure(1, weight=1)
        
        # Font selection
        ttk.Label(styling_frame, text="Font:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=2)
        self.font_var = tk.StringVar(value="Arial")
        self.font_combo = ttk.Combobox(styling_frame, textvariable=self.font_var, state="readonly")
        self.font_combo['values'] = ("Arial", "Times New Roman", "Courier New", "Verdana", "Georgia", "Helvetica", "Tahoma", "Trebuchet MS")
        self.font_combo.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.font_combo.bind("<<ComboboxSelected>>", self.on_styling_changed)
        
        # Font size
        ttk.Label(styling_frame, text="Size:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=2)
        self.size_var = tk.IntVar(value=24)
        size_spin = ttk.Spinbox(styling_frame, from_=8, to=120, textvariable=self.size_var, width=5)
        size_spin.grid(column=1, row=1, sticky=tk.W, padx=5, pady=2)
        size_spin.bind("<KeyRelease>", self.on_styling_changed)
        size_spin.bind("<<Increment>>", self.on_styling_changed)
        size_spin.bind("<<Decrement>>", self.on_styling_changed)
        
        # Text color
        ttk.Label(styling_frame, text="Color:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=2)
        color_frame = ttk.Frame(styling_frame)
        color_frame.grid(column=1, row=2, sticky=tk.W, padx=5, pady=2)
        
        # Create a button with color indicator
        self.color_indicator = tk.Canvas(color_frame, width=20, height=20, bg=self.text_color, 
                                       highlightthickness=1, highlightbackground="black")
        self.color_indicator.pack(side=tk.LEFT, padx=2)
        
        color_pick_btn = ttk.Button(color_frame, text="Choose Color", command=self.on_color_select)
        color_pick_btn.pack(side=tk.LEFT, padx=2)
        
        # Opacity slider
        ttk.Label(styling_frame, text="Opacity:").grid(column=0, row=3, sticky=tk.W, padx=5, pady=2)
        self.opacity_var = tk.DoubleVar(value=1.0)
        opacity_slider = ttk.Scale(styling_frame, from_=0.1, to=1.0, variable=self.opacity_var, 
                                  orient=tk.HORIZONTAL, command=self.on_styling_changed)
        opacity_slider.grid(column=1, row=3, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Outline
        self.outline_var = tk.BooleanVar(value=False)
        outline_check = ttk.Checkbutton(styling_frame, text="Outline", variable=self.outline_var, 
                                       command=self.on_styling_changed)
        outline_check.grid(column=0, row=4, sticky=tk.W, padx=5, pady=2)
        
        outline_frame = ttk.Frame(styling_frame)
        outline_frame.grid(column=1, row=4, sticky=tk.W, padx=5, pady=2)
        
        # Create a canvas for outline color indicator
        self.outline_indicator = tk.Canvas(outline_frame, width=20, height=20, bg=self.outline_color,
                                         highlightthickness=1, highlightbackground="black")
        self.outline_indicator.pack(side=tk.LEFT, padx=2)
        
        self.outline_btn = ttk.Button(outline_frame, text="Choose Color", 
                                    command=self.on_outline_color_select)
        self.outline_btn.pack(side=tk.LEFT, padx=2)
        
        # Outline width
        ttk.Label(styling_frame, text="Outline Width:").grid(column=0, row=5, sticky=tk.W, padx=5, pady=2)
        self.outline_width_var = tk.IntVar(value=1)
        outline_width_spin = ttk.Spinbox(styling_frame, from_=1, to=10, textvariable=self.outline_width_var, width=5)
        outline_width_spin.grid(column=1, row=5, sticky=tk.W, padx=5, pady=2)
        outline_width_spin.bind("<KeyRelease>", self.on_styling_changed)
        outline_width_spin.bind("<<Increment>>", self.on_styling_changed)
        outline_width_spin.bind("<<Decrement>>", self.on_styling_changed)
        
        # Shadow
        self.shadow_var = tk.BooleanVar(value=False)
        shadow_check = ttk.Checkbutton(styling_frame, text="Shadow", variable=self.shadow_var, 
                                      command=self.on_styling_changed)
        shadow_check.grid(column=0, row=6, sticky=tk.W, padx=5, pady=2)
        
        # Shadow blur
        ttk.Label(styling_frame, text="Shadow Blur:").grid(column=0, row=7, sticky=tk.W, padx=5, pady=2)
        self.shadow_blur_var = tk.IntVar(value=5)
        shadow_blur_spin = ttk.Spinbox(styling_frame, from_=1, to=20, textvariable=self.shadow_blur_var, width=5)
        shadow_blur_spin.grid(column=1, row=7, sticky=tk.W, padx=5, pady=2)
        shadow_blur_spin.bind("<KeyRelease>", self.on_styling_changed)
        shadow_blur_spin.bind("<<Increment>>", self.on_styling_changed)
        shadow_blur_spin.bind("<<Decrement>>", self.on_styling_changed)
        
        # Blend mode
        ttk.Label(styling_frame, text="Blend Mode:").grid(column=0, row=8, sticky=tk.W, padx=5, pady=2)
        self.blend_mode_var = tk.StringVar(value="normal")
        self.blend_mode_combo = ttk.Combobox(styling_frame, textvariable=self.blend_mode_var, state="readonly")
        self.blend_mode_combo['values'] = ("Normal", "Multiply", "Screen", "Overlay")
        self.blend_mode_combo.current(0)  # Normal by default
        self.blend_mode_combo.grid(column=1, row=8, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.blend_mode_combo.bind("<<ComboboxSelected>>", self.on_styling_changed)
        
        # Text warping
        self.warp_var = tk.BooleanVar(value=False)
        warp_check = ttk.Checkbutton(styling_frame, text="Apply Warping", variable=self.warp_var, 
                                    command=self.on_styling_changed)
        warp_check.grid(column=0, row=9, sticky=tk.W, padx=5, pady=2)
        
        # Warp type
        ttk.Label(styling_frame, text="Warp Type:").grid(column=0, row=10, sticky=tk.W, padx=5, pady=2)
        self.warp_type_var = tk.StringVar(value="perspective")
        self.warp_type_combo = ttk.Combobox(styling_frame, textvariable=self.warp_type_var, state="readonly")
        self.warp_type_combo['values'] = ("Perspective", "Arc", "Wave")
        self.warp_type_combo.current(0)  # Perspective by default
        self.warp_type_combo.grid(column=1, row=10, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.warp_type_combo.bind("<<ComboboxSelected>>", self.on_styling_changed)
        
        # Warp strength
        ttk.Label(styling_frame, text="Warp Strength:").grid(column=0, row=11, sticky=tk.W, padx=5, pady=2)
        self.warp_strength_var = tk.DoubleVar(value=0.2)
        warp_strength_slider = ttk.Scale(styling_frame, from_=0.1, to=0.5, variable=self.warp_strength_var, 
                                       orient=tk.HORIZONTAL, command=self.on_styling_changed)
        warp_strength_slider.grid(column=1, row=11, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Alignment
        ttk.Label(styling_frame, text="Alignment:").grid(column=0, row=12, sticky=tk.W, padx=5, pady=2)
        self.align_var = tk.StringVar(value="Center")
        align_combo = ttk.Combobox(styling_frame, textvariable=self.align_var, state="readonly")
        align_combo['values'] = ("Left", "Center", "Right")
        align_combo.current(1)  # Center by default
        align_combo.grid(column=1, row=12, sticky=(tk.W, tk.E), padx=5, pady=2)
        align_combo.bind("<<ComboboxSelected>>", self.on_styling_changed)
        
        # Export section
        self.setup_export_section(parent)
        
    def setup_export_section(self, parent):
        """Setup the export section of the UI"""
        export_frame = ttk.LabelFrame(parent, text="Export")
        export_frame.grid(column=0, row=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        export_frame.columnconfigure(0, weight=1)
        export_frame.columnconfigure(1, weight=1)
        
        # DPI selection
        ttk.Label(export_frame, text="Resolution (DPI):").grid(column=0, row=0, sticky=tk.W, padx=5, pady=2)
        self.dpi_var = tk.StringVar(value="300")
        self.dpi_combo = ttk.Combobox(export_frame, textvariable=self.dpi_var, state="readonly")
        self.dpi_combo['values'] = ("72", "150", "300", "600")
        self.dpi_combo.current(2)  # 300 DPI by default
        self.dpi_combo.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Quality selection for JPEG
        ttk.Label(export_frame, text="JPEG Quality:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=2)
        self.quality_var = tk.IntVar(value=95)
        quality_spin = ttk.Spinbox(export_frame, from_=1, to=100, textvariable=self.quality_var, width=5)
        quality_spin.grid(column=1, row=1, sticky=tk.W, padx=5, pady=2)
        
        # Export button
        self.export_btn = ttk.Button(export_frame, text="Export Image", command=self.on_export)
        self.export_btn.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
    def setup_right_panel(self, parent):
        """Setup the right panel with image preview"""
        # Configure parent
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Create a frame for the preview
        preview_frame = ttk.LabelFrame(parent, text="Preview")
        preview_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Create a canvas for the image preview with scrollbars
        self.preview_canvas = tk.Canvas(preview_frame, bg="#f0f0f0", highlightthickness=0)
        self.preview_canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        # Add scrollbars
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        h_scrollbar.grid(column=0, row=1, sticky=(tk.W, tk.E))
        
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        v_scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S))
        
        self.preview_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Create a label for the preview text
        self.preview_label = ttk.Label(self.preview_canvas, text="No preview available", 
                                      background="#f0f0f0")
        self.preview_window = self.preview_canvas.create_window((0, 0), window=self.preview_label, anchor=tk.NW)
        
        # Bind canvas resizing
        self.preview_canvas.bind("<Configure>", self.on_canvas_configure)
        
    def on_canvas_configure(self, event):
        """Handle canvas resizing"""
        # Update the scrollregion to encompass the preview label
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
        
    def on_select_image(self):
        """Handle image selection from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_path_var.set(os.path.basename(file_path))
            self.analyze_btn["state"] = tk.NORMAL
            
            # Display the original image
            self.display_image(file_path)
            
            # Clear any existing proposals
            self.proposals_list.delete(0, tk.END)
            self.current_proposals = []
            self.selected_proposal = None
            self.export_btn["state"] = tk.DISABLED
            
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
    
    def on_analyze_image(self):
        """Handle image analysis"""
        if not self.current_image_path:
            return
            
        # Get text length for better region sizing
        text = self.text_input.get("1.0", tk.END)
        text_length = len(text.strip()) if text.strip() else 20
        
        # Disable UI during analysis
        self.analyze_btn["state"] = tk.DISABLED
        self.status_var.set("Analyzing image...")
        
        # Create progress bar
        self.create_progress_bar()
        
        # Clear any existing proposals
        self.proposals_list.delete(0, tk.END)
        self.current_proposals = []
        self.selected_proposal = None
        self.export_btn["state"] = tk.DISABLED
        
        # Add a default centered proposal that's always available
        self.add_default_proposal()
        
        # Run analysis in a background thread
        try:
            self.current_task = ImageAnalysisTask(self.pipeline, self.current_image_path, text_length)
            self.current_task.start()
            
            # Start checking for task completion
            self.after(100, self.check_tasks)
        except Exception as e:
            self.status_var.set(f"Error starting analysis: {str(e)}")
            self.analyze_btn["state"] = tk.NORMAL
            self.remove_progress_bar()
            messagebox.showerror("Analysis Error", f"Failed to start analysis: {str(e)}")
            
            # Even if analysis fails, we still have the default proposal
            self.proposals_list.selection_set(0)
            self.on_proposal_selected(None)
            
    def add_default_proposal(self):
        """Add a default centered proposal that's always available"""
        try:
            # Get image dimensions
            if self.current_image_path:
                img = cv2.imread(self.current_image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    
                    # Create a default centered proposal
                    default_proposal = {
                        "id": "default_center",
                        "x": int(width * 0.2),
                        "y": int(height * 0.2),
                        "width": int(width * 0.6),
                        "height": int(height * 0.6),
                        "score": 1.0,
                        "name": "Centered (Default)"
                    }
                    
                    # Add to list of proposals
                    self.current_proposals.append(default_proposal)
                    
                    # Add to listbox
                    self.proposals_list.insert(tk.END, "Centered (Default)")
                    
                    # Enable the export button
                    self.export_btn["state"] = tk.NORMAL
                    
                    return True
            
            return False
        except Exception as e:
            print(f"Error adding default proposal: {e}")
            return False
    
    def on_analysis_complete(self, result):
        """Handle completion of image analysis"""
        # Check if the result is None or indicates an error
        if result is None:
            self.remove_progress_bar()
            self.status_var.set("Analysis failed, using default region")
            self.analyze_btn["state"] = tk.NORMAL
            
            # Ensure we have at least the default proposal selected
            if self.proposals_list.size() > 0 and not self.selected_proposal:
                self.proposals_list.selection_set(0)
                self.on_proposal_selected(None)
            return
        
        # Check if this is a progress update
        if isinstance(result, dict) and "status" in result and result["status"] == "progress":
            # Update progress bar
            if hasattr(self, "progress_bar") and self.progress_bar is not None:
                percent = result.get("percent", 0)
                self.progress_bar["value"] = percent
                
                # Update step label
                step = result.get("step", "processing")
                message = result.get("message", "Processing...")
                self.progress_step_var.set(f"Step: {step.capitalize()} - {message}")
                
                # Force update of UI
                self.update_idletasks()
            
            # Add to debug log
            if not hasattr(self, "last_debug_log"):
                self.last_debug_log = []
            self.last_debug_log.append(result)
            
            # Update debug window if it's open
            if hasattr(self, "debug_window") and self.debug_window is not None:
                self.update_debug_log([result])
            
            return
            
        # If we get here, analysis is complete
        # Remove the progress bar
        self.remove_progress_bar()
            
        # Extract proposals and log from the result
        if isinstance(result, dict) and "proposals" in result:
            proposals = result["proposals"]
            log = result.get("log", [])
            
            # Store the log for later viewing
            if not hasattr(self, "last_debug_log") or not isinstance(self.last_debug_log, list):
                self.last_debug_log = []
                
            # Add completion message to log
            completion_msg = {
                "status": "progress",
                "message": f"Analysis complete. Found {len(proposals)} potential text regions",
                "step": "complete",
                "percent": 100
            }
            self.last_debug_log.append(completion_msg)
            
            # Add any additional log entries
            if isinstance(log, list):
                self.last_debug_log.extend(log)
            else:
                self.last_debug_log.append(log)
            
            # Enable the debug button
            self.debug_btn["state"] = tk.NORMAL
        else:
            # For backward compatibility
            proposals = result
            
        # Ensure proposals is a list
        if not isinstance(proposals, list):
            proposals = []
            
        # Process the proposals to ensure they have the required fields
        processed_proposals = []
        for i, proposal in enumerate(proposals):
            # If proposal is missing required fields, add them
            if not isinstance(proposal, dict):
                continue
                
            # Ensure proposal has an id
            if 'id' not in proposal:
                proposal['id'] = i + 1
                
            # Ensure proposal has a score
            if 'score' not in proposal:
                proposal['score'] = 0.5  # Default score
                
            processed_proposals.append(proposal)
            
        self.current_proposals = processed_proposals
        
        # Clear and populate the proposals list
        self.proposals_list.delete(1, tk.END)  # Keep the default proposal
        for proposal in processed_proposals:
            try:
                item = ProposalItem(proposal)
                self.proposals_list.insert(tk.END, str(item))
            except Exception as e:
                print(f"Error adding proposal to list: {e}")
                continue
        
        # Generate and display visualization in a separate thread to prevent UI freezing
        def update_visualization():
            try:
                viz_image = self.pipeline.get_visualization(self.current_image_path, processed_proposals)
                # Schedule UI update on the main thread
                self.after(0, lambda: self.display_opencv_image(viz_image))
            except Exception as e:
                print(f"Error generating visualization: {e}")
        
        # Start visualization in a separate thread
        threading.Thread(target=update_visualization, daemon=True).start()
        
        # Re-enable UI
        self.analyze_btn["state"] = tk.NORMAL
        self.status_var.set(f"Analysis complete. Found {len(processed_proposals)} potential text regions")
    
    def create_progress_bar(self):
        """Create a progress bar to show analysis progress"""
        # Create a frame for the progress bar
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.grid(column=0, row=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Create a label for the current step
        self.progress_step_var = tk.StringVar(value="Step: Initializing")
        step_label = ttk.Label(self.progress_frame, textvariable=self.progress_step_var)
        step_label.pack(side=tk.TOP, anchor=tk.W, padx=5)
        
        # Create the progress bar
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5, pady=5)
        
    def remove_progress_bar(self):
        """Remove the progress bar when analysis is complete"""
        if hasattr(self, "progress_frame") and self.progress_frame is not None:
            # Set to 100% before removing for better visual feedback
            if hasattr(self, "progress_bar") and self.progress_bar is not None:
                self.progress_bar["value"] = 100
                self.progress_step_var.set("Step: Complete")
                self.update_idletasks()
                time.sleep(0.2)  # Short delay to show 100% before removing
                
            # Now remove the progress bar
            self.progress_frame.destroy()
            self.progress_frame = None
            self.progress_bar = None
        
    def show_last_debug_log(self):
        """Show the most recent debug log"""
        if self.last_debug_log:
            self.show_debug_log(self.last_debug_log)
        else:
            messagebox.showinfo("Debug Log", "No debug information available yet.")
        
    def show_debug_log(self, log_entries):
        """Display debug log in a new window"""
        # Check if debug window already exists
        if hasattr(self, 'debug_window') and self.debug_window is not None:
            try:
                # If window exists, just update it
                self.update_debug_log(log_entries)
                self.debug_window.focus_set()
                return
            except tk.TclError:
                # Window was closed, create a new one
                pass
                
        # Create new debug window
        self.debug_window = tk.Toplevel(self.master)
        self.debug_window.title("Analysis Debug Log")
        self.debug_window.geometry("800x500")
        self.debug_window.minsize(600, 400)
        
        # Store reference to log entries
        self.current_log_entries = log_entries.copy() if isinstance(log_entries, list) else [log_entries]
        
        # Set up protocol for window close
        self.debug_window.protocol("WM_DELETE_WINDOW", self.on_debug_window_close)
        
        # Create a frame for the log content
        main_frame = ttk.Frame(self.debug_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a header label
        header_label = ttk.Label(main_frame, text="Image Analysis Debug Information", 
                                font=("Helvetica", 12, "bold"))
        header_label.pack(pady=(0, 10), anchor=tk.W)
        
        # Create a text widget to display the log with custom styling
        self.log_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                            font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for different types of log entries
        self.log_text.tag_configure("header", foreground="blue", font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("timing", foreground="green", font=("Consolas", 10))
        self.log_text.tag_configure("error", foreground="red", font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("info", foreground="black", font=("Consolas", 10))
        self.log_text.tag_configure("progress", foreground="purple", font=("Consolas", 10, "italic"))
        self.log_text.tag_configure("highlight", foreground="#007F00", font=("Consolas", 10, "bold"))
        
        # Insert log entries
        self.update_debug_log(log_entries)
        
        # Add control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0), fill=tk.X)
        
        # Add auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_check = ttk.Checkbutton(
            button_frame, 
            text="Auto-scroll", 
            variable=self.auto_scroll_var
        )
        auto_scroll_check.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add clear button
        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_debug_log)
        clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add close button
        close_button = ttk.Button(button_frame, text="Close", command=self.on_debug_window_close)
        close_button.pack(side=tk.RIGHT)
        
        # Focus the window
        self.debug_window.focus_set()
        
    def update_debug_log(self, new_entries):
        """Update the debug log with new entries"""
        import time
        from datetime import datetime
        
        if not hasattr(self, 'log_text') or self.log_text is None:
            return
            
        # Make sure we have a list
        if not isinstance(new_entries, list):
            new_entries = [new_entries]
            
        # Add new entries to our stored entries
        if hasattr(self, 'current_log_entries'):
            for entry in new_entries:
                if entry not in self.current_log_entries:
                    self.current_log_entries.append(entry)
        
        # Enable editing temporarily
        self.log_text.config(state=tk.NORMAL)
        
        # Get current timestamp for log entries
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Process and insert new log entries
        for entry in new_entries:
            # Handle dictionary entries (structured logs)
            if isinstance(entry, dict):
                # Handle progress updates
                if "status" in entry:
                    if entry["status"] == "progress":
                        # Format progress updates with timestamp, percentage, step, and message
                        percent_str = f"[{entry.get('percent', '??')}%]" if "percent" in entry else ""
                        step_str = f"[{entry.get('step', 'unknown')}]" if "step" in entry else ""
                        
                        # Format the progress message with appropriate styling
                        progress_text = f"{timestamp} {percent_str} {step_str} {entry.get('message', '')}\n"
                        self.log_text.insert(tk.END, progress_text, "progress")
                        
                    elif entry["status"] == "error":
                        # Format error messages with timestamp
                        error_text = f"{timestamp} [ERROR] {entry.get('message', '')}\n"
                        self.log_text.insert(tk.END, error_text, "error")
                    
                    elif entry["status"] == "highlight":
                        # Format highlight messages with timestamp
                        highlight_text = f"{timestamp} [SUCCESS] {entry.get('message', '')}\n"
                        self.log_text.insert(tk.END, highlight_text, "highlight")
                        
                    else:
                        # Other status types
                        info_text = f"{timestamp} [{entry['status']}] {entry.get('message', '')}\n"
                        self.log_text.insert(tk.END, info_text, "info")
                else:
                    # Dictionary without status - just convert to string
                    self.log_text.insert(tk.END, f"{timestamp} {str(entry)}\n", "info")
            
            # Handle string entries (simple logs)
            elif isinstance(entry, str):
                if "error" in entry.lower() or "exception" in entry.lower() or "failed" in entry.lower():
                    self.log_text.insert(tk.END, f"{timestamp} {entry}\n", "error")
                    
                elif "seconds" in entry or ("time" in entry.lower() and "timing" not in entry.lower()):
                    self.log_text.insert(tk.END, f"{timestamp} {entry}\n", "timing")
                    
                elif any(keyword in entry.lower() for keyword in ["starting", "processing", "generating", 
                                                               "detecting", "analyzing", "initializing"]):
                    self.log_text.insert(tk.END, f"{timestamp} {entry}\n", "header")
                    
                elif "statistics" in entry.lower() or "found" in entry.lower() or "complete" in entry.lower():
                    self.log_text.insert(tk.END, f"{timestamp} {entry}\n", "highlight")
                    
                else:
                    self.log_text.insert(tk.END, f"{timestamp} {entry}\n", "info")
            
            # Handle any other type of entry
            else:
                self.log_text.insert(tk.END, f"{timestamp} {str(entry)}\n", "info")
        
        # Auto-scroll to the end if enabled
        if hasattr(self, 'auto_scroll_var') and self.auto_scroll_var.get():
            self.log_text.see(tk.END)
        
        # Make text widget read-only again
        self.log_text.config(state=tk.DISABLED)
        
    def clear_debug_log(self):
        """Clear the debug log text"""
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
            
    def on_debug_window_close(self):
        """Handle debug window closing"""
        if hasattr(self, 'debug_window') and self.debug_window is not None:
            self.debug_window.destroy()
            self.debug_window = None
            self.log_text = None
    
    def on_proposal_selected(self, event):
        """Handle selection of a region proposal"""
        try:
            # Get selected index
            selection = self.proposals_list.curselection()
            if not selection:
                return
                
            index = selection[0]
            selected_item = self.proposals_list.get(index)
            
            # Extract the ID from the selected item text
            proposal_id = selected_item.split(" ")[0]
            print(f"Selected proposal ID: {proposal_id}")
            
            # Find the corresponding proposal object
            selected_proposal_obj = None
            for proposal in self.current_proposals:
                if isinstance(proposal, dict) and str(proposal.get("id")) == str(proposal_id):
                    selected_proposal_obj = proposal
                    break
            
            # Store the proposal ID and object
            self.selected_proposal = proposal_id
            self.selected_proposal_obj = selected_proposal_obj
            
            # Enable export button
            self.export_btn["state"] = tk.NORMAL
            
            # Update preview
            self.update_preview()
            
            # Update status
            self.status_var.set(f"Selected region {proposal_id}")
        except Exception as e:
            print(f"Error selecting proposal: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error selecting proposal: {str(e)}")
    
    def on_text_changed(self, event=None):
        """Handle changes to the text input"""
        if self.selected_proposal:
            self.update_preview()
    
    def on_styling_changed(self, event=None):
        """Handle changes to styling options"""
        # Update outline button state
        if self.outline_var.get():
            self.outline_btn["state"] = tk.NORMAL
            self.outline_indicator["state"] = tk.NORMAL
        else:
            self.outline_btn["state"] = tk.DISABLED
            self.outline_indicator["state"] = tk.DISABLED
            
        # Update warp options state
        if self.warp_var.get():
            self.warp_type_combo["state"] = "readonly"
        else:
            self.warp_type_combo["state"] = tk.DISABLED
            
        # Update shadow options state
        if self.shadow_var.get():
            self.shadow_blur_var.set(self.shadow_blur_var.get())  # Trigger update
        
        # Trigger preview update
        self.update_preview()
        
    def update_preview(self):
        """Update the preview with current settings"""
        if not self.current_image_path or not self.selected_proposal:
            return
            
        # Get current text
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            # Display the image without text
            self.display_image(self.current_image_path)
            return
            
        # Disable UI during rendering
        self.toggle_ui_state(False)
        
        # Get styling parameters
        styling = {
            "font": self.font_var.get(),
            "font_size": self.size_var.get(),
            "color": self.text_color,
            "outline_color": self.outline_color if self.outline_var.get() else None,
            "outline_width": self.outline_width_var.get(),
            "shadow": self.shadow_var.get(),
            "shadow_blur": self.shadow_blur_var.get(),
            "opacity": self.opacity_var.get(),
            "alignment": self.align_var.get().lower(),
            "warp": self.warp_type_var.get().lower() if self.warp_var.get() else None,
            "warp_strength": self.warp_strength_var.get() / 100.0,
            "blend_mode": self.blend_mode_var.get().lower()
        }
        
        try:
            # Render preview using the pipeline
            print(f"Rendering preview with proposal ID: {self.selected_proposal}")
            print(f"Text: '{text}'")
            print(f"Styling: {styling}")
            
            rendered_image = self.pipeline.render_preview(
                self.current_image_path,
                self.selected_proposal,
                text,
                styling
            )
            
            # Convert PIL image to cv2 format for display
            cv_image = cv2.cvtColor(np.array(rendered_image), cv2.COLOR_RGBA2BGRA)
            
            # Display the image
            self.display_cv_image(cv_image)
            
        except Exception as e:
            print(f"Error rendering preview: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to displaying original image
            self.display_image(self.current_image_path)
        finally:
            # Re-enable UI
            self.toggle_ui_state(True)
    
    def on_preview_rendered(self, rendered_image):
        """Handle completion of preview rendering"""
        self.preview_image = rendered_image
        
        # Convert PIL Image to Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(rendered_image)
        
        # Store a reference to prevent garbage collection
        self.tk_image = tk_image
        
        # Display the preview
        self.preview_label.config(image=tk_image, text="")
        
        # Update canvas scrollregion
        self.preview_canvas.config(scrollregion=self.preview_canvas.bbox("all"))
        
        self.status_var.set("Preview ready")
    
    def check_tasks(self):
        """Periodically check for completed background tasks"""
        if self.current_task:
            try:
                # Check if task is still running
                if hasattr(self.current_task, 'is_alive') and self.current_task.is_alive():
                    # Force UI update to keep it responsive
                    self.update_idletasks()
                    
                    result = self.current_task.get_result()
                    if result:
                        success, data = result
                        
                        if isinstance(self.current_task, ImageAnalysisTask):
                            # Handle both progress updates and final results
                            if success:
                                # This is the final result - analysis is complete
                                # Process the actual results
                                self.on_analysis_complete(data)
                                
                                # Clear the task
                                self.current_task = None
                            else:
                                # Check if it's a progress update or an error
                                if isinstance(data, dict) and "status" in data:
                                    if data["status"] == "progress":
                                        # Just a progress update, don't clear the task
                                        self.on_analysis_complete(data)
                                        
                                        # Schedule next check
                                        self.after(100, self.check_tasks)
                                    else:
                                        # Error or other status
                                        self.status_var.set(f"Analysis status: {data.get('message', 'Unknown')}")
                                        self.after(100, self.check_tasks)
                                else:
                                    # Handle error
                                    self.status_var.set(f"Analysis error: {data}")
                                    self.remove_progress_bar()
                                    self.analyze_btn["state"] = tk.NORMAL
                                    self.current_task = None
                                    messagebox.showerror("Analysis Error", str(data))
                        elif isinstance(self.current_task, RenderPreviewTask):
                            if success:
                                # Preview rendering is complete
                                self.on_preview_rendered(data)
                                self.current_task = None
                            else:
                                # Error in preview rendering
                                self.status_var.set(f"Preview error: {data}")
                                self.current_task = None
                                messagebox.showerror("Preview Error", str(data))
                        
                        # If task is still running, schedule next check
                        if self.current_task:
                            self.after(100, self.check_tasks)
                    else:
                        # No result yet, schedule next check
                        self.after(100, self.check_tasks)
                else:
                    # Task is no longer running but didn't report a result
                    # This is likely an error condition
                    self.status_var.set("Task completed without result, using default region")
                    self.remove_progress_bar()
                    self.analyze_btn["state"] = tk.NORMAL
                    
                    # Make sure we have a default proposal
                    if not self.current_proposals or len(self.current_proposals) == 0:
                        self.add_default_proposal()
                    
                    # Select the default proposal
                    if self.proposals_list.size() > 0 and not self.selected_proposal:
                        self.proposals_list.selection_set(0)
                        self.on_proposal_selected(None)
                    
                    self.current_task = None
            except Exception as e:
                # Handle any exceptions in the task checking
                print(f"Error checking task: {e}")
                self.status_var.set(f"Error checking task: {str(e)}")
                self.remove_progress_bar()
                self.analyze_btn["state"] = tk.NORMAL
                
                # Make sure we have a default proposal
                if not self.current_proposals or len(self.current_proposals) == 0:
                    self.add_default_proposal()
                
                # Select the default proposal
                if self.proposals_list.size() > 0 and not self.selected_proposal:
                    self.proposals_list.selection_set(0)
                    self.on_proposal_selected(None)
                
                self.current_task = None
    
    def on_export(self):
        """Handle export button click"""
        if not self.selected_proposal or not self.current_image_path:
            messagebox.showerror("Export Error", "No image or region selected")
            return
            
        # Get text from input field
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Export Error", "No text entered")
            return
            
        # Get export path
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG Images", "*.png"), ("JPEG Images", "*.jpg"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
            
        # Show export options dialog
        self.show_export_options(text, file_path)
    
    
    def show_export_options(self, text, output_path):
        """Show dialog with export options"""
        # Create a dialog window
        dialog = tk.Toplevel(self)
        dialog.title("Export Options")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        
        # Create a frame for the options
        options_frame = ttk.Frame(dialog, padding=10)
        options_frame.pack(fill=tk.BOTH, expand=True)
        
        # Quality options
        ttk.Label(options_frame, text="Quality:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        quality_var = tk.IntVar(value=95)
        quality_spin = ttk.Spinbox(options_frame, from_=1, to=100, textvariable=quality_var, width=5)
        quality_spin.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Resolution options
        ttk.Label(options_frame, text="Resolution:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        resolution_var = tk.StringVar(value="Original")
        resolution_combo = ttk.Combobox(options_frame, textvariable=resolution_var, state="readonly")
        resolution_combo["values"] = ("Original", "1080p", "4K", "Custom")
        resolution_combo.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        
        # DPI options
        ttk.Label(options_frame, text="DPI:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        dpi_var = tk.IntVar(value=300)
        dpi_spin = ttk.Spinbox(options_frame, from_=72, to=600, textvariable=dpi_var, width=5)
        dpi_spin.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        export_btn = ttk.Button(
            button_frame, 
            text="Export", 
            command=lambda: self.do_export(
                text, 
                output_path, 
                quality_var.get(), 
                dpi_var.get(),
                dialog
            )
        )
        export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Center the dialog on the parent window
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def do_export(self, text, output_path, quality, dpi, dialog=None):
        """Export the rendered image with the given options"""
        try:
            # Show a progress dialog
            progress_dialog = tk.Toplevel(self)
            progress_dialog.title("Exporting...")
            progress_dialog.geometry("300x100")
            progress_dialog.resizable(False, False)
            progress_dialog.transient(self)
            progress_dialog.grab_set()
            
            # Add a progress bar
            ttk.Label(progress_dialog, text="Exporting image...").pack(pady=10)
            progress = ttk.Progressbar(progress_dialog, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
            progress.pack(pady=10)
            progress.start()
            
            # Center the dialog
            progress_dialog.update_idletasks()
            x = self.winfo_x() + (self.winfo_width() - progress_dialog.winfo_width()) // 2
            y = self.winfo_y() + (self.winfo_height() - progress_dialog.winfo_height()) // 2
            progress_dialog.geometry(f"+{x}+{y}")
            
            # Find the selected proposal object by ID
            selected_proposal_obj = None
            for proposal in self.current_proposals:
                if isinstance(proposal, dict) and str(proposal.get("id")) == str(self.selected_proposal):
                    selected_proposal_obj = proposal
                    break
            
            if not selected_proposal_obj:
                # Create a default centered region if no proposal is found
                img = cv2.imread(self.current_image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    selected_proposal_obj = {
                        "id": "default_center",
                        "x": int(width * 0.2),
                        "y": int(height * 0.2),
                        "width": int(width * 0.6),
                        "height": int(height * 0.6),
                        "score": 1.0
                    }
            
            # Create a thread to perform the export
            def export_thread():
                try:
                    # Get the region from the selected proposal
                    region = {
                        "x": selected_proposal_obj["x"],
                        "y": selected_proposal_obj["y"],
                        "width": selected_proposal_obj["width"],
                        "height": selected_proposal_obj["height"],
                        "rotation": selected_proposal_obj.get("rotation", 0)
                    }
                    
                    # Get styling parameters
                    styling = {
                        "font": self.font_var.get(),
                        "font_size": self.size_var.get(),
                        "color": self.text_color,
                        "outline_color": self.outline_color if self.outline_var.get() else None,
                        "outline_width": self.outline_width_var.get(),
                        "shadow": self.shadow_var.get(),
                        "shadow_blur": self.shadow_blur_var.get(),
                        "opacity": self.opacity_var.get(),
                        "alignment": self.align_var.get().lower(),
                        "warp": self.warp_type_var.get().lower() if self.warp_var.get() else None,
                        "warp_strength": self.warp_strength_var.get() / 100.0,
                        "blend_mode": self.blend_mode_var.get().lower()
                    }
                    
                    # Render and save
                    output_path = self.pipeline.render_final(
                        self.current_image_path,
                        text,
                        region,
                        styling,
                        output_path,
                        quality
                    )
                    
                    # Close the progress dialog
                    self.after(0, progress_dialog.destroy)
                    
                    # Show success message
                    self.after(0, lambda: messagebox.showinfo(
                        "Export Complete", 
                        f"Image exported successfully to:\n{output_path}"
                    ))
                    
                    # Close the options dialog if it exists
                    if dialog:
                        self.after(0, dialog.destroy)
                        
                except Exception as e:
                    # Close the progress dialog
                    self.after(0, progress_dialog.destroy)
                    
                    # Show error message
                    self.after(0, lambda: messagebox.showerror(
                        "Export Error", 
                        f"Error exporting image: {str(e)}"
                    ))
                    print(f"Export error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Start the export thread
            threading.Thread(target=export_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error setting up export: {str(e)}")
            print(f"Export setup error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_export_complete(self, progress_dialog, output_path):
        """Handle completion of export process"""
        progress_dialog.destroy()
        messagebox.showinfo("Export Complete", f"Image exported successfully to:\n{output_path}")
        self.status_var.set(f"Image exported to {output_path}")
        
    def _on_export_error(self, progress_dialog, error_message):
        """Handle export error"""
        progress_dialog.destroy()
        messagebox.showerror("Export Error", f"Failed to export image: {error_message}")
        self.status_var.set("Export failed")
    
    def display_image(self, image_path):
        """Display an image from file path"""
        # Open the image with PIL
        pil_image = Image.open(image_path)
        
        # Convert to Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Store a reference to prevent garbage collection
        self.tk_image = tk_image
        
        # Display the image
        self.preview_label.config(image=tk_image, text="")
        
        # Update canvas scrollregion
        self.preview_canvas.config(scrollregion=self.preview_canvas.bbox("all"))
    
    def display_cv_image(self, cv_image):
        """Display an OpenCV image"""
        # Convert OpenCV image (BGR) to RGB
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv_image_rgb)
        
        # Convert to Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Store a reference to prevent garbage collection
        self.tk_image = tk_image
        
        # Display the image
        self.preview_label.config(image=tk_image, text="")
        
        # Update canvas scrollregion
        self.preview_canvas.config(scrollregion=self.preview_canvas.bbox("all"))
    
    def toggle_ui_state(self, enabled):
        """Toggle the state of UI elements"""
        self.analyze_btn["state"] = tk.NORMAL if enabled else tk.DISABLED
        self.export_btn["state"] = tk.NORMAL if enabled and self.selected_proposal else tk.DISABLED
        self.text_input["state"] = tk.NORMAL if enabled else tk.DISABLED
        self.font_combo["state"] = "readonly" if enabled else tk.DISABLED
        self.size_var.set(self.size_var.get())  # Trigger update
        
        # Fix color button reference
        # self.color_btn["state"] = tk.NORMAL if enabled else tk.DISABLED
        
        self.outline_var.set(self.outline_var.get())  # Trigger update
        if hasattr(self, 'outline_btn'):
            if enabled and self.outline_var.get():
                self.outline_btn["state"] = tk.NORMAL
            else:
                self.outline_btn["state"] = tk.DISABLED
                
        self.opacity_var.set(self.opacity_var.get())  # Trigger update
        self.shadow_var.set(self.shadow_var.get())  # Trigger update
        self.shadow_blur_var.set(self.shadow_blur_var.get())  # Trigger update
        self.blend_mode_combo["state"] = "readonly" if enabled else tk.DISABLED
        self.warp_var.set(self.warp_var.get())  # Trigger update
        self.warp_type_combo["state"] = "readonly" if enabled and self.warp_var.get() else tk.DISABLED
        self.warp_strength_var.set(self.warp_strength_var.get())  # Trigger update
        self.align_var.set(self.align_var.get())  # Trigger update
        self.dpi_combo["state"] = "readonly" if enabled else tk.DISABLED

    def on_color_select(self):
        """Handle text color selection"""
        color = colorchooser.askcolor(initialcolor=self.text_color, title="Select Text Color")
        if color[1]:  # color is ((r,g,b), hexcode)
            self.text_color = color[1]
            self.color_indicator.config(bg=self.text_color)
            self.update_preview()
            
    def on_outline_color_select(self):
        """Handle outline color selection"""
        color = colorchooser.askcolor(initialcolor=self.outline_color, title="Select Outline Color")
        if color[1]:  # color is ((r,g,b), hexcode)
            self.outline_color = color[1]
            self.outline_indicator.config(bg=self.outline_color)
            self.update_preview()
    
def run_application():
    """Run the application"""
    root = tk.Tk()
    root.title("Intelligent Text Placement")
    root.geometry("1200x800")
    
    # Create main window
    window = MainWindow(root)
    
    # Start the task checker
    window.check_tasks()
    
    # Start the application
    root.mainloop()
