"""
Main Window UI Module for Intelligent Text Placement (Tkinter version)
"""
import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, scrolledtext
from tkinter import messagebox
import threading
import queue
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
from typing import Dict, List, Any, Optional, Tuple

from workflow.pipeline import TextPlacementPipeline


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
        self.watchdog_active = False
    
    def run(self):
        """Execute the task in a separate thread"""
        try:
            # Simple debug callback to track progress
            def debug_callback(data):
                if isinstance(data, dict) and "status" in data:
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
                
                # Send completion update
                debug_callback({
                    "status": "progress", 
                    "message": "Analysis complete", 
                    "step": "complete", 
                    "percent": 100
                })
                
                # Return the result
                self.queue.put((True, result))
                
            except Exception as e:
                # Log the error
                error_msg = f"Error processing image: {str(e)}"
                print(f"ERROR: {error_msg}")
                
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
                self.queue.put((True, []))
                
        except Exception as e:
            # Catch any other exceptions
            error_msg = f"Unexpected error in analysis task: {str(e)}"
            print(f"CRITICAL ERROR: {error_msg}")
            self.queue.put((True, []))
    
    def get_result(self):
        """Get the latest result or progress update"""
        if not self.queue.empty():
            return self.queue.get()
        
        # If there are progress updates but nothing in the queue,
        # return the latest progress update
        if self.progress_updates:
            return (False, self.progress_updates[-1])
        
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
    """Class for storing proposal information"""
    
    def __init__(self, proposal: Dict[str, Any]):
        self.proposal = proposal
        self.id = proposal['id']
        self.score = proposal['score']
        
    def __str__(self):
        return f"Region {self.id} (Score: {self.score:.2f})"


class MainWindow(tk.Frame):
    """Main application window for Intelligent Text Placement (Tkinter version)"""
    
    def __init__(self, master):
        super().__init__(master)
        
        # Store reference to master
        self.master = master
        
        # Initialize the pipeline with debug mode enabled
        self.pipeline = TextPlacementPipeline(dpi=300, debug=True)
        
        # Initialize state variables
        self.image_path = None
        self.current_proposals = []
        self.selected_proposal = None
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
        self.font_combo['values'] = ("Arial", "Times New Roman", "Courier New", "Verdana", "Georgia")
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
        self.color_btn = ttk.Button(styling_frame, text="      ", command=self.on_color_select)
        self.color_btn.grid(column=1, row=2, sticky=tk.W, padx=5, pady=2)
        
        # Create a colored canvas behind the button
        self.color_indicator = tk.Canvas(self.color_btn, width=20, height=20, bg=self.text_color)
        self.color_indicator.place(x=5, y=2)
        
        # Outline
        self.outline_var = tk.BooleanVar(value=False)
        outline_check = ttk.Checkbutton(styling_frame, text="Outline", variable=self.outline_var, 
                                       command=self.on_styling_changed)
        outline_check.grid(column=0, row=3, sticky=tk.W, padx=5, pady=2)
        
        self.outline_btn = ttk.Button(styling_frame, text="      ", command=self.on_outline_color_select, 
                                     state=tk.DISABLED)
        self.outline_btn.grid(column=1, row=3, sticky=tk.W, padx=5, pady=2)
        
        # Create a colored canvas for outline
        self.outline_indicator = tk.Canvas(self.outline_btn, width=20, height=20, bg=self.outline_color)
        self.outline_indicator.place(x=5, y=2)
        
        # Shadow
        self.shadow_var = tk.BooleanVar(value=False)
        shadow_check = ttk.Checkbutton(styling_frame, text="Shadow", variable=self.shadow_var, 
                                      command=self.on_styling_changed)
        shadow_check.grid(column=0, row=4, sticky=tk.W, padx=5, pady=2)
        
        # Alignment
        ttk.Label(styling_frame, text="Alignment:").grid(column=0, row=5, sticky=tk.W, padx=5, pady=2)
        self.align_var = tk.StringVar(value="center")
        align_combo = ttk.Combobox(styling_frame, textvariable=self.align_var, state="readonly")
        align_combo['values'] = ("Left", "Center", "Right")
        align_combo.current(1)  # Center by default
        align_combo.grid(column=1, row=5, sticky=(tk.W, tk.E), padx=5, pady=2)
        align_combo.bind("<<ComboboxSelected>>", self.on_styling_changed)
        
        # Export section
        export_frame = ttk.LabelFrame(parent, text="Export")
        export_frame.grid(column=0, row=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        export_frame.columnconfigure(0, weight=1)
        export_frame.columnconfigure(1, weight=1)
        
        # Resolution selection
        ttk.Label(export_frame, text="Resolution (DPI):").grid(column=0, row=0, sticky=tk.W, padx=5, pady=2)
        self.dpi_var = tk.StringVar(value="300 (High Print)")
        dpi_combo = ttk.Combobox(export_frame, textvariable=self.dpi_var, state="readonly")
        dpi_combo['values'] = ("72 (Screen)", "150 (Low Print)", "300 (High Print)", "600 (Professional)")
        dpi_combo.current(2)  # 300 DPI by default
        dpi_combo.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Export button
        self.export_btn = ttk.Button(export_frame, text="Export Image", 
                                   command=self.on_export, state=tk.DISABLED)
        self.export_btn.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
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
            self.image_path = file_path
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
        if not self.image_path:
            return
            
        # Get text length for better region sizing
        text = self.text_input.get("1.0", tk.END)
        text_length = len(text.strip()) if text.strip() else 20
        
        # Disable UI during analysis
        self.analyze_btn["state"] = tk.DISABLED
        self.status_var.set("Analyzing image...")
        
        # Run analysis in a background thread
        self.current_task = ImageAnalysisTask(self.pipeline, self.image_path, text_length)
        self.current_task.start()
    
    def on_analysis_complete(self, result):
        """Handle completion of image analysis"""
        # Check if we received progress update or final result
        if isinstance(result, dict) and "status" in result and result["status"] == "progress":
            # Update status with progress message
            self.status_var.set(result["message"])
            
            # Update progress bar if percent is provided
            if "percent" in result:
                if not hasattr(self, "progress_bar") or self.progress_bar is None:
                    # Create a progress bar if it doesn't exist
                    self.create_progress_bar()
                
                # Update the progress bar value
                self.progress_bar["value"] = result["percent"]
                
                # Force the UI to update
                self.update_idletasks()
                
                # Update the step label if provided
                if "step" in result:
                    step_name = result["step"].capitalize()
                    self.progress_step_var.set(f"Step: {step_name}")
            
            # Add to debug log if it exists
            if hasattr(self, "last_debug_log"):
                if not isinstance(self.last_debug_log, list):
                    self.last_debug_log = [self.last_debug_log]
                self.last_debug_log.append(result)
            else:
                self.last_debug_log = [result]
                
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
            
            # Update debug window if it's open, otherwise show it
            if hasattr(self, "debug_window") and self.debug_window is not None:
                self.update_debug_log([completion_msg])
            else:
                self.show_debug_log(self.last_debug_log)
        else:
            # For backward compatibility
            proposals = result
            
        self.current_proposals = proposals
        
        # Clear and populate the proposals list
        self.proposals_list.delete(0, tk.END)
        for proposal in proposals:
            item = ProposalItem(proposal)
            self.proposals_list.insert(tk.END, str(item))
        
        # Generate and display visualization
        viz_image = self.pipeline.get_visualization(self.image_path, proposals)
        self.display_opencv_image(viz_image)
        
        # Re-enable UI
        self.analyze_btn["state"] = tk.NORMAL
        self.status_var.set(f"Analysis complete. Found {len(proposals)} potential text regions")
        
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
        selection = self.proposals_list.curselection()
        if not selection:
            return
            
        index = selection[0]
        if index < 0 or index >= len(self.current_proposals):
            return
            
        self.selected_proposal = self.current_proposals[index]
        
        # Update the preview with this region
        self.update_preview()
        
        # Enable export button
        self.export_btn["state"] = tk.NORMAL
    
    def on_text_changed(self, event=None):
        """Handle changes to the text input"""
        if self.selected_proposal:
            self.update_preview()
    
    def on_styling_changed(self, event=None):
        """Handle changes to styling options"""
        # Update UI state for outline color
        outline_enabled = self.outline_var.get()
        self.outline_btn["state"] = tk.NORMAL if outline_enabled else tk.DISABLED
        
        # Update preview if we have a selection
        if self.selected_proposal:
            self.update_preview()
    
    def on_color_select(self):
        """Handle text color selection"""
        color = colorchooser.askcolor(self.text_color, title="Select Text Color")
        if color[1]:  # color is a tuple (RGB, hex)
            self.text_color = color[1]
            self.color_indicator.config(bg=self.text_color)
            
            if self.selected_proposal:
                self.update_preview()
    
    def on_outline_color_select(self):
        """Handle outline color selection"""
        color = colorchooser.askcolor(self.outline_color, title="Select Outline Color")
        if color[1]:  # color is a tuple (RGB, hex)
            self.outline_color = color[1]
            self.outline_indicator.config(bg=self.outline_color)
            
            if self.selected_proposal:
                self.update_preview()
    
    def update_preview(self):
        """Update the preview with current settings"""
        if not self.selected_proposal or not self.image_path:
            return
            
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            return
            
        # Collect styling parameters
        styling = {
            "font": self.font_var.get(),
            "font_size": self.size_var.get(),
            "color": self.text_color,
            "outline_color": self.outline_color if self.outline_var.get() else None,
            "shadow": self.shadow_var.get(),
            "alignment": self.align_var.get().lower()
        }
        
        # Disable UI during rendering
        self.status_var.set("Rendering preview...")
        
        # Run rendering in a background thread
        self.current_task = RenderPreviewTask(
            self.pipeline, 
            self.image_path,
            self.selected_proposal["id"],
            text,
            styling
        )
        self.current_task.start()
    
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
            # Check if task is still running
            if not hasattr(self.current_task, 'is_alive') or self.current_task.is_alive():
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
                            
                            # Make sure the progress bar is removed and status is updated
                            self.remove_progress_bar()
                            
                            if isinstance(data, dict) and "proposals" in data:
                                self.status_var.set(f"Analysis complete. Found {len(data['proposals'])} potential text regions")
                            else:
                                self.status_var.set("Analysis complete")
                                
                            # Force UI update
                            self.update_idletasks()
                            
                            # Clear the task
                            self.current_task = None
                        else:
                            # Check if it's a progress update or an error
                            if isinstance(data, dict) and "status" in data:
                                if data["status"] == "progress":
                                    # Just a progress update, don't clear the task
                                    self.on_analysis_complete(data)
                                    
                                    # Update the progress bar if percent is provided
                                    if "percent" in data and hasattr(self, "progress_bar") and self.progress_bar is not None:
                                        self.progress_bar["value"] = data["percent"]
                                        # Force update the UI
                                        self.update_idletasks()
                                        
                                        # Update step label if provided
                                        if "step" in data and hasattr(self, "progress_step_var"):
                                            step_name = data["step"].capitalize()
                                            self.progress_step_var.set(f"Step: {step_name}")
                                            
                                        # Force update the UI
                                        self.update_idletasks()
                                        
                                        # If we're at a high percentage but not complete, ensure we keep moving
                                        if data["percent"] > 80 and data["percent"] < 100:
                                            # Schedule a forced update if we stay at this percentage too long
                                            self.after(3000, lambda: self._check_progress_stalled(data["percent"]))
                                else:
                                    # Error occurred
                                    error_msg = data.get("message", "Unknown error")
                                    self.status_var.set(f"Error: {error_msg}")
                                    
                                    # Show debug log if available
                                    if "log" in data:
                                        self.show_debug_log(data["log"])
                                        
                                    self.current_task = None
                            else:
                                # Old format error
                                self.status_var.set(f"Error: {data}")
                                self.current_task = None
            else:
                # Task has completed but didn't properly clean up
                self.current_task = None
                
        # Schedule next check (more frequent checks for better responsiveness)
        self.after(50, self.check_tasks)
    
    def _check_progress_stalled(self, last_percent):
        """Check if progress has stalled at a high percentage and force completion if needed"""
        if self.current_task and hasattr(self, "progress_bar") and self.progress_bar is not None:
            current_percent = self.progress_bar["value"]
            # If we're still at the same percentage after 3 seconds, force progress
            if current_percent == last_percent and current_percent > 80 and current_percent < 100:
                # Force progress to move forward
                new_percent = min(100, current_percent + 5)
                self.progress_bar["value"] = new_percent
                self.progress_step_var.set("Step: Finalizing")
                self.update_idletasks()
                # Check again in 2 seconds if still not complete
                if new_percent < 100:
                    self.after(2000, lambda: self._check_progress_stalled(new_percent))
    
    def on_export(self):
        """Handle export button click"""
        if not self.preview_image or not self.selected_proposal:
            return
            
        # Get export path
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG Images", "*.png"), ("JPEG Images", "*.jpg"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        # Get DPI setting
        dpi_text = self.dpi_var.get()
        if "72" in dpi_text:
            dpi = 72
        elif "150" in dpi_text:
            dpi = 150
        elif "600" in dpi_text:
            dpi = 600
        else:
            dpi = 300
            
        # Update pipeline DPI
        self.pipeline.text_renderer.dpi = dpi
        
        # Get text and styling
        text = self.text_input.get("1.0", tk.END).strip()
        styling = {
            "font": self.font_var.get(),
            "font_size": self.size_var.get(),
            "color": self.text_color,
            "outline_color": self.outline_color if self.outline_var.get() else None,
            "shadow": self.shadow_var.get(),
            "alignment": self.align_var.get().lower()
        }
        
        # Render and save final image
        try:
            self.status_var.set(f"Exporting image to {file_path}...")
            
            output_path = self.pipeline.render_final(
                self.image_path,
                text,
                self.selected_proposal["region"],
                styling,
                file_path
            )
            
            self.status_var.set(f"Image exported to {output_path}")
        except Exception as e:
            self.status_var.set(f"Error during export: {str(e)}")
            messagebox.showerror("Export Error", str(e))
    
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
    
    def display_opencv_image(self, cv_image):
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
