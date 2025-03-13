"""
Export options dialog for TextTool
"""
import tkinter as tk
from tkinter import ttk

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
