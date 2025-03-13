"""
Intelligent Text Placement and Rendering Tool

Main entry point for the application.
"""
import sys
import os
import argparse
from ui.main_window import run_application


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Intelligent Text Placement and Rendering Tool"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run in headless mode (without GUI)"
    )
    parser.add_argument(
        "--image", 
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--text", 
        type=str,
        help="Text to place on the image"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Path for output image"
    )
    parser.add_argument(
        "--dpi", 
        type=int,
        default=300,
        help="Output resolution in DPI (default: 300)"
    )
    
    return parser.parse_args()


def run_headless(args):
    """Run in headless mode (CLI)"""
    from workflow.pipeline import TextPlacementPipeline
    
    # Validate arguments
    if not args.image or not os.path.isfile(args.image):
        print(f"Error: Input image not found: {args.image}")
        return 1
        
    if not args.text:
        print("Error: Text is required")
        return 1
        
    if not args.output:
        print("Error: Output path is required")
        return 1
        
    # Initialize pipeline
    pipeline = TextPlacementPipeline(dpi=args.dpi)
    
    # Process the image
    print(f"Analyzing image: {args.image}")
    proposals = pipeline.process_image(args.image, len(args.text))
    
    if not proposals:
        print("Error: No suitable regions found for text placement")
        return 1
        
    # Select the best proposal
    best_proposal = proposals[0]
    print(f"Selected region (score: {best_proposal['score']:.2f})")
    
    # Render the text
    print(f"Rendering text with {args.dpi} DPI...")
    try:
        output_path = pipeline.render_final(
            args.image,
            args.text,
            best_proposal["region"],
            best_proposal["styling"],
            args.output
        )
        print(f"Output saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error during rendering: {str(e)}")
        return 1


def main():
    """Application entry point"""
    args = parse_arguments()
    
    if args.headless:
        return run_headless(args)
    else:
        # Run GUI application
        run_application()
        return 0


if __name__ == "__main__":
    sys.exit(main())
