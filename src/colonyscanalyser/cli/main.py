"""
Main CLI entry point for ColonyScanalyser.

This module provides the main command line interface entry point for the
ColonyScanalyser tool, handling argument parsing and delegating to the
appropriate analysis functions.
"""


def main():
    """
    Main entry point for the ColonyScanalyser CLI.

    Basic implementation for testing with frames.
    """
    from pathlib import Path

    from ..core import ImageFileCollection
    from .args import create_parser

    parser = create_parser()
    args = parser.parse_args()

    print(f"ColonyScanalyser analyzing: {args.path}")

    # Basic test to load images
    try:
        frames_path = Path(args.path)
        print(f"Looking for images in: {frames_path}")

        image_files = ImageFileCollection.from_path(
            frames_path, ["png", "jpg", "jpeg", "tif", "tiff"], cache_images=False
        )
        print(f"Found {image_files.count} images")

        if image_files.count > 0:
            with image_files.items[0] as img:
                print(f"First image shape: {img.image.shape}")
                print(f"First image file: {img.file_path.name}")
                print(f"Timestamp: {img.timestamp}")
        else:
            print("No images found!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
