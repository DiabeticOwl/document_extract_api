import sys

from core.utils import noise_reduction, adaptive_thresholding, deskew
from pathlib import Path
from PIL import Image

# --- SCRIPT OVERVIEW ---
# This script serves as a command-line utility for manually testing and visualizing
# the effects of various image preprocessing techniques on a single image. Its primary
# purpose is to allow a developer to see how each transformation (noise reduction,
# thresholding, deskewing) alters an image before integrating these steps into an
# automated OCR pipeline.
#
#
# --- USAGE ---
# Run from the command line, passing the path to an image as an argument:
# > python manual_preprocessing_test.py /path/to/your/image.jpg


def main():
    """
    The main execution function for the script.
    It handles command-line arguments, loads the image, calls the processing
    functions, and saves the output files.
    """
    # --- 1. Get and Validate Input Image Path ---
    if len(sys.argv) != 2:
        print("Usage: python manual_preprocessing_test.py <path_to_image>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.is_file():
        print(f"Error: File not found at '{input_path}'")
        sys.exit(1)

    print(f"Processing image: {input_path}")

    # --- 2. Load the Image ---
    try:
        original_image = Image.open(input_path)
    except Exception as e:
        print(f"Error: Could not open or read the image file. Reason: {e}")
        sys.exit(1)

    # --- 3. Apply Transformations ---
    print("Applying Noise Reduction (Median Blur)...")
    noise_reduced_image = noise_reduction(original_image)

    print("Applying Adaptive Thresholding...")
    thresholded_image = adaptive_thresholding(original_image)

    print("Applying Deskewing...")
    deskewed_image = deskew(original_image)

    # --- 4. Save the Results ---
    # The output files are saved in the same directory as the input image
    # with descriptive names for easy comparison.
    output_dir = input_path.parent
    original_output_path = output_dir / "original.png"
    noise_output_path = output_dir / "processed_noise_reduction.png"
    threshold_output_path = output_dir / "processed_adaptive_threshold.png"
    deskew_output_path = output_dir / "processed_deskewed.png"

    print(f"\nSaving original image to: {original_output_path}")
    original_image.save(original_output_path)

    print(f"Saving noise-reduced image to: {noise_output_path}")
    noise_reduced_image.save(noise_output_path)

    print(f"Saving thresholded image to: {threshold_output_path}")
    thresholded_image.save(threshold_output_path)

    print(f"Saving deskewed image to: {deskew_output_path}")
    deskewed_image.save(deskew_output_path)

    print("\nProcessing complete. Check the output files in the source directory.")


if __name__ == "__main__":
    main()
