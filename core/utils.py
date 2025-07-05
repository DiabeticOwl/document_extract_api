import cv2
import numpy as np

from pathlib import Path
from PIL import Image

# --- PREPROCESSING UTILITY FUNCTIONS ---
# This module contains standalone functions for image preprocessing tasks.
# These functions are designed to be imported and used by other parts of the
# application, such as the OCR module or testing scripts.


def noise_reduction(image: Image.Image) -> Image.Image:
    """
    Applies a Median Blur filter to an image to remove salt-and-pepper noise.

    This is a common type of noise found in scanned documents, appearing as random
    black and white speckles. The median filter replaces each pixel's value with
    the median value of its neighbors, which is highly effective at removing
    these speckles without significantly blurring the sharp edges of text.

    Args:
        image (Image.Image): The input PIL Image.

    Returns:
        Image.Image: The processed PIL Image with noise reduced.
    """
    # Convert the PIL Image to a NumPy array in grayscale format, which is what
    # most OpenCV functions expect.
    cv_image = np.array(image.convert('L'))

    # Apply a 3x3 median blur. The kernel size (3) must be an odd number.
    # A small kernel is chosen to target speckle noise while preserving text clarity.
    denoised_image = cv2.medianBlur(cv_image, 3)

    # Convert the processed NumPy array back to a PIL Image for saving.
    return Image.fromarray(denoised_image)


def adaptive_thresholding(image: Image.Image) -> Image.Image:
    """
    Applies adaptive thresholding to create a clean, high-contrast,
    black-and-white image.

    Unlike global thresholding which uses one brightness value for the whole image,
    adaptive thresholding calculates a different threshold for small local regions.
    This makes it extremely effective for images with uneven lighting or shadows.

    Args:
        image (Image.Image): The input PIL Image.

    Returns:
        Image.Image: The processed, high-contrast PIL Image.
    """
    cv_image = np.array(image.convert('L'))

    # cv2.adaptiveThreshold parameters:
    # - src: The source grayscale image.
    # - maxValue (255): The value assigned to pixels that exceed the threshold.
    # - adaptiveMethod: We use GAUSSIAN_C, which calculates the threshold for a
    #   pixel based on a weighted sum of its neighbors (a Gaussian window).
    # - thresholdType: THRESH_BINARY means pixels above the threshold become white (255),
    #   and those below become black (0).
    # - blockSize (11): The size of the neighborhood area to calculate the threshold.
    #   Must be an odd number.
    # - C (2): A constant subtracted from the calculated mean. It's a fine-tuning
    #   parameter to adjust the threshold.
    thresholded_image = cv2.adaptiveThreshold(
        cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return Image.fromarray(thresholded_image)


def deskew(image: Image.Image) -> Image.Image:
    """
    Detects and corrects the skew (tilt) of an image containing text.

    This function works by finding the contours of the text, calculating the
    minimum bounding rectangle around them, and using the angle of that
    rectangle to rotate the entire image back to a horizontal position.

    Args:
        image (Image.Image): The input PIL Image.

    Returns:
        Image.Image: The deskewed PIL Image.
    """
    cv_image = np.array(image.convert('L'))

    # Invert the image colors. For contour detection, it's often easier to find
    # white objects on a black background.
    cv_image = cv2.bitwise_not(cv_image)

    # Find the coordinates of all "on" pixels (the text).
    coords = np.column_stack(np.where(cv_image > 0))

    # Get the minimum area bounding rectangle that encloses all text points.
    # The last element of the tuple returned by minAreaRect is the rotation angle.
    angle = cv2.minAreaRect(coords)[-1]

    # The angle from minAreaRect can be in the range [-90, 0). We need to
    # adjust it to be a standard rotation angle.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Get the image dimensions and calculate the center for rotation.
    (h, w) = image.size
    center = (w // 2, h // 2)

    # Calculate the rotation matrix.
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to the original (color or grayscale) image.
    # - flags=cv2.INTER_CUBIC: A high-quality interpolation method.
    # - borderMode=cv2.BORDER_REPLICATE: Fills in the new corners with replicated pixels.
    rotated_cv = cv2.warpAffine(np.array(image), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(rotated_cv)
