import cv2
import numpy as np
from pathlib import Path
from time import time

def imread_custom(image_path, options=cv2.IMREAD_UNCHANGED):
    """
    Read an image from a file path.
    Advantages:
        - Handles file paths with non-ASCII characters.
        - More robust than cv2.imread for certain file path issues.
        - Supports various image formats.
        - Returns images in BGR format as used by OpenCV.
        - Longer file paths are supported on Windows systems.
        - Can read images from network drives or special file systems.
        - Almost equivalent performance to cv2.imread.
    Args:
        image_path (str or Path): The path to the image file.
        options (int): The OpenCV image reading options.
    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    img_array = np.fromfile(str(image_path), dtype=np.uint8)

    # If file read failed, return None
    if img_array.size == 0:
        return None

    # Decode the array directly
    img = cv2.imdecode(img_array, options)
    return img

if __name__ == "__main__":
    start_time = time()
    end_time = time()