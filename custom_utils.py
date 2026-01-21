import cv2
import numpy as np
from pathlib import Path
from time import time

def imread_custom(image_path, options=cv2.IMREAD_UNCHANGED):
    """Read an image from a file path and return it as a NumPy array.
    Args:
        image_path (str or Path): The path to the image file.
        options (int): The OpenCV image reading options.
    Returns:
        numpy.ndarray: The image as a NumPy array.
    Advantages:
        - Handles file paths with non-ASCII characters.
        - More robust than cv2.imread for certain file path issues.
        - Supports various image formats.
        - Returns images in BGR format as used by OpenCV.
        - Longer file paths are supported on Windows systems.
        - Can read images from network drives or special file systems.
        - Almost equivalent performance to cv2.imread.
    """
    img_array = np.fromfile(image_path, dtype=np.uint8)
    
    # Decode the array directly
    return cv2.imdecode(img_array, options)


if __name__ == "__main__":
    # Example usage
    image_folder = Path("./data/images")
    image_file = "times-square.jpg"  # Replace with your image path
    image_path = image_folder / image_file
    start_time = time()
    image = imread_custom(image_path)
    end_time = time()
    print(f"Custom function time taken to read image: {end_time - start_time:.6f} seconds")
    cv2.imshow("Image custom load", image)
    cv2.waitKey(0)

    start_time = time()
    image_cv2 = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    end_time = time()
    print(f"cv2.imread time taken to read image: {end_time - start_time:.6f} seconds")
    cv2.imshow("Image using cv2.imread", image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
