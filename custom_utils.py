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
    img_array = np.fromfile(image_path, dtype=np.uint8)
    
    # Decode the array directly
    return cv2.imdecode(img_array, options)


def enhance_image(image_path, h=3):
    # Load the image
    img = imread_custom(image_path)
    
    # Denoising (Optional, but makes the quality look 'cleaner')
    # h=3 is a gentle setting; increase if the image is grainy
    dst = cv2.fastNlMeansDenoisingColored(img, None, h, 3, 7, 21)

    # Sharpening (Unsharp Masking)
    # Gaussian blur creates a 'mask' of the details
    gaussian_blur = cv2.GaussianBlur(dst, (0, 0), 3)
    # addWeighted blends the original with the mask to pop the edges
    # formula: sharp = original * (1 + amount) + blurred * (-amount)
    sharp_img = cv2.addWeighted(dst, 1.5, gaussian_blur, -0.5, 0)

    # Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    # This improves quality by fixing lighting/shadow issues locally
    lab = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final_img


def aggressive_sharpen(image_path, kernel):
    # Example of aggressive sharpening using a kernel
    img = imread_custom(image_path)
    # Apply the filter
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def ai_sharpen(image_path, model_path, cuda=False):
    # Initialize the Super Resolution object
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # Load the model
    path = str(model_path)
    sr.readModel(path)
    sr.setModel("edsr", 4) # Upscale by 4x

    # Set CUDA to make it fast (per our previous talk!)
    if cuda:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    img = imread_custom(image_path)
    result = sr.upsample(img)

    return result


if __name__ == "__main__":
    # Example usage
    # image_folder = Path("./data/images")
    # image_file = "times-square.jpg"  # Replace with your image path
    # image_path = image_folder / image_file
    # start_time = time()
    # image = imread_custom(image_path, cv2.IMREAD_GRAYSCALE)
    # end_time = time()
    # print(f"Custom function time taken to read image: {end_time - start_time:.6f} seconds")
    # print(f"Image shape: {image.shape}")
    # cv2.imshow("Image custom load", image)
    # cv2.waitKey(0)

    # start_time = time()
    # image_cv2 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    # end_time = time()
    # print(f"cv2.imread time taken to read image: {end_time - start_time:.6f} seconds")
    # print(f"Image shape: {image_cv2.shape}")
    # cv2.imshow("Image using cv2.imread", image_cv2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    img = Path('input.jpeg')
    model_path = Path('models', 'EDSR_x4.pb')
    
    # Define a sharp kernel
    # The center value (9) minus the surrounding values (-1) equals 1
    kernel = np.array([[ 0, -1,  0],
                       [-1,  9, -1],
                       [ 0, -1,  0]])

    # Usage
    # enhanced = enhance_image(img, h=7)
    # cv2.imwrite('enhanced_result.jpg', enhanced)
    
    # sharpened = aggressive_sharpen(img, kernel)
    # cv2.imwrite('aggressive_sharp.jpg', sharpened)

    # ai_sharpened = ai_sharpen(img, model_path, cuda=False)
    # cv2.imwrite('ai_sharpened.jpg', ai_sharpened)

    