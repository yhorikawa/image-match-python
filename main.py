import cv2
import os
from PIL import Image
import numpy as np

def convert_gif_to_png(gif_image_path):
    gif_image = Image.open(gif_image_path)
    png_image = gif_image.convert('RGB')
    png_image = np.array(png_image)
    png_image = cv2.cvtColor(png_image, cv2.COLOR_RGB2BGR)
    return png_image

def feature_matching(base_image_path, images_folder, threshold=10):
    base_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
    if base_image is None:
        base_image = convert_gif_to_png(base_image_path)
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    _, descriptors_base = sift.detectAndCompute(base_image, None)
    included_images = []

    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        if image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".webp") or image_path.endswith(".jpeg"):
            compare_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif image_path.endswith(".gif"):
            compare_image = convert_gif_to_png(image_path)
            compare_image = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
        else:
            print("Unsupported file format: " + image_path)
            continue

        _, descriptors_compare = sift.detectAndCompute(compare_image, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors_base, descriptors_compare)

        if len(matches) >= threshold:
            included_images.append(filename)

    return included_images

base_image_path = 'image.png'
images_folder = 'images'

included_images = feature_matching(base_image_path, images_folder)
print("Included Images:")
for img in included_images:
    print(img)
