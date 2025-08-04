import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath("./face-parsing.PyTorch"))
from test import evaluate
from demographic_face_analyze import without_beard_region

print('test')


def weights_calc(data):
    weights = []
    num_list = list(data.values())
    total = sum(num_list)
    for num in num_list:
        weights.append(num / total)
    return np.array(weights)

def information_analysis(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_data = {}
    for row in image:
        for pixel in row:
            brightness_data[pixel] = brightness_data.get(pixel, 0) + 1
    weight_list = weights_calc(brightness_data)
    level_list = np.array(list(brightness_data.keys()))
    avg_level = sum(level_list * weight_list)
    
    information = sum(abs(level_list - avg_level) * weight_list)
    return information

def analyze_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Get face mask
    mask = evaluate([image])[0]  # Note: evaluate() expects a list
    
    # Extract upper face region
    upper_face = without_beard_region(image, mask)
    
    # Calculate BIM
    bim = information_analysis(upper_face)
    print(f"Brightness Information Metric: {bim:.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Path to face image")
    args = parser.parse_args()
    analyze_single_image(args.image)
    