import cv2
import numpy as np
from PIL import Image

def crop_upper_body(image, box):
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    width = x2 - x1

    # Crop a smaller, more central region (e.g., 40%-60% height, 20%-80% width)
    shirt_y1 = y1 + int(height * 0.4)
    shirt_y2 = y1 + int(height * 0.6)

    shirt_x1 = x1 + int(width * 0.2)
    shirt_x2 = x2 - int(width * 0.2)

    cropped = image[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    return cropped

def draw_boxes(image, boxes, matches):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if matches[i] else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image