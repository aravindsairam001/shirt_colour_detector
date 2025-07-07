from sklearn.cluster import KMeans
import cv2
import numpy as np
import webcolors
from matplotlib.colors import XKCD_COLORS


def get_dominant_color(image_bgr):
    # Work directly in BGR, use k-means for dominant color
    pixels = image_bgr.reshape(-1, 3)
    pixels = np.array([px for px in pixels if np.mean(px) > 40])  # Remove dark pixels

    if len(pixels) == 0:
        return (0, 0, 0)

    # Use k-means to find the dominant color
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
    print(f"[DEBUG] dominant_bgr (kmeans): {dominant}")
    return tuple(dominant)

def hex_to_bgr(hex_value):
    """Convert hex to BGR tuple for OpenCV comparison."""
    hex_value = hex_value.lstrip('#')
    rgb = tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def closest_colour(requested_colour):
    # Use CSS3 color set and LAB color space for perceptual distance


    # Use XKCD color set and LAB color space for perceptual distance
    xkcd_bgr = {name.replace('xkcd:', ''): hex_to_bgr(hex) for name, hex in XKCD_COLORS.items()}

    # Convert both requested_colour and reference to LAB
    requested_lab = cv2.cvtColor(np.uint8([[requested_colour]]), cv2.COLOR_BGR2LAB)[0][0]
    min_dist = float('inf')
    closest_name = None
    for name, bgr in xkcd_bgr.items():
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        dist = np.linalg.norm(requested_lab - lab)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    print(f"[DEBUG] requested_colour: {requested_colour}, closest_xkcd: {closest_name}")
    return closest_name

def get_color_name(rgb_color):
    return closest_colour(rgb_color)
