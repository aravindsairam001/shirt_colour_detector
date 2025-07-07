import cv2
import numpy as np

def segment_human(image, bbox):
    """
    Segments the human from the image using GrabCut algorithm.
    
    Args:
        image (np.ndarray): Original BGR image
        bbox (tuple): Bounding box (x, y, w, h) from detector

    Returns:
        segmented (np.ndarray): Image with background removed (black)
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # Expand the box slightly for better segmentation
    x, y, w, h = bbox
    x = max(0, x - 10)
    y = max(0, y - 10)
    w = min(image.shape[1] - x, w + 20)
    h = min(image.shape[0] - y, h + 20)

    rect = (x, y, w, h)

    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply mask to original image
    segmented = image * mask2[:, :, np.newaxis]
    return segmented