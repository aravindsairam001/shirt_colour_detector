import cv2
import numpy as np

def preprocess_image(image_np, bbox=None, max_width=800):
    """
    Preprocess the image for shirt color detection.
    If bbox is provided, remove background using GrabCut for that region.
    """
    # Resize if image is too large
    h, w = image_np.shape[:2]
    if w > max_width:
        scale = max_width / w
        image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)))

    # Remove background if bbox is provided
    if bbox is not None:
        mask = np.zeros(image_np.shape[:2], np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        x, y, bw, bh = bbox
        x = max(0, x - 10)
        y = max(0, y - 10)
        bw = min(image_np.shape[1] - x, bw + 20)
        bh = min(image_np.shape[0] - y, bh + 20)
        rect = (x, y, bw, bh)
        cv2.grabCut(image_np, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image_np = image_np * mask2[:, :, np.newaxis]

    # Apply Gaussian Blur
    image_np = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Convert to LAB and apply CLAHE on L channel for contrast enhancement
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    image_np = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return image_np