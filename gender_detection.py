import cv2
import numpy as np

# -----------------------------
# Load gender classifier model
# -----------------------------
gender_net = cv2.dnn.readNetFromCaffe(
    "deploy_gender.prototxt",
    "gender_net.caffemodel"
)
gender_list = ['Male', 'Female']

# -----------------------------
# Load face detector model
# -----------------------------
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def crop_face_dnn(image_bgr):
    """
    Detects and returns the largest cropped face from the BGR image using OpenCV DNN.

    Args:
        image_bgr (np.ndarray): Input BGR image.

    Returns:
        np.ndarray or None: Cropped face image, or None if not found.
    """
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    max_conf = 0
    best_crop = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6 and confidence > max_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure box boundaries are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                best_crop = image_bgr[y1:y2, x1:x2]
                max_conf = confidence

    return best_crop

def predict_gender(person_crop_bgr):
    """
    Detect gender by first detecting face from full person crop.
    
    Args:
        person_crop_bgr (np.ndarray): The full bounding box image of the person (BGR).

    Returns:
        str: 'Male', 'Female', or 'Unknown'.
    """
    try:
        face_crop = crop_face_dnn(person_crop_bgr)
        if face_crop is None or face_crop.size == 0:
            return "Unknown"
        
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                     (78.426, 87.768, 114.895), swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        return gender_list[gender_preds[0].argmax()]
    except Exception as e:
        print(f"[ERROR] Gender prediction failed: {e}")
        return "Unknown"