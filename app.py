import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from detect_shirt_color import get_dominant_color, get_color_name
from gender_detection import predict_gender
from utils import crop_upper_body
from segment_human import segment_human

st.title("👕 Shirt Color Detection & Counting")
st.markdown("Upload an image. We'll detect all people and classify shirt colors automatically.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
show_crops = st.checkbox("👕 Show cropped shirt regions for debug")

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    model = YOLO("yolov5su.pt")
    results = model(image_np)  # YOLO expects RGB

    color_counts = {}
    # Use a copy of the original RGB image for annotation and display
    annotated_img = image_np.copy()  # Draw on RGB image

    summary_list = []
    person_boxes = []

    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) == 0:  # Class 0 is person
                box = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                person_boxes.append((x1, y1, x2, y2, box))

    # Sort by x1 (left to right)
    person_boxes.sort(key=lambda b: b[0])

    for person_id, (x1, y1, x2, y2, box) in enumerate(person_boxes):
        # Convert (x1, y1, x2, y2) to (x, y, w, h) for GrabCut
        w = x2 - x1
        h = y2 - y1
        bbox = (x1, y1, w, h)
        segmented = segment_human(image_bgr, bbox)
        if segmented.size == 0:
            continue

        dominant_bgr = get_dominant_color(segmented)
        color = get_color_name(dominant_bgr)

        # Crop face region roughly from top 25% of bounding box height
        person_crop = image_bgr[y1:y2, x1:x2]
        gender = predict_gender(person_crop)

        if show_crops:
            st.image(segmented, caption=f"ID: {person_id} Shirt Segment", width=150)
        color_counts[color] = color_counts.get(color, 0) + 1
        label = f"{person_id}"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,  # smaller font size
            (255, 0, 0),  # red for visibility
            1,  # thinner line
            lineType=cv2.LINE_AA
        )
        summary_list.append({"Color": color, "Gender": gender})

    # Display the annotated image in the original color scale (RGB) with improved quality
    st.image(
        annotated_img,
        caption="Detected People with Shirt Colors",
        channels="RGB",
        output_format="PNG",
        use_container_width=False
    )

    # # Display summary table
    # st.markdown("### 🧮 Shirt Color Summary")
    # st.table(color_counts)

    # Display individual person shirt color table
    st.markdown("### 👤 Individual Person Shirt Color")
    st.table(summary_list)