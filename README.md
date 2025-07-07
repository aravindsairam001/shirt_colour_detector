# 👕 Shirt Color & Gender Detection Web App

This Streamlit-based web application detects people in an uploaded image, segments their shirts, predicts the dominant shirt color, and determines their gender using deep learning and classical computer vision techniques.

---

## ✨ Features

- ✅ Detects multiple people in an image using YOLOv5s
- 🎨 Segments shirt region using GrabCut for better color isolation
- 🧠 Predicts **shirt color** using K-Means + perceptual color matching (XKCD dataset)
- 👦👧 Predicts **gender** using OpenCV DNN-based face detection and gender classification
- 📊 Shows a clean summary table for each person
- 🖼️ Option to debug and display cropped shirt segments

---

## 🖼️ How it Works

Upload an image and get:
- Cropped face
- Detected gender
- Shirt color
- Annotated image with bounding boxes and labels

---

## 🏗️ Project Structure

```bash
shirt-gender-detection/
├── app.py                            # Streamlit UI
├── detection/
│   ├── segment_humans.py            # YOLOv5 person detection
│   ├── detect_shirt_colour.py       # GrabCut and mask creation
│   ├── app.py          	     
│   └── gender_detector.py           # Caffe model-based gender detection
├── models/
│   ├── yolov5su.pt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── deploy.prototxt
│   ├── gender_net.caffemodel
│   └── deploy_gender.prototxt
├── utils.py
├── recordings/
│   └── (Optional) saved outputs
├── requirements.txt
└── README.md
```
---

📦 Models Used

1. Person Detection – YOLOv5s (Ultralytics)
   	* Model: yolov5su.pt (optimized for fast person detection)
   	* Source: Ultralytics YOLOv5
2. Shirt Segmentation – GrabCut Algorithm 
	* Library: OpenCV
	* Technique: Uses bounding box & edge detection to isolate upper body
3. Shirt Color Detection – KMeans + XKCD
	* Libraries: scikit-learn, webcolors
	* Color Mapping: Closest name match from XKCD dataset
4. Face Detection – OpenCV DNN
	* Model: deploy.prototxt, res10_300x300_ssd_iter_140000.caffemode
5. Gender Classification – CaffeNet
	* Model: deploy_gender.prototxt, gender_net.caffemodel
	* utput: 'Male' or 'Female' with confidence filter

---

🛠️ Installation

```bash
git clone https://github.com/your-username/shirt-gender-detection.git
cd shirt-gender-detection
pip install -r requirements.txt
```
---

▶️ Run the App
```bash
streamlit run app.py
```
---

Future Improvements

- Age Detection
- Shirt Patter Recognition

---

🙌 Acknowledgements
- Ultralytics YOLOv5
- OpenCV DNN & GrabCut
- XKCD Color Survey
- Caffe Gender Net (by Gil Levi & Tal Hassner)
