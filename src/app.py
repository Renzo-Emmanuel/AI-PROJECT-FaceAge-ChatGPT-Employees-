import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def detect_age_gender(image):
    # Model files
    faceProto = "../models/opencv_face_detector.pbtxt"
    faceModel = "../models/opencv_face_detector_uint8.pb"
    ageProto = "../models/age_deploy.prototxt"
    ageModel = "../models/age_net.caffemodel"
    genderProto = "../models/gender_deploy.prototxt"
    genderModel = "../models/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load networks
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, image)
    
    results = []
    
    if not faceBoxes:
        return None, "No face detected"

    for faceBox in faceBoxes:
        face = image[max(0, faceBox[1]-padding):
                    min(faceBox[3]+padding, image.shape[0]-1), max(0, faceBox[0]-padding):
                    min(faceBox[2]+padding, image.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        results.append({
            'gender': gender,
            'age': age[1:-1] + ' years'
        })

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return resultImg, results

# Streamlit UI
st.set_page_config(page_title="Gender & Age Detection", page_icon="👤", layout="wide")

st.title("🎯 Gender & Age Detection")
st.markdown("Upload an image to detect gender and age of faces in the photo")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This application uses deep learning models to predict:\n"
    "- **Gender**: Male or Female\n"
    "- **Age**: One of 8 ranges from 0-100 years\n\n"
    "Built with OpenCV and pre-trained models."
)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing faces"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if st.button("🔍 Detect Gender & Age", type="primary"):
            with st.spinner("Analyzing image..."):
                result_img, results = detect_age_gender(opencv_image)
                
                if result_img is not None:
                    # Convert back to RGB for display
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.header("📊 Results")
                        st.image(result_img_rgb, caption="Detection Results", use_column_width=True)
                        
                        if isinstance(results, list):
                            for i, result in enumerate(results):
                                st.success(f"**Face {i+1}:**\n- Gender: {result['gender']}\n- Age: {result['age']}")
                        else:
                            st.error(results)
                else:
                    st.error("No faces detected in the image. Please try another image.")

# Sample images section
st.header("🖼️ Try Sample Images")
sample_images = ['../sample_images/girl1.jpg', '../sample_images/girl2.jpg', '../sample_images/kid1.jpg', '../sample_images/kid2.jpg', '../sample_images/man1.jpg', '../sample_images/man2.jpg', '../sample_images/woman1.jpg']

cols = st.columns(4)
for i, sample in enumerate(sample_images):
    with cols[i % 4]:
        if os.path.exists(sample):
            sample_img = Image.open(sample)
            st.image(sample_img, caption=os.path.basename(sample), use_column_width=True)
            if st.button(f"Use {os.path.basename(sample)}", key=f"sample_{i}"):
                opencv_sample = cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR)
                result_img, results = detect_age_gender(opencv_sample)
                
                if result_img is not None:
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_img_rgb, caption=f"Results for {os.path.basename(sample)}")
                    
                    if isinstance(results, list):
                        for j, result in enumerate(results):
                            st.info(f"Face {j+1}: {result['gender']}, {result['age']}")

st.markdown("---")
st.markdown("💡 **Tip**: For best results, use clear images with visible faces and good lighting.")