# Import required modules
import cv2 as cv
import time
import argparse
import os

def draw_info_box(frame, text):
    # Box settings
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1

    # Get text size
    lines = text.split("\n")
    width = 0
    height = 0
    for line in lines:
        (w, h), _ = cv.getTextSize(line, font, scale, thickness)
        width = max(width, w)
        height += h + 5

    # Box position — bottom right
    x2 = frame.shape[1] - 10
    y2 = frame.shape[0] - 10
    x1 = x2 - width - 20
    y1 = y2 - height - 20

    # Draw filled rectangle
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Write text
    y = y1 + 25
    for line in lines:
        cv.putText(frame, line, (x1 + 10, y), font, scale, (255, 255, 255), 1)
        y += 22


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "..", "models")

faceProto = os.path.join(models_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(models_dir, "age_deploy.prototxt")
ageModel = os.path.join(models_dir, "age_net.caffemodel")
genderProto = os.path.join(models_dir, "gender_deploy.prototxt")
genderModel = os.path.join(models_dir, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = [
    ' Newborn (0-2)',
    ' Toddler (3-5)',
    ' Child (6-12)',
    ' Teen (13-19)',
    ' Young Adult (20-25)',
    ' Adult (26-35)',
    ' Middle Age (36-45)',
    ' Mature (46-59)',
    ' Senior (60-100)'
]
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream
cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20
while True:
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                     max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # -------- GENDER PREDICTION --------
        # -------- GENDER PREDICTION --------
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()

        gender_idx = int(genderPreds[0].argmax())
        gender = genderList[gender_idx]
        gender_conf = float(genderPreds[0][gender_idx] * 100)

        # -------- AGE PREDICTION --------
        ageNet.setInput(blob)
        agePreds = ageNet.forward()

        probs = [float(p) for p in agePreds[0]]
        age_idx = int(np.argmax(probs)) if 'np' in globals() else int(probs.index(max(probs)))
        age_label = ageList[age_idx]
        age_conf = float(probs[age_idx] * 100)

        # -------- ESTIMATED EXACT AGE (WEIGHTED MIDPOINTS) --------
        # Define ranges matching your ageList order
        age_ranges = [(0, 2), (3, 5), (6, 12), (13, 19), (20, 25), (26, 35), (36, 45), (46, 59), (60, 100)]
        midpoints = [(a + b) / 2.0 for (a, b) in age_ranges]

        # Weighted expected age
        estimated_age = sum(p * m for p, m in zip(probs, midpoints))

        # -------- PRINT TO CONSOLE --------
        print(f"Gender: {gender}, conf = {gender_conf:.2f}%")
        print(f"Age Group: {age_label}, conf = {age_conf:.2f}%")
        print(f"Estimated Age: {estimated_age:.1f} years")

        # -------- DRAW FACE LABEL --------
        label = f"{gender}, {age_label}"
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # -------- DRAW BOTTOM-RIGHT INFO BOX --------
        info_text = (
            f"Estimated Age: {estimated_age:.1f} years\n"
            f"Age Group: {gender} {age_label}\n"
            f"Confidence Level: {age_conf:.2f}%"
        )
        draw_info_box(frameFace, info_text)

        cv.putText(frameFace, label, (bbox[0], bbox[1]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow("Age Gender Demo", frameFace)

    # Wait for 1ms and get key
    key = cv.waitKey(1) & 0xFF

    # ESC key closes webcam
    if key == 27:
        break

    # Check if the OpenCV window has been closed
    if cv.getWindowProperty("Age Gender Demo", cv.WND_PROP_AUTOSIZE) < 0:
        break

    print("time : {:.3f}".format(time.time() - t))

cap.release()
cv.destroyAllWindows()

