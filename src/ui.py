import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class GenderAgeDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gender & Age Detection")
        self.root.geometry("800x600")
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_widgets()
        
    def load_models(self):
        try:
            faceProto = "../models/opencv_face_detector.pbtxt"
            faceModel = "../models/opencv_face_detector_uint8.pb"
            ageProto = "../models/age_deploy.prototxt"
            ageModel = "../models/age_net.caffemodel"
            genderProto = "../models/gender_deploy.prototxt"
            genderModel = "../models/gender_net.caffemodel"

            self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            self.genderList = ['Male', 'Female']

            self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
            self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
            self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
            
            messagebox.showinfo("Success", "Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Gender & Age Detection", 
                              font=("Arial", 20, "bold"), fg="blue")
        title_label.pack(pady=10)
        
        # Upload button
        upload_btn = tk.Button(self.root, text="Upload Image", 
                              command=self.upload_image, 
                              font=("Arial", 12), bg="lightblue", 
                              width=15, height=2)
        upload_btn.pack(pady=10)
        
        # Image display frame
        self.image_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Image label
        self.image_label = tk.Label(self.image_frame, text="No image selected", 
                                   font=("Arial", 12), bg="white")
        self.image_label.pack(expand=True)
        
        # Results frame
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Results label
        self.results_label = tk.Label(self.results_frame, text="Results will appear here", 
                                     font=("Arial", 12), bg="lightyellow", 
                                     relief=tk.SUNKEN, bd=2, height=3)
        self.results_label.pack(fill=tk.X)
        
        # Sample images frame
        sample_frame = tk.Frame(self.root)
        sample_frame.pack(pady=10)
        
        tk.Label(sample_frame, text="Try Sample Images:", font=("Arial", 10, "bold")).pack()
        
        sample_btn_frame = tk.Frame(sample_frame)
        sample_btn_frame.pack()
        
        samples = ['../sample_images/girl1.jpg', '../sample_images/man1.jpg', '../sample_images/kid1.jpg', '../sample_images/woman1.jpg']
        for sample in samples:
            if os.path.exists(sample):
                btn = tk.Button(sample_btn_frame, text=os.path.basename(sample), 
                               command=lambda s=sample: self.load_sample(s),
                               font=("Arial", 8))
                btn.pack(side=tk.LEFT, padx=5)
    
    def highlightFace(self, net, frame, conf_threshold=0.7):
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
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return frameOpencvDnn, faceBoxes
    
    def detect_age_gender(self, image_path):
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                return None, "Could not read image"
            
            padding = 20
            resultImg, faceBoxes = self.highlightFace(self.faceNet, frame)
            
            if not faceBoxes:
                return resultImg, "No face detected"
            
            results = []
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1]-padding):
                           min(faceBox[3]+padding, frame.shape[0]-1), 
                           max(0, faceBox[0]-padding):
                           min(faceBox[2]+padding, frame.shape[1]-1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                
                self.genderNet.setInput(blob)
                genderPreds = self.genderNet.forward()
                gender = self.genderList[genderPreds[0].argmax()]

                self.ageNet.setInput(blob)
                agePreds = self.ageNet.forward()
                age = self.ageList[agePreds[0].argmax()]

                results.append(f"Gender: {gender}, Age: {age[1:-1]} years")
                
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            
            return resultImg, results
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def load_sample(self, sample_name):
        if os.path.exists(sample_name):
            self.process_image(sample_name)
        else:
            messagebox.showerror("Error", f"Sample image {sample_name} not found")
    
    def process_image(self, image_path):
        try:
            # Detect age and gender
            result_img, results = self.detect_age_gender(image_path)
            
            if result_img is not None:
                # Convert BGR to RGB for display
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # Resize image for display
                height, width = result_img_rgb.shape[:2]
                max_height = 300
                if height > max_height:
                    ratio = max_height / height
                    new_width = int(width * ratio)
                    result_img_rgb = cv2.resize(result_img_rgb, (new_width, max_height))
                
                # Convert to PIL and display
                pil_image = Image.fromarray(result_img_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                # Display results
                if isinstance(results, list):
                    result_text = "\n".join(results)
                else:
                    result_text = results
                    
                self.results_label.configure(text=result_text)
            else:
                self.results_label.configure(text=results)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeDetectorUI(root)
    root.mainloop()