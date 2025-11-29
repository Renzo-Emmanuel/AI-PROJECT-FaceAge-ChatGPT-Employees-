# Gender-and-Age-Detection   <img alt="GitHub" src="https://img.shields.io/github/license/smahesh29/Gender-and-Age-Detection">


<h2>Objective :</h2>
<p>To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.</p>

<h2>About the Project :</h2>
<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

<h2>Dataset :</h2>
<p>For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.</p>

<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>argparse</li>
  
       pip install argparse
</ul>

<h2>Project Structure :</h2>
<ul>
  <li><b>models/</b> - Contains all AI model files
    <ul>
      <li>opencv_face_detector.pbtxt & opencv_face_detector_uint8.pb - Face detection models</li>
      <li>age_deploy.prototxt & age_net.caffemodel - Age prediction models</li>
      <li>gender_deploy.prototxt & gender_net.caffemodel - Gender prediction models</li>
    </ul>
  </li>
  <li><b>src/</b> - Source code files
    <ul>
      <li>detect.py - Original command line script</li>
      <li>ui.py - Desktop GUI application</li>
      <li>app.py - Web interface (Streamlit)</li>
    </ul>
  </li>
  <li><b>sample_images/</b> - Sample images for testing</li>
  <li><b>Example/</b> - Example output images</li>
  <li>run_ui.py - Main launcher for desktop UI</li>
  <li>run_ui.bat - Windows batch file launcher</li>
 </ul>
 <p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>
 
 <h2>Usage :</h2>
 
 <h3>🖥️ Desktop UI (Recommended)</h3>
 <ul>
  <li>Double-click <b>run_ui.bat</b> (Windows) or run <code>python run_ui.py</code></li>
  <li>Click "Upload Image" to select your own image</li>
  <li>Or click on sample images to test</li>
  <li>Results will be displayed with detected faces highlighted</li>
 </ul>
 
 <h3>🌐 Web Interface</h3>
 <ul>
  <li>Navigate to src folder: <code>cd src</code></li>
  <li>Run: <code>streamlit run app.py</code></li>
  <li>Open browser to the displayed URL</li>
 </ul>
 
 <h3>⌨️ Command Line (Original)</h3>
 <ul>
  <li>Navigate to src folder: <code>cd src</code></li>
  <li><b>For image detection:</b> <code>python detect.py --image ../sample_images/girl1.jpg</code></li>
  <li><b>For webcam detection:</b> <code>python detect.py</code></li>
  <li>Press <b>Ctrl + C</b> to stop webcam mode</li>
 </ul>

# Working:
[![Watch the video](https://img.youtube.com/vi/ReeccRD21EU/0.jpg)](https://youtu.be/ReeccRD21EU)

<h2>Examples :</h2>
<p><b>NOTE:- I downloaded the images from Google,if you have any query or problem i can remove them, i just used it for Educational purpose.</b></p>

    >python detect.py --image girl1.jpg
    Gender: Female
    Age: 25-32 years
    
<img src="Example/Detecting age and gender girl1.png">

    >python detect.py --image girl2.jpg
    Gender: Female
    Age: 8-12 years
    
<img src="Example/Detecting age and gender girl2.png">

    >python detect.py --image kid1.jpg
    Gender: Male
    Age: 4-6 years    
    
<img src="Example/Detecting age and gender kid1.png">

    >python detect.py --image kid2.jpg
    Gender: Female
    Age: 4-6 years  
    
<img src="Example/Detecting age and gender kid2.png">

    >python detect.py --image man1.jpg
    Gender: Male
    Age: 38-43 years
    
<img src="Example/Detecting age and gender man1.png">

    >python detect.py --image man2.jpg
    Gender: Male
    Age: 25-32 years
    
<img src="Example/Detecting age and gender man2.png">

    >python detect.py --image woman1.jpg
    Gender: Female
    Age: 38-43 years
    
<img src="Example/Detecting age and gender woman1.png">
              
