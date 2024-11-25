import cv2
import dlib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load dlib's pre-trained models for face detection and landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Get the uploaded file
    file = request.files['file']
    
    # Save the file temporarily
    file_path = 'uploads/' + file.filename
    file.save(file_path)
    
    # Load the image
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray)
    
    # Detect facial landmarks for each detected face
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    
    # Save the image with detected landmarks for display
    output_path = 'uploads/processed_' + file.filename
    cv2.imwrite(output_path, img)
    
    return f"File uploaded and processed: {output_path}"

if __name__ == '__main__':
    app.run(debug=True)

