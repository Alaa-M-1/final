from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def load_facenet_model():
    model_url = "https://tfhub.dev/google/facenet/1"
    model = hub.load(model_url)
    return model

model = load_facenet_model()


app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/captures/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

camera = cv2.VideoCapture(0)  # Open the webcam
detection_paused = False  # Flag to control the detection
SHARPNESS_THRESHOLD = 100.0  # Adjust this threshold based on your experiments


# Load known criminal faces
criminal_images_folder = 'static/criminals/'


def load_known_faces(known_faces_folder, model):
    known_face_encodings = []
    known_face_names = []
    known_face_ids = ["Y32510","Y32511","Y32512","Y32513","Y32514","Y32515","Y32516","Y32517","Y32518","Y32519","Y32520","Y32521","Y32522","Y32523","Y32524","Y32525","Y32526","Y32579","Y32580"]

    for filename in os.listdir(known_faces_folder):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(known_faces_folder, filename)
            image = preprocess_image(img_path)
            face_embeddings = model(image[np.newaxis, ...])  # Add batch dimension
            known_face_encodings.append(face_embeddings[0].numpy())
            known_face_names.append(os.path.splitext(filename)[0])  # Name or ID of the criminal

    return known_face_encodings, known_face_names, known_face_ids

criminal_encodings, criminal_names, criminal_ids = load_known_faces(criminal_images_folder)


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def get_face_embedding(face_image, model):
    # Preprocess the image for FaceNet
    face_image = cv2.resize(face_image, (160, 160))
    face_image = np.expand_dims(face_image, axis=0)
    face_image = (face_image / 255.0)  # Normalize image
    embedding = model(face_image).numpy()
    return embedding.flatten()

def detect_faces_and_capture(frame):
    global detection_paused
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = detector.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check sharpness of the frame
    sharpness = calculate_sharpness(frame)

    if len(faces) > 0 and sharpness > SHARPNESS_THRESHOLD:
        for (x, y, w, h) in faces:
            face_image = rgb_frame[y:y+h, x:x+w]
            
            # Extract embedding for the detected face
            face_embedding = get_face_embedding(face_image, model)
            
            # Compare the detected face embedding to known criminals
            known_criminal_embeddings = [np.array(embedding) for embedding in criminal_encodings]
            distances = [np.linalg.norm(face_embedding - criminal_embedding) for criminal_embedding in known_criminal_embeddings]
            
            min_distance_index = np.argmin(distances)
            threshold = 0.6  # Adjust the threshold as needed
            
            if distances[min_distance_index] < threshold:
                name = criminal_names[min_distance_index]
                id = criminal_ids[min_distance_index]
                print(f"Criminal detected: {name}, {id}!")
                
                # Draw rectangle and put text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Wanted Criminal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            else:
                print("No criminal match.")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "No criminal match.", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        
        # Save the frame
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(file_path, frame)
        print(f"Image saved to {file_path}")

        detection_paused = True  # Pause the detection
        return file_path
    return None


def detect_faces(frame):
    global detection_paused
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Draw rectangles around the detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    return None



def generate_frames():
    frame = None
    global detection_paused
    
    while True:
        if not detection_paused:
            success, frame = camera.read()
            if not success:
                    break
            else:
                # Detect faces and capture the frame if a face is found
                detect_faces_and_capture(frame)
                
                # Convert the frame to a JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                # Yield the frame for the video feed
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            if frame == None:
                break
            # When paused, just show the last frame until continue is pressed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def generate_frames2(camera_source):
    while True:
        success, frame = camera_source.read()
        if not success:
            break
        else:
            # Detect faces and capture the frame if a face is found
            detect_faces(frame)
            
            # Convert the frame to a JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame for the video feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames2(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_feed')
def detection_feed():
    # Return the last captured image as a feed
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/continue_detection', methods=['POST'])
def continue_detection():
    global detection_paused
    detection_paused = False  # Resume the detection
    return jsonify({'status': 'success'})  # Return a JSON response

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()