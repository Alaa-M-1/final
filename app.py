from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import os
import datetime
import face_recognition
import numpy as np


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

def load_known_faces(known_faces_folder):
    known_face_encodings = []
    known_face_names = []
    known_face_ids = ["Y32510","Y32511","Y32512","Y32513","Y32514","Y32515","Y32516","Y32517","Y32518","Y32519","Y32520","Y32521","Y32522","Y32523","Y32524","Y32525","Y32526","Y32579","Y32580"]

    for filename in os.listdir(known_faces_folder):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(known_faces_folder, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Name or ID of the criminal
            
    return known_face_encodings, known_face_names, known_face_ids

criminal_encodings, criminal_names, criminal_ids = load_known_faces(criminal_images_folder)


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def detect_faces_and_capture(frame):
    global detection_paused
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Check sharpness of the frame
    sharpness = calculate_sharpness(frame)

    if len(face_locations) > 0 and sharpness > SHARPNESS_THRESHOLD:
        # Draw rectangles around the detected faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Compare the detected face to known criminals
            matches = face_recognition.compare_faces(criminal_encodings, face_encoding)
            name = "No criminal match."
            
            
            if True in matches:
                match_index = matches.index(True)
                name = criminal_names[match_index]
                id = criminal_ids[match_index]
                print(f"Criminal detected: {name}, {id}!")
                # Display "Wanted Criminal" above the rectangle
                cv2.putText(frame, "Wanted Criminal", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


                # Display the name below the rectangle
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2)

            else:
                print("No criminal match.")
                # Display "Wanted Criminal" above the rectangle
                cv2.putText(frame, "No criminal match.", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ( 255,0, 0), 2)

            
            
        
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