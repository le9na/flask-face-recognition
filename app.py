#Include Libraries
import cv2
import face_recognition as fp
import numpy as np
from flask import Flask, render_template, Response

app=Flask(__name__)

# Camera Capture from WebCam
cam = cv2.VideoCapture(0)

# Prepare Data
abdulelah = fp.load_image_file("data/abdulelah.jpg")
saleh = fp.load_image_file("data/saleh.jpg")
yosef = fp.load_image_file("data/yosef.jpg")

abdulelah_encoding = fp.face_encodings(abdulelah)[0]
saleh_encoding = fp.face_encodings(saleh)[0]
yosef_encoding = fp.face_encodings(yosef)[0]

known_encodings = [
    abdulelah_encoding,
    saleh_encoding,
    yosef_encoding
]

known_names = [
    "abdulelah",
    "saleh",
    "yosef"
]

face_locations = []
face_encodings = []
names = []

def generate_frames():
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        else:

            minimize_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_frame = minimize_frame[:, :, ::-1] # Converts from BGR to RGB

            face_locations = fp.face_locations(rgb_frame)
            face_encodings = fp.face_encodings(rgb_frame, face_locations)
            names = []

            for face_encoding in face_encodings:
                match = fp.compare_faces(known_encodings, face_encodings)
                name = "Not known"

                face_distances = fp.face_distance(known_encodings, face_encodings)
                match_index = np.argmin(face_distances)
                if match[match_index]:
                    name = known_names[match_index]
                
                names.append(name)
            
            # Display:
            for (top, right, bottom, left), name in zip(face_locations, names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.rectangle(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
