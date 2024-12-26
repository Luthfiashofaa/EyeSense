import cv2
import numpy as np
import os
import time
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
import threading

# Inisialisasi Flask dan Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Tentukan path untuk model dan dataset
MODEL_PATH = "model/trained_faces.yml"
DATASET_PATH = "dataset"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model yang sudah dilatih
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

# Fungsi untuk mengenali wajah
def recognize_face():
    cap = cv2.VideoCapture(0)  # Membuka kamera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:  # Jika tidak ada wajah yang terdeteksi
            print("No face detected")
            socketio.emit('face_detected', {'name': 'No face detected'})
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                label, confidence = face_recognizer.predict(roi_gray)

                if confidence < 100:  # Ambil nama berdasarkan label jika confidence rendah
                    name = get_name_from_label(label)
                    print(f"Detected: {name} with confidence: {confidence}")
                    
                    # Emit nama yang terdeteksi ke client (Flutter)
                    socketio.emit('face_detected', {'name': name})

        time.sleep(1)  # Delay untuk menghindari kamera terlalu cepat membaca frame

    cap.release()
    cv2.destroyAllWindows()

def get_name_from_label(label):
    # Ambil nama dari label yang telah dilatih
    try:
        with open("model/user_names.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                stored_label, name = line.strip().split(":")
                if int(stored_label) == label:
                    return name
    except FileNotFoundError:
        return "Unknown"
    return "Unknown"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    # Mulai deteksi wajah dalam thread terpisah
    thread = threading.Thread(target=recognize_face)
    thread.daemon = True  # Agar thread berhenti saat aplikasi dimatikan
    thread.start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
