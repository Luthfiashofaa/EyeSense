import cv2
import numpy as np
import os

DATASET_PATH = "dataset"
MODEL_PATH = "model/trained_faces.yml"

def train_faces():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_images = []
    face_ids = []
    user_names = {}

    user_id = 0
    for user_name in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, user_name)
        if os.path.isdir(user_path):
            user_names[user_id] = user_name
            for filename in os.listdir(user_path):
                img_path = os.path.join(user_path, filename)
                if filename.endswith(".jpg"):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    face_images.append(img)
                    face_ids.append(user_id)
            user_id += 1

    face_recognizer.train(face_images, np.array(face_ids))

    if not os.path.exists("model"):
        os.makedirs("model")
    face_recognizer.save(MODEL_PATH)

    # Simpan nama pengguna ke file
    with open("model/user_names.txt", "w") as file:
        for uid, uname in user_names.items():
            file.write(f"{uid}:{uname}\n")

    print("Training complete!")

if __name__ == "__main__":
    train_faces()
