import cv2
import os

# Path tempat menyimpan dataset
DATASET_PATH = "dataset"

def add_faces(user_name):
    user_path = os.path.join(DATASET_PATH, user_name)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            filename = os.path.join(user_path, f"{user_name}_{count}.jpg")
            cv2.imwrite(filename, face)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Add Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter User Name: ")
    add_faces(user_name)
