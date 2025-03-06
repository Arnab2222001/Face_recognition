import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

directory = "Trained_images"
known_face_encodings = []
known_face_names = []

if not os.path.exists(directory):
    os.makedirs(directory)

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(directory, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

students = set(known_face_names)

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")
csv_filename = f"{current_date}.csv"

with open(csv_filename, "w", newline="") as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Student Name", "Check-in Time"])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]

            if name in students:
                students.remove(name)
                check_in_time = datetime.now().strftime("%H:%M:%S")
                print(f"{name} checked in at {check_in_time}")
                lnwriter.writerow([name, check_in_time])

            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("x"):
            print("Attendance Taken Successfully")
            break

video_capture.release()
cv2.destroyAllWindows()
