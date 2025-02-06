import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

directory = 'Trained_images'
known_faces_names = []
known_face_encoding = []

for filename in os.scandir(directory):
    if filename.is_file():
        known_faces_names.append(filename.path.replace("\\", "/"))

for known_faces_name in known_faces_names:
    loaded_image = face_recognition.load_image_file(known_faces_name)
    generated_encoding = face_recognition.face_encodings(loaded_image)[0]
    known_face_encoding.append(generated_encoding)

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(
                known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    student_name = name.replace(
                        directory+"/", "").replace(".jpg", "")
                    now_new = datetime.now()
                    current_time = now_new.strftime("%H:%M:%S")
                    print("'" + student_name + "'" +
                          "Checked in at : " + current_time)
                    lnwriter.writerow([student_name, current_time])
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Attendance Taken Successfully")
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
