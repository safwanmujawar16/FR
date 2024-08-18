import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
 
 
video=cv2.VideoCapture(0)
ben_image=face_recognition.load_image_file("Ben.jpg")
ben_encoding=face_recognition.face_encodings(ben_image)[0]
 
harry_image=face_recognition.load_image_file("Harry.jpg")
harry_encoding=face_recognition.face_encodings(harry_image)[0]
 
known_face_encoding = [
    ben_encoding,
    harry_encoding
]
 
known_face_names= [
    "Ben",
    "Harry"
]
 
students= known_face_names.copy()
 
face_locations=[]
face_encodings=[]
face_names=[]
s=True
 
now=datetime.now()
current_date=now.strftime("%Y-%m-%d")
 
f = open(current_date+'.csv', 'w+', newline='')
lnwriter=csv.writer(f)
 
# with open(current_date + '.csv', 'w+', newline='') as f:
#     lnwriter = csv.writer(f)
 
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame")
        break
    _,frame=video.read()
    small_frame= cv2.resize(frame,(600,400),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(small_frame)
        face_encodings=face_recognition.face_encodings(small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index= np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]
 
            face_names.append(name)
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
 
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
f.close()