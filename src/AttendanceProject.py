import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime

path = "ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        data = f.readlines()
        nameList = []
        for row in data:
            entry = row.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.write(f"\n{name},{dtString} is present at this time in class\n")


encodeListKnown = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, fx=0.5, fy=0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    if len(facesCurFrame) == 0:
        print("No faces found.")
        break

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        # print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            print(name)
            y1, x1, y2, x2 = faceLoc
            y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            y1, x1, y2, x2 = y1 - 35, x1 - 20, y2 + 20, x2 + 20
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow("Webcam", img)
            cv2.waitKey(1)
            markAttendance(name)

        # cv2.imshow("Webcam", img)
        # cv2.waitKey(1)
