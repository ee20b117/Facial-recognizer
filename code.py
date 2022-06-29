import numpy as np
import os
import cv2
import face_recognition as fr

#training the model

train_names = []
train_encodings = []
dataset_path = './train/'
images = os.listdir(dataset_path)
for i in images:
    image = fr.load_image_file(dataset_path+i)
    image_path = dataset_path + i
    train_names.append(os.path.splitext(os.path.basename(image_path))[0])
    encoding = fr.face_encodings(image)[0] 
    train_encodings.append(encoding)
print(train_names)

#testing the model

test_path = "./test/test.jpg"
image = cv2.imread(test_path)
face_locations = fr.face_locations(image)
test_encodings = fr.face_encodings(image,face_locations)

for face_location, test_encoding in zip(face_locations,test_encodings):
    (top,right,bottom,left) = face_location
    matches = fr.compare_faces(train_encodings, test_encoding)
    face_distances = fr.face_distance(train_encodings, test_encoding)
    possible_match = np.argmin(face_distances)
    if (matches[possible_match]!=0):
        name = train_names[possible_match]
    cv2.rectangle(image,(left,bottom),(right,top),(240,240,240),1)
    cv2.rectangle(image, (left, bottom - 28), (right, bottom), (240, 240, 240), cv2.FILLED)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, name, (left, bottom - 6), font, 1.0, (0, 0, 0), 1)

cv2.imshow("Did I pass?!",image)
cv2.waitKey(0)
