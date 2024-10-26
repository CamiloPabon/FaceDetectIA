import cv2
import os
import numpy as np

faces = ['camilo','antonio','steven'
         ]
labels_y = []
faces_x = []

for face in faces:
    archivos = os.listdir(face)
    for archivo in archivos:
        img = cv2.imread(os.path.join(face,archivo))
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_gray_fix = cv2.resize(img_gray,(150,150))
        if face.startswith("camilo"):
            labels_y.append(0)
        else:
            if face.startswith("antonio"):
                labels_y.append(1)
            else:
                labels_y.append(2)

        faces_x.append(img_gray_fix)

eigen = cv2.face.EigenFaceRecognizer_create()
eigen.train(faces_x, np.array(labels_y))
eigen.write("eigen.xml")

fisher = cv2.face.FisherFaceRecognizer_create()
fisher.train(faces_x, np.array(labels_y))
fisher.write("fisher.xml")

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces_x, np.array(labels_y))
lbph.write("lbph.xml")


