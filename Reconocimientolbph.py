import cv2

labels = ['camilo','antonio','steven']
video="antonio"
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_reg = cv2.face.LBPHFaceRecognizer_create()
face_reg.read("lbph.xml")

cap= cv2.VideoCapture("videos/"+video+".mp4")
#cap= cv2.VideoCapture(0)

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    faces = face_detect.detectMultiScale(frame, 1.3, 5, minSize=(50, 50), maxSize=(600, 600))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y + h, x:x + w]
        img_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        img_gray_fix = cv2.resize(img_gray, (150, 150))
        predict = face_reg.predict(img_gray_fix)

        cv2.putText(frame, labels[predict[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(predict[1]), (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.putText(frame,"Identificacion con lbph ",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 10), 3)
    cv2.imshow('frame', frame)
    k=cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()