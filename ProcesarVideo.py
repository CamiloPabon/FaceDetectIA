import cv2

video="antonio"
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap= cv2.VideoCapture("videos/"+video+".mp4")

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    faces = face_detect.detectMultiScale(frame, 1.3, 5, minSize=(10, 10), maxSize=(600, 600))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        k=cv2.waitKey(5)
        if k == 99:
            face = frame[y:y + h, x:x + w]
            cv2.imwrite(video+"/"+str(i)+".jpg" , face)
            i += 1
    cv2.putText(frame,"Cantidad de rostros "+str(i),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    k=cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()