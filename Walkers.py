import cv2

body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

cap = cv2.VideoCapture('walking.avi')

while True:
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    bodies = body_cascade.detectMultiScale(gray,1.1,5)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255))

    cv2.imshow("Walkers",frame)
    
    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()
