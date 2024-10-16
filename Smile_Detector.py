import cv2

cascade_path = 'C:/Users/Dinethra/Desktop/Programs/Python A.I/Smile Detector/haarcascade_frontalface_default.xml'
cascade_path2 = 'C:/Users/Dinethra/Desktop/Programs/Python A.I/Smile Detector/haarcascade_smile.xml'

face_detector = cv2.CascadeClassifier(cascade_path)
smile_detector = cv2.CascadeClassifier(cascade_path2)

webcam = cv2.VideoCapture(0)

while True:
    #read single frame
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_grayscale)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), 2)

        the_face = frame[y:y+h , x:x+w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smile = smile_detector.detectMultiScale(face_grayscale,1.9,37)
        

        for (X, Y, W, H) in smile:
            cv2.rectangle(the_face, (X,Y), (X+W, Y+H), (0,0, 255), 2)
            #fontstyle
            font = cv2.FONT_HERSHEY_SIMPLEX
  
            # org
            org = (X, Y)
  
            # fontScale
            fontScale = 0.5
    
            # Blue color in BGR
            color = (255, 0, 0)
    
            # Line thickness of 2 px
            thickness = 1
            cv2.putText(the_face,"Smiling",org,font,fontScale,color,thickness)

    cv2.imshow('Smile_detector',frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break




print("Code completed")