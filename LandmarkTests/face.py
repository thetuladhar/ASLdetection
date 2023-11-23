import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
import numpy as np

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=5)

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            #connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            #connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            #connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())


    def plotLandmark(LandmarkIndex):
        try:
            landmark = results.multi_face_landmarks[0].landmark[LandmarkIndex]
            xland = landmark.x
            yland = landmark.y
            #print("x",xland,"\ty",yland)
        except:
            pass
            #beyond frame
            xland = 0
            yland = 0

        #Normalize for plotting
        xnorm = int(xland * image.shape[1])
        ynorm = int(yland * image.shape[0])
        #DRAW
        cv2.circle(image, (xnorm, ynorm), radius=2, color=(0, 0, 255), thickness=2)
        cv2.putText(image,str(LandmarkIndex),((xnorm),ynorm),cv2.FONT_HERSHEY_DUPLEX,.5,(255,255, 255),1) #text

    for i in range(0,478,1):
        plotLandmark(i)

    #NOTABLE LANDMARKS
    #lips 0 15 61 306
    #eyes 130 159 145 173
    #plot 386 374 398 466
    #iris 468 473
    ##nose 4 9 10

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh',image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()