import mediapipe as mp #import mediapipe
import cv2 #import opencv2
import numpy as np
import pandas as pd

import pickle
with open('RightHandTraining.pkl', 'rb') as f:#
    model=pickle.load(f)

mp_drawing = mp.solutions.drawing_utils #mediapipe drawing helpers
mp_holistic = mp.solutions.holistic #mediapipe

#Initialize variables
classification='Nil'
probability=[0]
hand="none"

#VIDEO CAPTURE
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #make detections
    results = holistic.process(image)
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #FACE TESSALATION(mesh in face)
    '''mp_drawing.draw_landmarks(
        image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,#MODEL
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=1, circle_radius=1),#specks
        mp_drawing.DrawingSpec(color=(82, 39, 24), thickness=1, circle_radius=10)#connections
     )'''
    #FACE CONTOURS
    '''mp_drawing.draw_landmarks(
       image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
    )'''

    #RIGHT HAND
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),  # specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)  # skeleton
    )

    #LEFT HAND
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),  # specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)  # skeleton
    )
    #POSE
    '''p_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),#specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)#skeleton
    )'''
    try:
        try:
            right_hand = results.right_hand_landmarks.landmark
            right_hand_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
            row = right_hand_row
            hand="right"
        # extract left hand landmarks
        except AttributeError:
            left_hand = results.left_hand_landmarks.landmark
            left_hand_row = list(
               np.array([[1-landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
            row = left_hand_row
            hand="left"

        #print(hand)
        #Make Detections
    except:
        pass
        classification = 'Nil'
        probability = [0]
        hand="None"
    else:
        X = pd.DataFrame([row])
        classification = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        print(hand, classification)

    flip=cv2.flip(image, 1)
    cv2.putText(flip, classification, (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), cv2.LINE_AA)
    cv2.putText(flip, hand, (900, 100+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), cv2.LINE_AA)
    cv2.putText(flip,str(max(probability)), (900+150, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), cv2.LINE_AA)
    flip2 = cv2.flip(flip, 1)
    cv2.imshow('HandGesture Test', cv2.flip(flip2, 1))
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
cap.release()