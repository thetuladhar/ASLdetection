#make sure you install opencv2, mediapipe and scikit-learn

import mediapipe as mp #import mediapipe
import cv2 #import opencv2
import numpy as np
import csv
import os



mp_drawing = mp.solutions.drawing_utils #mediapipe drawing helpers
mp_holistic = mp.solutions.holistic #mediapipe

#Get feed from webcam
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
    '''mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),#specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)#skeleton
    )'''

    class_name = "okay"
    try:
        #extract right hand landmarks
        right_hand=results.right_hand_landmarks.landmark
        right_hand_row=list(
           np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

        #extract left hand landmarks
        left_hand = results.left_hand_landmarks.landmark
        left_hand_row = list(
            np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
        #adding them together
        row = right_hand_row + left_hand_row
        row.insert(0, class_name)

        #print(len(row))#168
        with open('coords.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
    except:
        pass

    #FIND NUMBER OF COORDINATES
    #try:
    #    num_coords = len(results.right_hand_landmarks.landmark) + len(results.left_hand_landmarks.landmark)
    #    print(num_coords)#42 coordinates (21+21 =42)
    #except:
    #    pass

    #SETUP CSV data namining convetion
    '''landmarks = ['class']
    for val in range(1, 42+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    with open('right_coords.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)'''

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
cap.release()







