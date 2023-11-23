import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as hands:
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
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    def plotLandmark0(LandmarkIndex):
        try:
            landmark = results.multi_hand_landmarks[0].landmark[LandmarkIndex]
            xland = landmark.x
            yland = landmark.y
            # print("x",xland,"\ty",yland)
        except:
            pass
            # beyond frame
            xland = 0
            yland = 0

        # Normalize for plotting
        xnorm = int(xland * image.shape[1])
        ynorm = int(yland * image.shape[0])
        # DRAW
        cv2.circle(image, (xnorm, ynorm), radius=1, color=(225, 0, 255), thickness=1)
        cv2.putText(image, str(LandmarkIndex), ((xnorm), ynorm), cv2.FONT_HERSHEY_DUPLEX, .5, (207, 65, 1), 1)
    def plotLandmark1(LandmarkIndex):
        try:
            landmark = results.multi_hand_landmarks[1].landmark[LandmarkIndex]
            xland = landmark.x
            yland = landmark.y
            # print("x",xland,"\ty",yland)
        except:
            pass
            # beyond frame
            xland = 0
            yland = 0

        # Normalize for plotting
        xnorm = int(xland * image.shape[1])
        ynorm = int(yland * image.shape[0])
        # DRAW
        cv2.circle(image, (xnorm, ynorm), radius=1, color=(225, 0, 255), thickness=1)
        cv2.putText(image, str(LandmarkIndex), ((xnorm), ynorm), cv2.FONT_HERSHEY_DUPLEX, .5, (0,102,253), 1)

    for i in range(0,21):
        plotLandmark0(i)
        plotLandmark1(i)
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()