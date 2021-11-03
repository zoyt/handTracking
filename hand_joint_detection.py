# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math
from tensorflow.keras.models import load_model
import os

# also "pip install wrapt==1.12.1" ref by tensorflow
# also "pip install msvc-runtime" referenced by mp
# also "conda install h5py" referenced by tensorflow

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

#dir = os.path.normcase()

# Load the gesture recognizer model
model = load_model("C:/Users/Owen Hoyt/dev/hand_tracking/Vidvan/mp_hand_gesture")

# Load class names
f = open('C:/Users/Owen Hoyt/dev/hand_tracking/Vidvan/mp_hand_gesture/gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(1)


def angle_2p_3d(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

    v1mag = np.sqrt([v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
    v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    angle_rad = np.arccos(res)

    return math.degrees(angle_rad)


while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)

    className = ''
    # THUMB_CMC_ANGLE = 0
    # THUMB_MPC_ANGLE = 0
    # THUMB_IP_ANGLE = 0

    THUMB_CMC_ANGLE, THUMB_MPC_ANGLE, THUMB_IP_ANGLE = 0, 0, 0
    INDEX_MCP_ANGLE, INDEX_PIP_ANGLE, INDEX_DIP_ANGLE = 0, 0, 0
    MIDDLE_MPC_ANGLE, MIDDLE_PIP_ANGLE, MIDDLE_DIP_ANGLE = 0, 0, 0
    RING_MPC_ANGLE, RING_PIP_ANGLE, RING_DIP_ANGLE = 0, 0, 0
    PINKY_MPC_ANGLE, PINKY_PIP_ANGLE, PINKY_DIP_ANGLE = 0, 0, 0

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        coords = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                lmz = int(lm.z * -10000)

                landmarks.append([lmx, lmy])

                coords.append([lmx, lmy, lmz])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # plot landmarks
            #mpDraw.plot_landmarks(handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
            # print(landmarks)
            print(coords)

            THUMB_CMC_ANGLE = angle_2p_3d(coords[0], coords[1], coords[2])
            THUMB_MPC_ANGLE = angle_2p_3d(coords[1], coords[2], coords[3])
            THUMB_IP_ANGLE = angle_2p_3d(coords[2], coords[3], coords[4])

            INDEX_MCP_ANGLE = angle_2p_3d(coords[0], coords[5], coords[6])
            INDEX_PIP_ANGLE = angle_2p_3d(coords[5], coords[6], coords[7])
            INDEX_DIP_ANGLE = angle_2p_3d(coords[6], coords[7], coords[8])

            MIDDLE_MPC_ANGLE = angle_2p_3d(coords[0], coords[9], coords[10])
            MIDDLE_PIP_ANGLE = angle_2p_3d(coords[9], coords[10], coords[11])
            MIDDLE_DIP_ANGLE = angle_2p_3d(coords[10], coords[11], coords[12])

            RING_MPC_ANGLE = angle_2p_3d(coords[0], coords[13], coords[14])
            RING_PIP_ANGLE = angle_2p_3d(coords[13], coords[14], coords[15])
            RING_DIP_ANGLE = angle_2p_3d(coords[14], coords[15], coords[16])

            PINKY_MPC_ANGLE = angle_2p_3d(coords[0], coords[17], coords[18])
            PINKY_PIP_ANGLE = angle_2p_3d(coords[17], coords[18], coords[19])
            PINKY_DIP_ANGLE = angle_2p_3d(coords[18], coords[19], coords[20])

    # TODO: CALCULATE ANGLES BETWEEN CONNECTED JOINTS IN LANDMARKS

    reshape = np.array([THUMB_CMC_ANGLE, THUMB_MPC_ANGLE, THUMB_IP_ANGLE,
                        INDEX_MCP_ANGLE, INDEX_PIP_ANGLE, INDEX_DIP_ANGLE,
                        MIDDLE_MPC_ANGLE, MIDDLE_PIP_ANGLE, MIDDLE_DIP_ANGLE,
                        RING_MPC_ANGLE, RING_PIP_ANGLE, RING_DIP_ANGLE,
                        PINKY_MPC_ANGLE, PINKY_PIP_ANGLE, PINKY_DIP_ANGLE
                        ]).reshape(5, 3)
    reshape = reshape.astype(int)
    reshape = np.rot90(reshape)
    reshape = np.flipud(reshape)

    # print(type(disp1))
    # show the prediction on the frame
    # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #                1, (0,0,255), 2, cv2.LINE_AA)
    # show the individual angle measures by order of thumb, index, middle, ring, pinky, with all three joints below

    text = str(reshape[:][:])
    y0, dy = 50, 15
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # cv2.putText(frame, str(reshape[:][:]),  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
