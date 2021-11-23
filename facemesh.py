import cv2
import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from numpy.lib.function_base import iterable

lips_indexes = frozenset([0,13,14,17,37,39,40,61,78,80,
                        81,82,84,87,88,91,95,146,178,181,
                        185,191,267,269,270,291,308,310,311,312,
                        314,317,318,321,324,375,402,405,409,415])
lefteye_indexes = frozenset([249,263,362,373,374,380,381,382,384,385,
                        386,387,388,390,398,466])
righteye_indexes = frozenset([7,33,133,144,145,153,154,155,157,158,
                        159,160,161,163,173,246])
faceoval_indexes = frozenset([10,21,54,58,67,93,103,109,127,132,
                        136,148,149,150,152,162,172,176,234,251,
                        284,288,297,323,332,338,356,361,365,377,
                        378,379,389,397,400,454])
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh  

def ret_face_landmarks(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh  

    with mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = face_mesh.process(frame)
        return results

def cover_facemesh(frame):

    with mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = face_mesh.process(frame)

        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for i in lips_indexes:
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    cv2.circle(frame, (x, y), 3, (100,0,0))
                for i in lefteye_indexes:
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    cv2.circle(frame, (x, y), 3, (100,0,0))
                for i in righteye_indexes:
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    cv2.circle(frame, (x, y), 3, (100,0,0))
                for i in faceoval_indexes:
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    cv2.circle(frame, (x, y), 3, (100,0,0))  
    return frame