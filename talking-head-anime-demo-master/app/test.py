import os
import sys

sys.path.append(os.getcwd())

from tkinter import Tk, Frame, LEFT, Label, BOTH, GROOVE, Button, filedialog, PhotoImage, messagebox

import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import dlib
import torch

from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from puppet.head_pose_solver import HeadPoseSolver
from poser.poser import Poser
from puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from tha.combiner import CombinerSpec
from tha.face_morpher import FaceMorpherSpec
from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from util import rgba_to_numpy_image, extract_pytorch_image_from_filelike



class HeadPoseSolver:
    def __init__(self, image_width=640, image_height=480):
        self.rotation_vec = None
        self.translation_vec = None

        self.head_pose_box_points = np.float32([[10, 10.5, 17.0],
                                                [7, 7.5, 7.0],
                                                [7, -7.5, 7.0],
                                                [10, -10.5, 17.0],
                                                [-10, 10.5, 17.0],
                                                [-7.0, 7.5, 7.0],
                                                [-7.0, -7.5, 7.0],
                                                [-10.0, -10.5, 17.0]])

        # From https://github.com/lincolnhard/head-pose-estimation/blob/master/video_test_shape.py
        self.face_model_points = np.float32([[6.825897, 6.760612, 4.402142],
                                             [1.330353, 7.122144, 6.903745],
                                             [-1.330353, 7.122144, 6.903745],
                                             [-6.825897, 6.760612, 4.402142],
                                             [5.311432, 5.485328, 3.987654],
                                             [1.789930, 5.393625, 4.413414],
                                             [-1.789930, 5.393625, 4.413414],
                                             [-5.311432, 5.485328, 3.987654],
                                             [2.005628, 1.409845, 6.165652],
                                             [-2.005628, 1.409845, 6.165652]])

        K = [
            image_width, 0.0, image_width / 2,
            0, image_width, image_height / 2,
            0, 0, 1
        ]
        self.camera_matrix = np.array(K).reshape(3, 3).astype(np.float32)

        D = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.distortion_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

    def solve_head_pose(self, face_landmarks):
        indices = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35]
        image_pts = np.zeros((len(indices), 2))
        for i in range(len(indices)):
            part = face_landmarks.part(indices[i])
            image_pts[i, 0] = part.x
            image_pts[i, 1] = part.y

        _, rotation_vec, translation_vec = cv2.solvePnP(self.face_model_points,
                                                        image_pts,
                                                        self.camera_matrix,
                                                        self.distortion_coeffs)
        projected_head_pose_box_points, _ = cv2.projectPoints(self.head_pose_box_points,
                                                              rotation_vec,
                                                              translation_vec,
                                                              self.camera_matrix,
                                                              self.distortion_coeffs)
        projected_head_pose_box_points = tuple(map(tuple, projected_head_pose_box_points.reshape(8, 2)))

        # Calculate euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        return projected_head_pose_box_points, euler_angles




video_capture = cv2.VideoCapture("http://192.168.0.6:4747/video")
face_detector = dlib.get_frontal_face_detector()


there_is_frame, frame = video_capture.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
faces = face_detector(rgb_frame)
face_rect = faces[0]
landmark_locator = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
face_landmarks = landmark_locator(rgb_frame, face_rect)
print("\n\n\nlandmarks\n", face_landmarks)
