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


class puppeteer() :
    cuda = torch.device('cuda')
    poser = MorphRotateCombinePoser256Param6(
        morph_module_spec=FaceMorpherSpec(),
        morph_module_file_name="data/face_morpher.pt",
        rotate_module_spec=TwoAlgoFaceRotatorSpec(),
        rotate_module_file_name="data/two_algo_face_rotator.pt",
        combine_module_spec=CombinerSpec(),
        combine_module_file_name="data/combiner.pt",
        device=cuda)
    head_pose_solver = HeadPoseSolver()
    torch_device = cuda
    pose_size = len(poser.pose_parameters())
    source_image = None
    posed_image = None
    current_pose = None
    last_pose = None


    def load_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="data/illust")
        if len(file_name) > 0:
            self.load_image_from_file(file_name)
    def load_image_from_file(self, file_name):
        # image = PhotoImage(file=file_name)
        # if image.width() != self.poser.image_size() or image.height() != self.poser.image_size():
        #     message = "The loaded image has size %dx%d, but we require %dx%d." \
        #               % (image.width(), image.height(), self.poser.image_size(), self.poser.image_size())
        #     messagebox.showerror("Wrong image size!", message)

        self.source_image = extract_pytorch_image_from_filelike(file_name).to(self.torch_device).unsqueeze(dim=0)
    
               



    def get_posed_image(self, face_landmarks, frame):
        self.load_image_from_file('app/illust/waifu_04_256.png')
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)


        if euler_angles is not None and self.source_image is not None:
            current_pose = torch.zeros(self.pose_size, device=self.torch_device)
            current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
            current_pose[1] = max(min(-euler_angles.item(1) / 15.0, 1.0), -1.0)
            current_pose[2] = max(min(euler_angles.item(2) / 15.0, 1.0), -1.0)

            if self.last_pose is None:
                last_pose = current_pose
            else:
                current_pose = current_pose * 0.5 + self.last_pose * 0.5
                last_pose = current_pose


            eye_min_ratio = 0.15
            eye_max_ratio = 0.25
            left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
            current_pose[3] = 1 - left_eye_normalized_ratio
            right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                            eye_min_ratio,
                                                                            eye_max_ratio)
            current_pose[4] = 1 - right_eye_normalized_ratio
            min_mouth_ratio = 0.02
            max_mouth_ratio = 0.3
            mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
            current_pose[5] = mouth_normalized_ratio
            current_pose = current_pose.unsqueeze(dim=0)


            posed_image = self.poser.pose(self.source_image, current_pose).detach().cpu()
            numpy_image = rgba_to_numpy_image(posed_image[0])
            # pil_image = PIL.Image.fromarray(np.uint8(np.rint(numpy_image * 255.0)), mode='RGBA')
            # photo_image = PIL.ImageTk.PhotoImage(image=pil_image)
            tonify_frame = np.uint8(np.rint(numpy_image * 255.0))
            tonify_frame = cv2.cvtColor(tonify_frame, cv2.COLOR_RGBA2BGR)
        return tonify_frame
