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

import mediapipe as mp

def ret_multi_facemesh(frame):
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
        return results.multi_face_landmarks



class point:
    x = int
    y = int

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return point(self.x + other.x, self.y + other.y)
    def __radd__(self, other):
        return point(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return point(self.x - other.x, self.y - other.y)    
    def __rsub__(self, other):
        return point(self.x - other.x, self.y - other.y)     
    def __truediv__(self, other):
        return point(self.x // other, self.y // other) 
    def __rtruediv__(self, other):
        return point(self.x // other, self.y // other)

class dlib_face_landmark:
    point_list = []
    i = 0
    def put(self, x, y):
        self.point_list.insert(self.i, point(x, y))
        self.i += 1
    def part(self, i):
        return self.point_list[i]

def mediapipe_to_dlib_face_landmarks(media_face_landmarks, height, width):
    dlib_face_landmarks = dlib_face_landmark()

    for i in range(468):
        x = int(media_face_landmarks.landmark[i].x * width)
        y = int(media_face_landmarks.landmark[i].y * height)
        dlib_face_landmarks.put(x, y)

    return dlib_face_landmarks


class PuppeteerApp:
    def __init__(self,
                 master,
                 poser: Poser,
                 video_capture,
                 torch_device: torch.device):
        self.master = master
        self.poser = poser
        self.video_capture = video_capture
        self.torch_device = torch_device
        self.head_pose_solver = HeadPoseSolver()

        self.master.title("Puppeteer")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        top_frame = Frame(self.master)
        top_frame.pack()

        if True:
            source_image_frame = Frame(top_frame, width=256, height=256)
            source_image_frame.pack_propagate(0)
            source_image_frame.pack(side=LEFT)

            self.source_image_label = Label(source_image_frame, text="Nothing yet!")
            self.source_image_label.pack(fill=BOTH, expand=True)

        if True:
            control_frame = Frame(top_frame, width=256, height=192)
            control_frame.pack_propagate(0)
            control_frame.pack(side=LEFT)

            self.video_capture_label = Label(control_frame, text="Nothing yet!")
            self.video_capture_label.pack(fill=BOTH, expand=True)

        if True:
            posed_image_frame = Frame(top_frame, width=256, height=256)
            posed_image_frame.pack_propagate(0)
            posed_image_frame.pack(side=LEFT, fill='y')

            self.posed_image_label = Label(posed_image_frame, text="Nothing yet!")
            self.posed_image_label.pack(fill=BOTH, expand=True)

        bottom_frame = Frame(self.master)
        bottom_frame.pack(fill='x')

        self.load_source_image_button = Button(bottom_frame, text="Load Image ...", relief=GROOVE,
                                               command=self.load_image)
        self.load_source_image_button.pack(fill='x')

        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None

        self.master.after(50, self.update_image())

    def load_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="data/illust")
        if len(file_name) > 0:
            self.load_image_from_file(file_name)

    def load_image_from_file(self, file_name):
        image = PhotoImage(file=file_name)
        if image.width() != self.poser.image_size() or image.height() != self.poser.image_size():
            message = "The loaded image has size %dx%d, but we require %dx%d." \
                      % (image.width(), image.height(), self.poser.image_size(), self.poser.image_size())
            messagebox.showerror("Wrong image size!", message)
        self.source_image_label.configure(image=image, text="")
        self.source_image_label.image = image
        self.source_image_label.pack()

        self.source_image = extract_pytorch_image_from_filelike(file_name).to(self.torch_device).unsqueeze(dim=0)

    def update_image(self):
        there_is_frame, frame = self.video_capture.read()
        if not there_is_frame:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        euler_angles = None
        face_landmarks = None
        multi_face_landmarks = ret_multi_facemesh(rgb_frame)    
        if(multi_face_landmarks == None):
            self.master.after(500//60, self.update_image)
            return
        face_landmarks = mediapipe_to_dlib_face_landmarks(multi_face_landmarks[0], height, width)
        face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)
        self.draw_face_landmarks(rgb_frame, face_landmarks)
        # self.draw_face_box(rgb_frame, face_box_points)

        resized_frame = cv2.flip(cv2.resize(rgb_frame, (192, 256)), 1)
        pil_image = PIL.Image.fromarray(resized_frame, mode='RGB')
        photo_image = PIL.ImageTk.PhotoImage(image=pil_image)
        self.video_capture_label.configure(image=photo_image, text="")
        self.video_capture_label.image = photo_image
        self.video_capture_label.pack()

        if euler_angles is not None and self.source_image is not None:
            self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
            self.current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
            self.current_pose[1] = max(min(-euler_angles.item(1) / 15.0, 1.0), -1.0)
            self.current_pose[2] = max(min(euler_angles.item(2) / 15.0, 1.0), -1.0)

            if self.last_pose is None:
                self.last_pose = self.current_pose
            else:
                self.current_pose = self.current_pose * 0.5 + self.last_pose * 0.5
                self.last_pose = self.current_pose

            eye_min_ratio = 0.15
            eye_max_ratio = 0.25
            left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
            self.current_pose[3] = 1 - left_eye_normalized_ratio
            right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                            eye_min_ratio,
                                                                            eye_max_ratio)
            self.current_pose[4] = 1 - right_eye_normalized_ratio

            min_mouth_ratio = 0.02
            max_mouth_ratio = 0.3
            mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
            self.current_pose[5] = mouth_normalized_ratio

            self.current_pose = self.current_pose.unsqueeze(dim=0)

            posed_image = self.poser.pose(self.source_image, self.current_pose).detach().cpu()
            numpy_image = rgba_to_numpy_image(posed_image[0])
            pil_image = PIL.Image.fromarray(np.uint8(np.rint(numpy_image * 255.0)), mode='RGBA')
            photo_image = PIL.ImageTk.PhotoImage(image=pil_image)
            self.posed_image_label.configure(image=photo_image, text="")
            self.posed_image_label.image = photo_image
            self.posed_image_label.pack()

        self.master.after(500//60, self.update_image)

    def on_closing(self):
        self.video_capture.release()
        self.master.destroy()

    def draw_face_box(self, frame, face_box_points):
        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        temp1 = []
        temp2 = []
        for start, end in line_pairs:
            temp1.append(int(face_box_points[start][0]))
            temp1.append(int(face_box_points[start][1]))
            temp2.append(int(face_box_points[end][0]))
            temp2.append(int(face_box_points[end][1]))
            cv2.line(frame, temp1, temp2, (255, 0, 0), thickness=2)

    def draw_face_landmarks(self, frame, face_landmarks):
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


        for i in lips_indexes:
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)
        for i in lefteye_indexes:
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)
        for i in righteye_indexes:
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)
        for i in faceoval_indexes:
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)            


if __name__ == "__main__":
    cuda = torch.device('cuda')
    poser = MorphRotateCombinePoser256Param6(
        morph_module_spec=FaceMorpherSpec(),
        morph_module_file_name="data/face_morpher.pt",
        rotate_module_spec=TwoAlgoFaceRotatorSpec(),
        rotate_module_file_name="data/two_algo_face_rotator.pt",
        combine_module_spec=CombinerSpec(),
        combine_module_file_name="data/combiner.pt",
        device=cuda)

    
    video_capture = cv2.VideoCapture("http://192.168.0.6:4747/video")

    master = Tk()
    PuppeteerApp(master, poser, video_capture, cuda)
    master.mainloop()
