import cv2
import threading
import facemesh
import client
                   
class RecordingThread (threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True
        
        self.cap = camera

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('opencv/save video/static/video.avi',fourcc, 20.0, (int(width), int(height)))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                frame = facemesh.cover_facemesh(frame)
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()

class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture('http://192.168.0.6:4747/video')
      
        # Initialize video recording environment
        self.is_record = False
        self.out = None
        
        # Thread for recording
        self.recordingThread = None

        # socket communication for getting tonify image
        self.client = client.client('127.0.0.1', 5000)

    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        ret, frame = self.cap.read()        
        if ret:
            mesh_frame = facemesh.cover_facemesh(frame)
            ret, jpeg = cv2.imencode('.jpg', mesh_frame)

            # Record video
            # if self.is_record:
            #     if self.out == None:
            #         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #         self.out = cv2.VideoWriter('./static/video.avi',fourcc, 20.0, (640,480))
                
            #     ret, frame = self.cap.read()
            #     if ret:
            #         self.out.write(frame)
            # else:
            #     if self.out != None:
            #         self.out.release()
            #         self.out = None  

            return jpeg.tobytes()
      
        else:
            return None

    def get_tonify_frame(self):
        ret, frame = self.cap.read()        
        if ret:
            height, width, _ = frame.shape
            result = facemesh.ret_face_landmarks(frame)
            if result.multi_face_landmarks is None:
                return None
            face_landmarks = result.multi_face_landmarks[0]

            self.client.send_image(frame)
            self.client.send_landmark(face_landmarks, width, height)
            tonify_frame = self.client.recv_image()
            ret, jpeg = cv2.imencode('.jpg', tonify_frame)

            return jpeg.tobytes()
      
        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
