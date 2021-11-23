import socket
import numpy
import cv2
import facemesh

class client():

    def __init__(self, HOST, PORT) -> None:
        self.sock_serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.HOST = HOST
        self.PORT = PORT
        self.running = True
        self.sock_serv.connect((self.HOST, self.PORT))


    def recv_image(self):
        length = b''
        count = 16
        while count:
            buf = self.sock_serv.recv(16)
            length += buf
            count -= len(buf)

        stringData = b''
        length = int(length)
        while length:
            buf = self.sock_serv.recv(length)
            stringData += buf
            length -= len(buf)
            data = numpy.frombuffer(stringData, dtype='uint8')

        image = cv2.imdecode(data, 1)
        return image

    def send_image(self, frame):
        result, imgencode = cv2.imencode('.jpg', frame)
        print(type(frame))
        print(frame)
        data = numpy.array(imgencode)
        stringData = data.tostring()
        self.sock_serv.send(str(len(stringData)).ljust(16).encode())
        self.sock_serv.send(stringData)



    def send_landmark(self, face_landmarks, width, height):
        for i in range(468):
            x = int(face_landmarks.landmark[i].x * width)
            y = int(face_landmarks.landmark[i].y * height)
            self.sock_serv.send((str(x)+' ').encode())
            self.sock_serv.send((str(y)+' ').encode())

    def terminate(self) -> None:
        self.sock_serv.close()
        self.running = False