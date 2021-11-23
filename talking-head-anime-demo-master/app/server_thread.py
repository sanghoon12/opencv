import threading
import socket
from landmark import dlib_face_landmark
from puppeteer import puppeteer
import numpy
import cv2

class TCP_server(threading.Thread):
    def __init__(self, sock_clnt, clients):
        threading.Thread.__init__(self)
        self.running = True

        self.sock_clnt = sock_clnt
        self.clients = clients
        
        self.frame = None
        self.face_landmarks = None
        self.tonify_frame = None
        self.puppet = puppeteer()

    def run(self):
        while self.running :
            self.recv_image()
            self.send_image()

                
    def recv_image(self):
        length = b''
        count = 16
        
        while count:
            buf = self.sock_clnt.recv(16)
            length += buf
            count -= len(buf)
        
        stringData = b''
        length = int(length)
        while length:
            buf = self.sock_clnt.recv(length)
            stringData += buf
            length -= len(buf)
            data = numpy.frombuffer(stringData, dtype='uint8')

        image = cv2.imdecode(data, 1)
        self.frame = image
        self.recv_landmark()


    def recv_landmark(self):
        self.face_landmarks = dlib_face_landmark()    
        landmark_list = []
    
        while len(landmark_list) < 936:
            buf = self.sock_clnt.recv(1024)
            if not buf:
                break
            for num in buf.split(' '.encode()):
                if num == b'':
                    continue
                landmark_list.append(int(num))

        while len(landmark_list) != 0:
            x = landmark_list.pop(0)
            y = landmark_list.pop(0)
            self.face_landmarks.put(x, y)


    def send_image(self):
        self.tonify_frame = self.puppet.get_posed_image(self.face_landmarks, self.frame)
        print(type(self.tonify_frame))
        print(self.tonify_frame)
        result, imgencode = cv2.imencode('.jpg', self.tonify_frame)
        data = numpy.array(imgencode)
        stringData = data.tobytes()
        self.sock_clnt.send(str(len(stringData)).ljust(16).encode())
        self.sock_clnt.send(stringData)
        


    # def terminate(self):
    #     self.sock_clnt.close()
    #     self.clients[self.sock_clnt] = False
    #     self.running = False








if __name__ == "__main__":
    HOST = '127.0.0.1'
    PORT = 5000

    sock_serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_serv.bind((HOST, PORT))
    sock_serv.listen(10)

    clients = {}
    while True:
        print("server wait...")
        try :
            sock_clnt, addr_clnt = sock_serv.accept()
            clients[sock_clnt] = True
            print("server :: connect ", sock_clnt)
            serv_thread = TCP_server(sock_clnt, clients)
            serv_thread.start()
        except ConnectionResetError:
            print("클라이언트 연결 종료")
    