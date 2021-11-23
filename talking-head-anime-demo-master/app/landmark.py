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




    

def landmark_request_clnt():
    sock_serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_serv.connect((HOST, PORT))

    for i in range(468):
        x = int(face_landmarks.landmark[i].x * width)
        y = int(face_landmarks.landmark[i].y * height)
        sock_serv.send((str(x)+' ').encode())
        sock_serv.send((str(y)+' ').encode())

