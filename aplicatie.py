import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import pickle
import dlib
import math
import numpy as np

model = pickle.load(open('fer_poly_degree2_5_-15_no_contempt_fear_disgust.sav','rb'))
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
faceDet = cv2.CascadeClassifier(r"haarcascade_frontalface_alt2.xml")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


class Application:
    counter_expresii = 0
    emotion = 0
    predictie = []
    emotions = {
            "0": "Neutral",
            "1": "Happy",
            "2": "Surpised",
            "3": "Sad",
            "4": "Angry",
            "5": "Disgusted",
            "6": "Afraid",
            "7": "Contempt"
    }
    
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = Video(self.video_source)

        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        self.btn_screenshot=tkinter.Button(window, text="Screenshot", width=50, command=self.screenshot)
        self.btn_screenshot.pack(anchor=tkinter.CENTER, expand=True)
        
        self.var1 = tkinter.IntVar()
        
        self.cb_dots=tkinter.Checkbutton(window, text="Display Facial Landmarks", variable = self.var1, onvalue=1, offvalue=0)
        self.cb_dots.pack(anchor=tkinter.CENTER)
        
        self.var2 = tkinter.IntVar()
        
        self.face_rectangle=tkinter.Checkbutton(window, text="Display Facial Expression", variable = self.var2, onvalue=1, offvalue=0)
        self.face_rectangle.pack(anchor=tkinter.CENTER)
        
        self.delay = 15
        self.update()
        
        self.window.mainloop()
        
        

    def screenshot(self):
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def draw_landmarks(self, image):
        detections = detector(image, 1)
        for k,d in enumerate(detections):
            shape = predictor(image, d)
            for i in range(0,68):
                    cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
    
    def get_landmarks(self, image):
            data = {}
            prediction_data = []
            detections = []
            facefeatures = faceDet.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in facefeatures:
                face_image = image[y:y+h, x:x+w]
                try:
                    resize = cv2.resize(face_image, (48, 48))
                    detections = detector(resize, 1)
                except:
                    pass
            
            if len(detections) < 1:
                prediction_data.append("error")
            else:
                for k,face in enumerate(detections):
                    shape_predict = predictor(resize, face)
                    x_coord = []
                    y_coord = []
                    for i in range(0,68):
                        x_coord.append(float(shape_predict.part(i).x))
                        y_coord.append(float(shape_predict.part(i).y))
                    xmean = np.mean(x_coord)
                    ymean = np.mean(y_coord)
                    x_center = [(x-xmean) for x in x_coord]
                    y_center = [(y-ymean) for y in y_coord]
                    features = []
                    for x, y, w, z in zip(x_center, y_center, x_coord, y_coord):
                        features.append(w)
                        features.append(z)
                        mean_np = np.asarray((ymean,xmean))
                        coord_np = np.asarray((z,w))
                        distance = np.linalg.norm(coord_np-mean_np)
                        features.append(distance)
                        features.append((math.atan2(y, x)*360)/(2*math.pi))
                    data['features'] = features
                    prediction_data.append(data['features'])
            return prediction_data

    def update(self):
        self.counter_expresii+=1
        start = time.time()
        
        ret, frame = self.vid.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
       
        if self.var1.get() == 1:
            self.draw_landmarks(frame)
        
        if self.var2.get() == 1:
            detections = detector(frame, 1)
            for i, d in enumerate(detections):
                x1, y1, x2, y2 = d.left(), d.top(), d.right() + 1, d.bottom() + 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText((frame),self.emotions[str(self.emotion)],(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)
       
        prediction_data = self.get_landmarks(clahe_image)
        
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
       
        if prediction_data[0] == "error":
            print("no face detected on this one")
        else:
            prediction_np = np.array(prediction_data)
            result = model.predict(prediction_np)
            self.predictie.append(result)
            
            if(self.counter_expresii%5==0):
                predictie_np = np.array(self.predictie)
                c = np.bincount(predictie_np[:, 0])
                self.emotion = np.argmax(c)
                print(self.emotion)
                self.predictie = []
        end = time.time()
        print(end - start)

        self.window.after(self.delay, self.update)


class Video:
    def __init__(self, video_source=0):
       
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Error: cannot open webcam.", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

Application(tkinter.Tk(), "Recunoasterea expresiilor faciale in imagini portret")