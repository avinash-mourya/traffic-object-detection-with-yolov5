from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import cv2
import numpy as np
from yolov5_detection import Yolov5Detector
from tracking_with_yolov5 import Yolov5Tracker
from traffic_object_detection import TrafficDetection
import time
class GUI:
    def __init__(self, vs):
        self.vs = vs
        # self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.root = tki.Tk()
        self.panel = None
        self.fps = None
        self.counting_result=None
        self.two_w,self.three_w,self.four_w,self.truck,self.bus,self.total=None,None,None,None,None,None
        # btn = tki.Button(self.root, text="save",
        #                  command=self.saveVideo)
        # btn.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        font = ("Arial",25)
        self.panel = tki.Frame(self.root)
        self.panel.pack(side="top", padx=10, pady=10)
        self.canvas = tki.Label(self.panel,text="loading...",anchor="center",font=font,fg="blue")
        self.canvas.pack(side="left", padx=10, pady=10)
        self.counting_result = tki.Frame(self.root)
        self.counting_result.pack(side="bottom", padx=10, pady=10)
        self.Quit_btn = tki.Button(self.counting_result,text="Quit",font=("Arial",12), command=self.onClose,bg="red",fg="white",width=6)
        self.Quit_btn.grid(row=2,column=5)
        ##threading
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoStream, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("Traffic")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        # self.root.geometry("640x360")

    def videoStream(self):
        try:
            if not vs.isOpened():
                print("video source not correct")
            f = 0
            start_time = datetime.datetime.now()
            # {"motorcycle": [], "car": [], "auto": [], "truck": [], "bus": []}
            trackIds, position, speed_e,fps = [], {}, 0,0.0
            two_w,three_w,four_w,truck,bus,total=0,0,0,0,0,0
            while(self.vs.isOpened()):
                ret,self.frame = self.vs.read()
                if ret != True:
                    break
                f+=1
                xyxys, xywhs, labels, confs, img = detector.detect(self.frame, model, stride, device, imgsz)
                cv2.line(img, (30, 140), (500, 140), (0, 0, 255), 2)
                xywhss,xyxyss,confss, labelss = [], [], [],[]
                for det_box, xywh, conf, label in zip(xyxys, xywhs, confs, labels):
                    det_box = det_box[:4]
                    if int(det_box[3]) >= 140 and int(det_box[2]) <= 550:
                        xywhss.append(xywh)
                        confss.append(conf)
                        xyxyss.append(det_box)
                        labelss.append(label)
                outputs = track_v.start_tracking(deepsort, xywhss, confss, img)
                print(outputs)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]

                    for i, box in enumerate(bbox_xyxy):
                        color = (0, 255, 0)
                        id = int(identities[i])
                        label = labelss[::-1][i]

                        # Object counting
                        if label == "motorcycle":
                            two_w,total = traffic_d.Obj_counting(id,label,trackIds,two_w,total)
                        elif label=="auto":
                            three_w,total = traffic_d.Obj_counting(id,label,trackIds,three_w,total)
                        elif label=="car":
                            four_w,total = traffic_d.Obj_counting(id,label,trackIds,four_w,total)
                        elif label=="truck":
                            truck,total=traffic_d.Obj_counting(id,label,trackIds,truck,total)
                        elif label=="bus":
                            bus,total = traffic_d.Obj_counting(id,label,trackIds,bus,total)

                        fps = detector.calculate_fps(start_time, f)
                        label_c = f"{label}{id} {confss[i][0]:.2f}"
                        img = detector.draw_bboxes(img, label_c, box, 2, color, fps)

                img = cv2.resize(img,(650,360))
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                font = ("Arial", 12)
                self.canvas.configure(image=image)
                self.canvas.image = image
                result = tki.Label(self.counting_result, text=f"Counting Results", width=12, font=font,
                                   anchor="center", fg="blue")
                result.grid(row=0, column=2, padx=2)
                # result.pack(padx=10, pady=10)
                if self.two_w is None:
                    self.two_w = tki.Label(self.counting_result,text=f"Two Wheeler \n\n{two_w}",width=13,font=font,anchor="center",bg="#8080c0",fg="white")
                    self.two_w.grid(row =1,column =0,padx=2)
                else:
                    self.two_w.configure(text=f"Two Wheeler\n\n{two_w}")

                if self.three_w is None:
                    self.three_w = tki.Label(self.counting_result,text=f"Three Wheeler\n\n{three_w}",font=font,width=13,anchor="center",bg="#8080c0",fg="white")
                    self.three_w.grid(row=1,column=1,padx=2)
                else:
                    self.three_w.configure(text=f"Three Wheeler\n\n{three_w}")

                if self.four_w is None:
                    self.four_w = tki.Label(self.counting_result,text=f"Four Wheeler\n\n{four_w}",width=13,font=font,anchor="center",bg="#8080c0",fg="white")
                    self.four_w.grid(row =1,column =2,padx= 2)
                else:
                    self.four_w.configure(text=f"Four Wheeler\n\n{four_w}")

                if self.truck is None:
                    self.truck = tki.Label(self.counting_result,text=f"Truck\n\n{truck}",font=font,width=10,anchor="center",bg="#8080c0",fg="white")
                    self.truck.grid(row=1,column=3,padx=1)
                else:
                    self.truck.configure(text=f"Truck\n\n{truck}")

                if self.bus is None:
                    self.bus = tki.Label(self.counting_result,text=f"Bus\n\n{bus}",font=font,width=10,anchor="center",bg="#8080c0",fg="white")
                    self.bus.grid(row=1,column=4,padx=2)
                else:
                    self.bus.configure(text=f"Bus\n\n{bus}")

                if self.total is None:
                    self.total = tki.Label(self.counting_result, text=f"Total Vehicle\n\n{total}", font=font, width=10, anchor="center",
                                         bg="#8080c0", fg="white")
                    self.total.grid(row=1, column=5,pady=2)
                else:
                    self.total.configure(text=f"Total Vehicle\n\n{total}")

                if self.fps is None:
                    self.fps = tki.Label(self.counting_result, text=f"FPS\n\n{fps:.2f}", font=font, width=13, anchor="center",
                                         bg="#8080c0", fg="white")
                    self.fps.grid(row=2, column=0,pady=2)
                else:
                    self.fps.configure(text=f"FPS\n\n{fps:.2f}")

        except RuntimeError:
            print("[INFO] caught a RuntimeError")


    def saveVideo(self):
        pass

    def onClose(self):
        print("[INFO] closing...")
        # self.stopEvent.set()
        self.vs.release()
        self.root.quit()
        # self.root.destroy()

if __name__ == '__main__':
    detector = Yolov5Detector()
    track_v = Yolov5Tracker()
    model, stride, device, imgsz = detector.load_model()
    deepsort = track_v.tracker()
    traffic_d = TrafficDetection()
    vs = cv2.VideoCapture("traffic3.mp4")
    time.sleep(2.0)
    # start the app
    gui = GUI(vs)
    gui.root.mainloop()