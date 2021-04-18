import cv2
# import torch
import numpy as np
from yolov5_detection import Yolov5Detector
from tracking_with_yolov5 import Yolov5Tracker
import datetime
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class TrafficDetection:
    def __init__(self):
        pass

    def distance_to_camera(self, knownWidth, focalLength, pixelWidth):
        return (knownWidth * focalLength) / pixelWidth

    def Obj_counting(self, Id, label, trackIds, count, total):
        if Id in trackIds:
            count = count
            total = total
        else:
            count += 1
            total += 1
            trackIds.append(Id)
        return count,total

    def speed_estimation(self, position1, position2, fps):
        # t2 = datetime.datetime.now()
        if position1 != position2:
            #calculate centroid of vehicles
            x_c1 = (position1[0] + position1[2]) / 2
            y_c1 = (position1[1] + position1[3]) / 2
            x_c2 = (position2[0] + position2[2]) / 2
            y_c2 = (position2[1] + position2[3]) / 2
            # cv2.circle(img,(int(x_c1),int(y_c1)),5,(255,255,0),-1)
            h = np.abs(position2[3] - position2[1])
            # w1 = np.abs(position1[2] - position1[0])
            # w2 = np.abs(position2[2] - position2[0])
            distance = np.sqrt((math.pow((x_c2 - x_c1), 2) + math.pow((y_c2 - y_c1), 2)))
            #222222
            # z1 = self.distance_to_camera(1.5,668,w1)
            # z2 = self.distance_to_camera(1.5,668,w2)
            # distance = np.abs(z1-z2)
            # print("time "+"%.2f"%time_s)qq
            # print(f"height {h}")
            # ppm = h/1.6
            ppm = 45
            print(ppm)
            distance = distance * ppm
            # print("dtc",distance)
            # time_s = (t2 - t1).total_seconds()
            # t = f * 10
            # print(f"time {time_s}")
            kph = (distance*fps)*3.6
            # print(f"fps {fps}")
            # print("time " + "%.2f" % time_s)
            print("speed " + "%.2f" % kph)

            return kph
        else:
            return 0.0

    def main(self,video_capture,writeVideo_flag=False):
        f = 0
        fps = 0
        start_time = datetime.datetime.now()
        trackIds, position, speed_e = [],{},0
        car_count, total = (0,0)
        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        while True:
            ret, frame = video_capture.read()
            f += 1
            t1 = start_time
            # t1 = datetime.datetime.now()
            if ret != True:
                break
            xyxys, xywhs, labels, confs, img = detector.detect(frame, model, stride, device, imgsz)
            cv2.line(img, (100, 170), (800, 170), (0, 0, 255), 2)
            xywhss,confss,labelss = [],[],[]
            for det_box, xywh, conf, label in zip(xyxys, xywhs, confs, labels):
                det_box = det_box[:4]
                if int(det_box[3]) >= 140:
                    xywhss.append(xywh)
                    confss.append(conf)
                    labelss.append(label)
            outputs = track_v.start_tracking(deepsort, xywhss, confss, img)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                for i, (box, label) in enumerate(zip(bbox_xyxy, labelss)):
                    color = (0, 255, 0)
                    if label:
                        id = int(identities[i])
                        #Object counting
                        if label=="car":
                            car_count,total= self.Obj_counting(id, label, trackIds, car_count,total)
                            #speed estimation
                            fps = detector.calculate_fps(start_time, f)
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            if id in position.keys():
                                position1 = position[id]
                                position[id]=[x1, y1, x2, y2]

                                speed_e = self.speed_estimation(position1, [x1, y1, x2, y2], fps)
                                # self.speed_list[id].append(self.speed_e)
                            else:
                                position[id] = [x1, y1, x2, y2]
                                # self.speed_list[id] = []

                        label_c = f"{label}{id} {confss[i][0]:.2f}"
                        img = detector.draw_bboxes(img, label_c, box, 2, color, fps)
                        cv2.putText(img, f"car:{car_count}",(30,30), 0,1, [225, 255, 255], thickness=2,
                                    lineType=cv2.LINE_AA)
                        if speed_e != 0.0:
                            cv2.putText(img, f"{speed_e:.2f}km/hr", (int(box[2]), int(box[3])), 0, 0.7, [225, 255, 255], thickness=2,
                                        lineType=cv2.LINE_AA)
            cv2.imshow("", img)
            if writeVideo_flag:
                out.write(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        if writeVideo_flag:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Yolov5Detector()
    track_v = Yolov5Tracker()
    model,stride,device,imgsz = detector.load_model()
    deepsort = track_v.tracker()
    traffic_d = TrafficDetection()
    video_capture = cv2.VideoCapture("traffic.mp4")
    traffic_d.main(video_capture)

