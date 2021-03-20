import cv2
# import torch
import numpy as np
from yolov5_detection import Yolov5Detector
from tracking_with_yolov5 import Yolov5Tracker
import datetime
import math

class TrafficDetection:
    def __init__(self):
        pass

    def distance_to_camera(self, knownWidth, focalLength, pixelWidth):
        return (knownWidth * focalLength) / pixelWidth

    def Obj_counting(self,Id,label,trackIds,count,total):
        if Id in trackIds:
            count = count
            total = total
        else:
            count += 1
            total += 1
            trackIds.append(Id)
        return count,total

    def speed_estimation(self, position1, position2, fps) -> float:
        if position1 != position2:
            # time_s = (datetime.datetime.now() - t2).total_seconds()
            x_c1 = (position1[0] + position1[2]) / 2
            y_c1 = (position1[1] + position1[3]) / 2
            x_c2 = (position2[0] + position2[2]) / 2
            y_c2 = (position2[1] + position2[3]) / 2
            h = np.abs(position2[3] - position2[1])
            distance = np.sqrt((math.pow((x_c2 - x_c1), 2) + math.pow((y_c2 - y_c1), 2)))
            # height_p = np.abs(position1[3]-position1[1])
            # print("time "+"%.2f"%time_s)
            a = 1.6 / h
            # ppm=4.4
            # print(a)
            distance = distance * a
            # print("dtc",distance)
            kph = (distance/fps) * 3.6
            # print("distance " + "%.2f" % distance)
            # print("speed " + "%.2f" % kph)
            return kph
        else:
            return 0.0

    def main(self,video_capture,writeVideo_flag=False):
        f = 0
        start_time = datetime.datetime.now()
        trackIds,car_count,position,speed_e = [],0,{},0
        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))

        while True:
            ret, frame = video_capture.read()
            f += 1
            if ret != True:
                break
            xyxys, xywhs, labels, confs, img = detector.detect(frame, model, stride, device, imgsz)
            cv2.line(img, (120, 145), (380, 145), (0, 0, 255), 2)
            xywhss,confss,labelss = [],[],[]
            for det_box, xywh, conf, label in zip(xyxys, xywhs, confs, labels):
                det_box = det_box[:4]
                if int(det_box[3]) >= 145 and int(det_box[2]) <= 400:
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
                            car_count = self.Obj_counting(id,trackIds,car_count)
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
                        if speed_e!=0.0:
                            cv2.putText(img, f"speed:{speed_e:.2f}", (int(box[2]), int(box[3])), 0, 0.7, [225, 255, 255], thickness=2,
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

