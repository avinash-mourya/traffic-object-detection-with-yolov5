from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
import cv2
import torch
from yolov5_detection import Yolov5Detector
import datetime
import numpy as np
# from yolov5_detection import detect
# import torch.backends.cudnn as cudnn
class Yolov5Tracker:
    def __init__(self):
        pass

    def tracker(self):
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=False)
        return deepsort

    def start_tracking(self,deepsort,xywhs,confs,im0):
        xywhs = torch.Tensor(xywhs)
        confs = torch.Tensor(confs)
        # Pass detections to deepsort
        outputs = deepsort.update(xywhs, confs, im0)
        return outputs

if __name__ == '__main__':
    detector = Yolov5Detector()
    track_v = Yolov5Tracker()
    model,stride,device,imgsz = detector.load_model()
    deepsort = track_v.tracker()
    video_capture = cv2.VideoCapture("traffic2.mp4")
    f = 0
    start_time = datetime.datetime.now()
    trackIds,car_count = [],0
    while True:
        ret, frame = video_capture.read()
        f += 1
        if ret != True:
            break
        xyxys, xywhs, labels, confs, img = detector.detect(frame, model, stride, device, imgsz)
        outputs= track_v.start_tracking(deepsort,xywhs,confs,img)
        fps = detector.calculate_fps(start_time,f)
        # print(outputs)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]

            for i, box in enumerate(bbox_xyxy):
                color = (0, 255, 0)
                id = int(identities[i])
                label = labels[::-1][i]
                # car_count = Obj_counting(id,trackIds,car_count)
                # trackIds.append(id)
                label_c = f"{label}{id} {confs[i][0]:.2f}"
                img = detector.draw_bboxes(img, label_c, box, 2, color, fps)
                # cv2.putText(img, f"car:{car_count}",(30,30), 0,1, [225, 255, 255], thickness=2,
                #             lineType=cv2.LINE_AA)

        cv2.imshow("",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()