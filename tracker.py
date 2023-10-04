import os.path
import sys
import threading

import cv2
import torch.cuda
import numpy as np

curpath = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curpath, 'yolov5')))
from mutils.draw import draw_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes, Profile, LOGGER, xyxy2xywh
from yolov5.utils.plots import colors
from yolov5.models.common import DetectMultiBackend
from deep_sort import DeepSort


def cross_product(line: list, point: list):
    if len(line) != 4 or len(point) != 2:
        return 0
    else:
        x1 = line[0] - point[0]
        y1 = line[1] - point[1]
        x2 = line[2] - point[0]
        y2 = line[3] - point[1]
        return x1 * y2 - x2 * y1


class Tracker(object):

    def __init__(self, weights='yolov5s.pt'):
        # params
        self.weights = weights
        self.imgsz = [640, 640]
        self.augment = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.line_thickness = 1
        self.half = False
        self.batch_size = 1

        # cross line counter
        self.__line__ = []
        self.__arrow__ = []
        self.__val__ = []
        self.__tracker_map__ = {}
        self.__in__ = set()
        self.__out__ = set()
        self.__lock__ = threading.Lock()

        # load model
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=None, fp16=self.half)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.batch_size, 3, *self.imgsz))
        print("load yolov5 successfully")
        # load deepsort
        cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.__deepsort__ = DeepSort(model_path="deep_sort/deep/checkpoint/ckpt.t7", use_cuda=cuda)
        print(f"load deepsort successfully cuda={cuda}")

    def run(self, image):
        dt = (Profile(), Profile(), Profile(), Profile())
        # precess
        with dt[0]:
            img0 = image.copy()
            img = letterbox(img0, self.imgsz, stride=self.stride)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.model.device)
            img = img.half() if self.model.fp16 else img.float()
            img /= 255.0
            if len(img.shape) == 3:
                img = img[None]
        # run
        with dt[1]:
            pred = self.model(img, augment=self.augment)[0]
        with dt[2]:
            pred = non_max_suppression(pred,
                                       self.conf_thres,
                                       self.iou_thres,
                                       self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)
        det = pred[0]  # batch size = 1
        det = det[(det[:, 5] == 0).nonzero().squeeze(1)]  # filter person det..
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()
            box_xywh = xyxy2xywh(det[:, :4]).cpu()
            confs = det[:, 4:5].cpu()
            with dt[3]:
                outputs = self.__deepsort__.update(box_xywh, confs, image)
                # cross line
                width = image.shape[1]
                height = image.shape[0]
                if len(self.__val__) > 0:
                    for i, op in enumerate(outputs):
                        # 先以中心坐标计算cp值
                        center = [
                            (op[0] + op[2]) / (2 * width),
                            (op[1] + op[3]) / (2 * height)
                        ]
                        cpv = cross_product(self.__line__, center)
                        # 判断是否在map中
                        tracker_id = op[4]
                        if tracker_id in self.__tracker_map__.keys():
                            # 存在, 当cp值变化时计数并判断进出
                            if self.__tracker_map__[tracker_id] * cpv < 0:

                                if self.__val__[1] * cpv > 0:
                                    # 与 val[1] 一致，则为进入
                                    self.__in__.add(tracker_id)
                                    if tracker_id in self.__out__:
                                        self.__out__.remove(tracker_id)
                                else:
                                    # 与 val[1] not match，则为退出
                                    self.__out__.add(tracker_id)
                                    if tracker_id in self.__in__:
                                        self.__in__.remove(tracker_id)

                        else:
                            # 不存在，直接存储
                            self.__tracker_map__[tracker_id] = cpv
                    # draw line and arrow
                    p1 = [int(self.__line__[0] * width), int(self.__line__[1] * height)]
                    p2 = [int(self.__line__[2] * width), int(self.__line__[3] * height)]
                    ar1 = [int(self.__arrow__[0] * width), int(self.__arrow__[1] * height)]
                    ar2 = [int(self.__arrow__[2] * width), int(self.__arrow__[3] * height)]
                    cv2.line(image, p1, p2, [0,255,0], thickness=3)
                    cv2.arrowedLine(image, ar1, ar2, color=[0, 255,255], thickness=3)
                    cv2.putText(image,
                                text=f'IN:{len(self.__in__)} | OUT:{len(self.__out__)}',
                                org=[10, 40],
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                color=[0,255,255],
                                fontScale=3,
                                thickness=3)
                if len(outputs):
                    image = draw_boxes(image, outputs[:, :4], outputs[:, -1])
        LOGGER.info(f"infer:{dt[1].dt * 1E3:.1f}ms nms:{dt[2].dt * 1E3:.1f}ms deepsort:{dt[3].dt * 1E3:.1f}ms")
        #LOGGER.info(f"{self.__val__}")
        return image

    def update_config(self, line: list, arrow: list):
        self.__lock__.acquire()
        try:
            self.__line__ = line
            self.__arrow__ = arrow
            self.__val__ = []
            self.__tracker_map__ = {}
            self.__in__ = set()
            self.__out__ = set()
            if len(line) == 4 and len(arrow) == 4:
                self.__val__ = [
                    cross_product(line, arrow[:2]),
                    cross_product(line, arrow[-2:])
                ]
        finally:
            self.__lock__.release()


# test
if __name__ == '__main__':
    det = Tracker(weights='./yolov5s.pt')
    img = cv2.imread("yolov5/data/images/bus.jpg")
    alarm, image = det.run(img)

    cv2.imshow("test", image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyWindow("test")
