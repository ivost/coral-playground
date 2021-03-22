"""
https://wiki.loliot.net/docs/lang/python/libraries/yolov4/python-yolov4-edge-tpu/

"""
import cv2
from yolov4.tflite import YOLOv4

yolo = YOLOv4()

yolo.config.parse_names("./dataset/coco.names")
yolo.config.parse_cfg("./config/yolov4-tiny-relu-tpu.cfg")
# yolo.summary()
yolo.load_tflite("./config/yolov4-tiny-int8_edgetpu.tflite")

video = "/home/ivo/Videos/airport-01.m4v"

#video = "/home/ivo/Videos/airport-03.mp4"

#video = "../videos/RTSP/rtsp0.mkv"
#video = "../videos/airport-03-HD.mp4"

size = (1920, 1080)
# size = (192, 108)

outp = "/home/ivo/video_out/a.avi"

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
vw = cv2.VideoWriter(outp, fourcc, 20.0, size)

# yolo.inference(video, is_image=False, cv_frame_size=size)
conf = 0.2
yolo.inference(video, is_image=False, prob_thresh=conf, video_writer=vw)

vw.release()

# yolo.inference("./data/kite.jpg")
# yolo.inference("./data/kite.jpg")

