"""
https://wiki.loliot.net/docs/lang/python/libraries/yolov4/python-yolov4-edge-tpu/

"""
import datetime
import logging as log
import sys
import os
from pathlib import Path

import cv2

# from YOLOv4.tflite import YOLOv4


class MyYolo:

    def __init__(self):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)

        self.yolo = YOLOv4()

        self.yolo.config.parse_names("./dataset/coco.names")
        self.yolo.config.parse_cfg("./config/YOLOv4-tiny-relu-tpu.cfg")
        self.yolo.summary()
        self.yolo.load_tflite("./config/YOLOv4-tiny-int8_edgetpu.tflite")

        self.temp_video = "/tmp/a.avi"
        self.output_file = "/home/ivo/a.mp4"
        self.output_dir = "/home/ivo/video_out"
        self.model_name = "YOLOv4-edgetpu"
        self.output_type = "mp4"

        self.fps = 24.0
        self.conf = 0.2
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        self.size = (1920, 1080)
        self.input = "rtsp"

    def run(self, video):

        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        if not ret:
            log.error("Capture error")
            raise EOFError("Capture error " + video)
        # self.fps = cap.get(cv2.CAP_PROP_FPS)
        # cap.release()
        self.input = video
        self.output_file = self._output_filename()

        shape = frame.shape
        self.size = (shape[1], shape[0])
        log.debug(f"fps {self.fps}, frame size {self.size}")
        # vw = cv2.VideoWriter(self.temp_video, self.fourcc, self.fps, self.size)
        # self.yolo.inference(video, is_image=False, prob_thresh=self.conf, video_writer=vw)
        self.yolo.inference(video, is_image=False, prob_thresh=self.conf)
        # vw.release()
        self._convert_to_mp4()

    def _output_filename(self):
        dir = self.output_dir
        assert (os.path.isdir(dir))
        assert (os.path.exists(dir))

        ts = datetime.datetime.now().isoformat(timespec='seconds')
        ts = ts.replace(":", "-")
        if str(self.input).startswith("rtsp:"):
            name = f"rtsp"
        else:
            name = Path(self.input).resolve().name
            assert len(name) > 0
            if '.' in name:
                el = name.split('.')
                name = el[-2]

        fname = f"{name}_{ts}_{self.model_name}.{self.output_type}"

        self.output_file = os.path.join(dir, fname)
        log.debug(f"generate_output_file {self.input} -> {self.output_file}")
        return self.output_file

    def _convert_to_mp4(self):
        import subprocess
        inp = self.temp_video
        if not os.path.exists(inp):
            log.info(f"{inp} not found")
            return
        outp = self.output_file
        log.info(f"converting {inp} to {outp}")
        # ffmpeg -i debug.avi -y a.mp4
        res = subprocess.run(["ffmpeg", "-i", inp, "-y", outp])
        # log.info(str(res))
        if ("returncode=0" in str(res)) and os.path.exists(outp):
            log.debug(f"Deleting {inp}")
            os.remove(inp)
        return

if __name__ == '__main__':
    # name = "airport-03"
    # video = "/home/ivo/Videos/" + name + ".m4v"
    y = MyYolo()
    # rtsp = "rtsp://192.168.1.129:554/channel1"
    input = "data/kite.jpg"
    y.run(input)

