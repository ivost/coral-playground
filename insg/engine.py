import logging as log
import sys

from PIL import Image, ImageDraw
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from common.config import Config
from common.imageproc import ImageProc

VERSION = "2021.3.20.1"


class Engine:

    def __init__(self, message, config_ini='config.ini', log_level=log.INFO):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log_level, stream=sys.stdout)
        log.info(f"\n{message} {VERSION}\n")

        self.c = Config()
        self.c.read(config_ini)

        n = self.c.network

        model = Config.existing_path(n.model)
        self.model = str(model)
        self.input = Config.existing_path(self.c.input.images)
        labels = Config.existing_path(n.labels)
        self.labels = read_label_file(labels)
        log.info(f"Loading model: {self.model}")

        self.img_proc = ImageProc(self.c)
        self.img_proc.prepare()

        # initialize coral
        log.info(f"Initializing Coral")
        self.coral = make_interpreter(self.model)
        self.coral.allocate_tensors()
        self.size = common.input_size(self.coral)
        return

    def prepare_input(self, images):
        pass

    def model_check(self):
        pass

    def detection_results_original(self, objects, original_file):
        log.info(f"process_detection_results {original_file} - {len(objects)} objects")
        img: Image = Image.open(original_file).convert('RGB')
        img.load()
        ax = img.size[0] / self.size[0]
        ay = img.size[1] / self.size[1]
        # log.debug(f"original size: {img.size}, proc.size: {self.size}")
        # log.debug(f"ax: {ax}, ay: {ay}")
        return self.draw_boxes(objects, img, ax, ay)

    def detection_results(self, objects, img):
        return self.draw_boxes(objects, img)

    def draw_boxes(self, objects, img, ax=1, ay=1):
        tone = self.img_proc.image_brightness(self.img_proc.files[0])
        color = "white" if tone == "dark" else "purple"

        canvas = ImageDraw.Draw(img)
        for obj in objects:
            label = self.labels.get(obj.id, obj.id)
            log.debug(f"{obj.score:.2f} {label}")
            if ax != 1 and ay != 1:
                bb = obj.bbox.scale(ax, ay).map(int)
            else:
                bb = obj.bbox

            canvas.rectangle([(bb.xmin, bb.ymin), (bb.xmax, bb.ymax)], outline=color)
            pos = (bb.xmin + 4, bb.ymin + 4)
            canvas.text(pos, f"{obj.score:.2f}-{label}", fill=color)
        return img


if __name__ == '__main__':
    engine = Engine("init")
