
import logging as log
import time
from pathlib import Path

from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.adapters import detect

from insg.common import Stats
from insg.engine import Engine

version = "v.2021.3.28"


class Detect(Engine):
    def __init__(self, log_level=log.INFO):
        super().__init__("Image Object Detection", "detect.ini", log_level)

    def run(self):
        log.info(f"Image Object detection {version}")
        stats = Stats()
        repeat = int(self.c.input.repeat)

        img_proc = self.img_proc
        if len(img_proc.files) == 0:
            log.info(f"empty input set")
            exit(0)

        log.info(f"repeating {repeat} time(s)")
        stats.begin()

        log.info(f"Initializing")
        log.info(f"Image preparation, size: {self.size}")
        images = img_proc.preprocess_images(self.size)
        log.info(f"{len(images)} image(s)")

        image = images[0]
        scale = (1.0, 1.0)
        conf = float(self.c.network.confidence)
        preview = "true" in str(self.c.output.preview).lower()
        log.debug(f"confidence {conf}")
        common.set_input(self.coral, image)
        self.coral.invoke()
        res = detect.get_objects(self.coral, conf, scale)
        log.debug(f"START - repeating {repeat} time(s)")
        for _ in range(repeat):
            failed = 0
            for idx in range(len(images)):
                file = self.img_proc.files[idx]
                image = images[idx]
                common.set_input(self.coral, image)
                self.coral.invoke()
                objects = detect.get_objects(self.coral, conf, scale)
                log.debug(f"==== in {file} - detected {len(objects)} object(s)")
                if not objects:
                    failed += 1
                    continue
                img = self.detection_results_original(objects, file)
                if preview:
                    img.show()
                p: str = self.generate_image_out_path(file)
                img.save(p)
                image = self.detection_results(objects, image)
                image.show()

    def generate_image_out_path(self, file):
        name = Path(file).stem
        type = Path(file).suffix
        model = Path(self.model).stem
        img = Path(self.c.output.dir, name + "_" + model + type)
        log.info(f"Output image: {str(img)}")
        return str(img)


if __name__ == '__main__':
    # d = Detect(log_level=log.DEBUG)
    d = Detect()
    d.run()
