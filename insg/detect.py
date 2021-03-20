
import logging as log
import time

from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.adapters import detect

from insg.common import Stats
from insg.engine import Engine

version = "v.2021.3.20"


class Detect(Engine):
    def __init__(self, log_level=log.INFO):
        super().__init__("Image Object Detection", version, "detect.ini", log_level)

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
        log.info(f"{len(images)} images")

        image = images[0]
        scale = (1.0, 1.0)
        conf = float(self.c.network.confidence)
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
                # inference results
                for obj in objects:
                    label = self.labels.get(obj.id, obj.id)
                    log.debug(f"{obj.score:.2f} {obj.id} {label}")

                # image = image.convert('RGB')
                draw_objects(ImageDraw.Draw(image), objects, self.labels)
                # # image.save(args.output)
                # # todo: either move cv2 or use tk
                image.show()

                # if cid > 0:
                #     count += 1
                #     util.copy_to_dir(args, src_file_path=path, dest_dir_path=Path(out_dir, str(cid)))
                #     continue
                # util.copy_to_dir(args, src_file_path=path, dest_dir_path=failed_dir)


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        label = labels.get(obj.id, obj.id)
        bbox = obj.bbox
        pos = (bbox.xmin + 2, bbox.ymin + 2)
        # draw.font()
        # draw.textsize(2)
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='yellow')
        draw.text(pos, f"{obj.score:.2f}-{label}", fill='yellow')


if __name__ == '__main__':
    # d = Detect(log_level=log.DEBUG)
    d = Detect()
    d.run()
