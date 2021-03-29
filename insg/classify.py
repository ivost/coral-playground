
import logging as log

from pycoral.adapters import classify
from pycoral.adapters import common

from insg.common import Stats
from insg.engine import Engine

CLASS = "birds"
IGNORE_IDS = [964]
version = "v.2021.3.28"


class Classify(Engine):
    def __init__(self, log_level=log.INFO):
        super().__init__("Image Classification", "classify.ini", log_level)

    def run(self):
        stats = Stats()
        repeat = 1 # int(self.c.input.repeat)
        img_proc = self.img_proc
        images = img_proc.preprocess_images(self.size)
        log.info(f"{len(self.img_proc.files)} images")
        log.info(f"repeating {repeat} time(s)")
        stats.begin()
        exclude = [964]
        for _ in range(repeat):
            for idx in range(len(self.img_proc.files)):
                common.set_input(self.coral, images[idx])
                self.coral.invoke()
                stats.mark()
                classes = classify.get_classes(self.coral, 3, 0.1)
                failed = 1
                for c in classes:
                    if c.id not in exclude:
                        failed = 0
                        log.info(f'{c.score} - {c.id}: {self.labels.get(c.id, c.id)}')
                stats.bump(failed)
        stats.end()
        log.info(stats.summary())


if __name__ == '__main__':
    # c = Classify(log_level=log.DEBUG)
    c = Classify()
    c.run()
