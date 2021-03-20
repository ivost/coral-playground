r"""Example using PyCoral to classify a given image using an Edge TPU.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
bash examples/install_requirements.sh

-n 2 --verbose 1 --input /home/ivo/Pictures/birds --model  ./models/mobilenet/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite

cd ~/coral

python3 pycoral/examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/images/parrot.jpg

```
"""

import logging as log
import os
import sys

import time

from pathlib import Path
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from common import util
from common.args import parse_args
from insg.common import Stats

from insg.engine import Engine

CLASS = "birds"
IGNORE_IDS = [964]
version = "v.2021.3.19"


class Classify(Engine):
    # todo cmd line arg to choose ini file
    def __init__(self, log_level=log.INFO):
        super().__init__("Classification benchmark", version, "classify.ini")

    def main(self):
        stats = Stats()
        repeat = 1 # int(self.c.input.repeat)
        stats.begin()
        img_proc = self.img_proc
        img_proc.preprocess_images(self.size)
        log.info(f"{len(self.img_proc.files)} images")
        log.info(f"repeating {repeat} time(s)")
        for _ in range(repeat):
            print(".", end="", flush=True)
            # assuming batch size = 1
            for idx in range(len(self.img_proc.files)):
                images, images_hw = self.img_proc.preprocess_batch(idx, self.batch_size, self.channels, self.height, self.width)
                ###############################
                # inference
                stats.mark()

                # res = self.network.infer(inputs={self.input_blob: images})
                # failed = not self.process_classification_results(res, idx)

                failed = 0
                stats.bump(failed)
                ###############################
        stats.end()
        print("", flush=True)
        log.info(stats.summary())


if __name__ == '__main__':
    c = Classify()
    c.main()



# def main():
#     t0 = time.perf_counter()
#     args = init()
#     log.info(f"Classification benchmark {version}")
#     # must use raw string and valid regex "cat*.jpg" -> "cat.*\.jpg"
#     # args.re_path = R'cat.*\.jpg'
#     # check how many images are available
#     # log.info(f"{args.input}: {count} images matching {args.re_path}")
#     # args.input = INPUT_DIR
#     assert os.path.exists(args.input)
#     out_dir = None
#     failed_dir = None
#     if args.output:
#         out_dir = Path(args.output, "results").absolute()
#         if not out_dir.exists():
#             os.makedirs(out_dir)
#         failed_dir = Path(out_dir, "not-classified")
#         if not failed_dir.exists():
#             os.makedirs(failed_dir)
#
#     # by default max files is 10
#     count = util.count_images(args)
#     if count < args.count:
#         args.count = count
#
#     log.info(f"Loading {args.count} image(s) from {args.input} with total {count} image(s)")
#
#     util.load_images(args)
#     if len(args.files) == 0:
#         log.info(f"empty input set")
#         exit(0)
#
#     log.info(f"Loaded {len(args.files)} image(s)")
#
#     labels = read_label_file(args.labels) if args.labels else {}
#
#     log.info(f"Initializing")
#     interpreter = make_interpreter(args.model)
#     interpreter.allocate_tensors()
#     args.size = common.input_size(interpreter)
#     log.info(f"Image preparation")
#     images = util.preproces_images(args)
#     # Note: The first inference on Edge TPU is slow because it includes
#     # loading the model into Edge TPU memory.
#     image = images[0]
#     common.set_input(interpreter, image)
#     interpreter.invoke()
#     classes = classify.get_classes(interpreter, args.top, args.confidence)
#
#     repeat = 1
#     # accumulates inference time
#     inference_duration = 0
#     total = 0
#     idx = 0
#     failed = 0
#     log.info(f"START - repeating {repeat} time(s)")
#     for _ in range(repeat):
#         for image in images:
#             path = Path(args.files[idx]).absolute()
#             total += 1
#             idx += 1
#             t1 = time.perf_counter()
#             # inference
#             common.set_input(interpreter, image)
#             interpreter.invoke()
#             classes = classify.get_classes(interpreter, args.top, args.confidence)
#             inference_duration += time.perf_counter() - t1
#             # inference results
#             count = 0
#             cid = 0
#             for c in classes:
#                 if c.id not in IGNORE_IDS:
#                     if cid == 0:
#                         cid = c.id
#                     if args.verbose > 0:
#                         log.debug(f"{c.id:4d} - {labels.get(c.id, c.id)}, {c.score:5.2f}")
#             if cid > 0:
#                 count += 1
#                 if out_dir:
#                     util.copy_to_dir(args, src_file_path=path, dest_dir_path=Path(out_dir, str(cid)))
#                 continue
#
#             failed += 1
#             if failed_dir:
#                 util.copy_to_dir(args, src_file_path=path, dest_dir_path=failed_dir)
#
#     dur = time.perf_counter() - t0
#     avg = (inference_duration * 1000) / total
#     log.info(f"  Total images: {total}, not classified: {failed}")
#     log.info(f"Inference time: {inference_duration*1000:.0f} ms")
#     log.info(f"       Average: {avg:.2f} ms")
#     log.info(f"  Elapsed time: {dur*1000:.0f} ms")
#     if out_dir:
#         log.info(f"Results are in {out_dir}")
#     log.info(f"  END")
#     return
#
#
# def init():
#     log.basicConfig(format="[ %(levelname)s ] %(message)s",
#                     level=log.DEBUG, stream=sys.stdout)
#     args = parse_args("classification")
#     return args
#
#
# if __name__ == '__main__':
#     main()
