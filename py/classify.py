r"""Example using PyCoral to classify a given image using an Edge TPU.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
bash examples/install_requirements.sh

cd ~/coral

python3 pycoral/examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/images/parrot.jpg

```
"""

import logging as log
import os
import shutil
import sys

import time

from pathlib import Path
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from common import util
from common.args import parse_args

# todo: add to args

CLASS = "birds"
IGNORE_IDS = [964]
version = "v.2021.1.16"

def main():
    log.info(f"Classification benchmark {version}")
    t0 = time.perf_counter()
    args = init()
    # must use raw string and valid regex "cat*.jpg" -> "cat.*\.jpg"
    # args.re_path = R'cat.*\.jpg'
    # check how many images are available

    # log.info(f"{args.input}: {count} images matching {args.re_path}")
    # args.input = INPUT_DIR
    assert os.path.exists(args.input)
    failed_dir = Path(args.input, "not-classified")
    if not failed_dir.exists():
        os.makedirs(failed_dir)

    # by default max files is 10
    count = util.count_images(args)
    if count < args.count:
        args.count = count

    log.info(f"Loading {args.count} images from {args.input} with total {count} images")

    util.load_images(args)
    if len(args.files) == 0:
        log.info(f"empty input set")
        exit(0)

    log.info(f"Loaded {len(args.files)} images")

    labels = read_label_file(args.labels) if args.labels else {}

    log.info(f"Initializing")
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    args.size = common.input_size(interpreter)
    log.info(f"Image preparation")
    images = util.preproces_images(args)
    # Note: The first inference on Edge TPU is slow because it includes
    # loading the model into Edge TPU memory.
    image = images[0]
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, args.top, args.confidence)

    repeat = 1
    # accumulates inference time
    inference_duration = 0
    total = 0
    idx = 0
    failed = 0
    log.info(f"Start - Repeating {repeat} time(s)")
    for _ in range(repeat):
        for image in images:
            path = Path(args.files[idx]).absolute()
            total += 1
            idx += 1
            t1 = time.perf_counter()
            common.set_input(interpreter, image)
            interpreter.invoke()
            classes = classify.get_classes(interpreter, args.top, args.confidence)
            inference_duration += time.perf_counter() - t1

            count = 0
            for c in classes:
                if c.id not in IGNORE_IDS:
                    if args.verbose > 0:
                        log.debug(f"{c.id:4d} - {labels.get(c.id, c.id)}, {c.score:5.2f}")
                    count += 1

            if count == 0:
                failed += 1
                dest = Path(failed_dir, path.name)
                if dest.exists():
                    continue
                if args.verbose > 0:
                    log.debug(f"copying unrecognized image {path.name} to {dest}")
                shutil.copy2(path, failed_dir)
    dur = time.perf_counter() - t0
    avg = (inference_duration * 1000) / total
    log.info(f"Inference time: {inference_duration*1000:.0f} ms")
    log.info(f"  Total images: {total}, not classified: {failed}")
    log.info(f"       Average: {avg:.0f} ms")
    log.info(f"End - Elapsed time: {dur*1000:.0f} ms")


def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)
    args = parse_args("classification")
    return args


if __name__ == '__main__':
    main()
