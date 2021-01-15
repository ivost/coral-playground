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
import sys

import time

from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from common import util
from common.args import parse_args


def main():
    args = init()
    # must use raw string and valid regex "cat*.jpg" -> "cat.*\.jpg"
    # args.re_path = R'cat.*\.jpg'
    # check how many images are available
    count = util.count_images(args)
    # log.info(f"{args.input}: {count} images matching {args.re_path}")

    # by default limited to 10
    util.load_images(args)
    if len(args.files) == 0:
        log.info(f"nothing to do")
        exit(0)

    log.info(f"loaded {len(args.files)} images")

    labels = read_label_file(args.labels) if args.labels else {}

    interpreter = make_interpreter(args.model)

    interpreter.allocate_tensors()
    args.size = common.input_size(interpreter)
    images = util.preproces_images(args)
    # Note: The first inference on Edge TPU is slow because it includes
    # loading the model into Edge TPU memory.
    image = images[0]
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, args.top, args.confidence)

    # repeat = args.count
    repeat = 100

    inference_duration = 0
    total = 0
    for _ in range(repeat):
        for image in images:
            start = time.perf_counter()
            common.set_input(interpreter, image)
            interpreter.invoke()
            classes = classify.get_classes(interpreter, args.top, args.confidence)
            inference_duration += time.perf_counter() - start
            #print('-------RESULTS--------')
            # for c in classes:
            #     print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
            #interpreter.reset_all_variables()
            total += 1

    print('Total time for %d total inferences: %.2f ms' % (total, inference_duration * 1000))
    print('Average: %.2f ms' % ((inference_duration * 1000)/total))


def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)
    args = parse_args("classification")
    return args


if __name__ == '__main__':
    main()
