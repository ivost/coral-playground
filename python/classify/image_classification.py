r"""Example using PyCoral to classify a given image using an Edge TPU.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
bash examples/install_requirements.sh

classify_image.py

python examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""


import logging as log
import sys

from python.common import util
from python.common.args import parse_args

import time

from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def main():
    args = init()
    # check how many images are available
    args.count = util.count_images(args)
    util.load_images(args)

    log.info(f"{len(args.files)} images")

    labels = read_label_file(args.labels) if args.labels else {}

    interpreter = make_interpreter(args.model)

    interpreter.allocate_tensors()
    size = common.input_size(interpreter)

    # image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
    # 
    # print('----INFERENCE TIME----')
    # # print('Note: The first inference on Edge TPU is slow because it includes',
    # #       'loading the model into Edge TPU memory.')
    # 
    # common.set_input(interpreter, image)
    # interpreter.invoke()
    # classes = classify.get_classes(interpreter, args.top_k, args.threshold)
    # # interpreter.reset_all_variables()
    # 
    # start = time.perf_counter()
    # # repeat = args.count
    # repeat = 10
    # 
    # for _ in range(repeat):
    #     common.set_input(interpreter, image)
    #     interpreter.invoke()
    #     classes = classify.get_classes(interpreter, args.top_k, args.threshold)
    #     interpreter.reset_all_variables()
    # 
    # inference_time = time.perf_counter() - start
    # print('Total time for %d inferences: %.2f ms' % (repeat, inference_time * 1000))
    # print('Average: %.2f ms' % ((inference_time * 1000)/repeat))
    # 
    # print('-------RESULTS--------')
    # for c in classes:
    #     print('%s: %.5f' % (labels.get(c.id, c.id), c.score))



def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)
    args = parse_args("classification")
    return args

if __name__ == '__main__':
    main()
