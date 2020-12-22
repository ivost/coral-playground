r"""Example using PyCoral to classify a given image using an Edge TPU.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
bash examples/install_requirements.sh classify_image.py
python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""

import argparse
import time

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite',
                        help='File path of .tflite file.')
    parser.add_argument('-i', '--input', default='/test_data/parrot.jpg',
                        help='Image to be classified.')
    parser.add_argument('-l', '--labels', default='/test_data/inat_bird_labels.txt',
                        help='File path of labels file.')
    parser.add_argument('-k', '--top_k', type=int, default=1,
                        help='Max number of classification results')
    parser.add_argument('-t', '--threshold', type=float, default=0.0,
                        help='Classification score threshold')
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    args = parser.parse_args()

    labels = read_label_file(args.labels) if args.labels else {}

    interpreter = make_interpreter(*args.model.split('@'))
    interpreter.allocate_tensors()

    size = common.input_size(interpreter)
    image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, image)

    print('----INFERENCE TIME----')
    print('Note: The first inference on Edge TPU is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        classes = classify.get_classes(interpreter, args.top_k, args.threshold)
        print('%.1fms' % (inference_time * 1000))

    print('-------RESULTS--------')
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))


if __name__ == '__main__':
    main()
