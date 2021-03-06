# Lint as: python3
r"""

https://github.com/google-coral/pycoral/tree/master/examples

https://coral.ai/models/


Example using PyCoral to detect objects in a given image.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
bash examples/install_requirements.sh object_detection.py
python3 examples/object_detection.py \
  --model test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels test_data/coco_labels.txt \
  --input test_data/grace_hopper.bmp \
  --output ${HOME}/grace_hopper_processed.bmp
```
"""

import argparse
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

verbose = False
verbose = True


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def main():
    print('Simple image detection demo 1.0')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input',
                        default='/test_data/grace_hopper.bmp',
                        help='File path of image to process')

    parser.add_argument('-m', '--model',
                        help='File path of .tflite file',
                        default='/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')

    parser.add_argument('-l', '--labels', help='File path of labels file', default='/test_data/coco_labels.txt')

    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects')
    parser.add_argument('-o', '--output',
                        default='/tmp/grace_hopper_processed.bmp',
                        help='File path for the result image with annotations')
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    args = parser.parse_args()

    print('model', args.model)
    ##
    labels = read_label_file(args.labels) if args.labels else {}
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    image = Image.open(args.input)

    # size = common.input_size(interpreter)
    # image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)

    #print('image ', image.width, image.height)

    tensor, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))


    # print('scale', scale)
    # print('tensor ', tensor.width, tensor.height)

    # print('----INFERENCE TIME----')
    # print('Note: The first inference is slow because it includes',
    #       'loading the model into Edge TPU memory.')
    # common.set_input(interpreter, image)
    interpreter.invoke()
    objs = detect.get_objects(interpreter, args.threshold, scale)

    # repeat = args.count
    repeat = 1

    start = time.perf_counter()

    for _ in range(repeat):
        # tensor, scale = common.set_resized_input(
        #     interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        # start2 = time.perf_counter()
        interpreter.invoke()
        objs = detect.get_objects(interpreter, args.threshold, scale)
        # inference_time = time.perf_counter() - start2
        # print('%.2f ms' % (inference_time * 1000))

        if not objs:
            print('No objects detected')

        if verbose:
            print('-------RESULTS--------')
            for obj in objs:
                print(labels.get(obj.id, obj.id))
                print('  id:    ', obj.id)
                print('  score: ', obj.score)
                print('  bbox:  ', obj.bbox)

    inference_time = time.perf_counter() - start
    print('Total time for %d inferences: %.2f ms' % (repeat, inference_time * 1000))
    print('Average: %.2f ms' % ((inference_time * 1000)/repeat))

    if args.output:
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image.save(args.output)
        # image.show()


if __name__ == '__main__':
    main()
