import argparse
import time
import numpy as np

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model',
                      default="models/yolov4_416_full_integer_quant_edgetpu.tflite",
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input',
                      default="",
                      help='File path of image to process')
  parser.add_argument('-l', '--labels', help='File path of labels file',
                      default="models/coco_labels.txt")
  parser.add_argument('-t', '--threshold', type=float, default=0.2,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  #image = Image.open(args.input)
  #_, scale = common.set_resized_input(
  #    interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  sizes = interpreter.get_input_details()[0]["shape"]
  print(sizes)
  input_data = np.ones(sizes)
  scale, zero_point = input_details[0]['quantization']
  input_data = input_data / scale + zero_point
  input_data = input_data.astype(np.uint8)

  #common.set_input(interpreter, input_data)
  interpreter.set_tensor(input_index, input_data)


  print('----INFERENCE TIME----')
  print('Note: The first inference is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    print(inference_time)
    output_data = interpreter.get_tensor(output_index)
    print(output_data)

if __name__ == '__main__':
  main()
