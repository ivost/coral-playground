"""
MIT License

Copyright (c) 2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import sys
print("common __init__")

sys.path.append("/home/ivo/github/coral-playground/yolov4-tflite/YOLOv4")
sys.path.append("/home/ivo/github/coral-playground/yolov4-tflite/YOLOv4/common")
sys.path.append("/home/ivo/github/coral-playground/yolov4-tflite/YOLOv4/common/metalayer")
sys.path.append("/home/ivo/github/coral-playground/yolov4-tflite/YOLOv4/tflite")

print(sys.path)

import media
import parser
import config


# from . import _common
#from config import YOLOConfig
# from _common import (
#     get_yolo_detections as _get_yolo_detections,
#     get_yolo_tiny_detections as _get_yolo_tiny_detections,
#     fit_to_original as _fit_to_original,
# )
