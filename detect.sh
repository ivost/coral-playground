python3 ./insg/detect.py \
  --model  ./models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite  \
  --labels ./models/mobilenet/coco_labels.txt \
  --input ./images/coco \
  --confidence 0.3 \
  -n 100
