python3 ./py/detect.py \
  --model  /test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite  \
  --labels /test_data/coco_labels.txt \
  --input /test_data/images/coco_val \
  --confidence 0.3 \
  -n 200
