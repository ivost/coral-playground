
python3 ./py/classify.py \
  --model  ./models/mobilenet/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels ./models/mobilenet/inat_bird_labels.txt \
  --input /home/ivo/Pictures/birds \
  -n 100
