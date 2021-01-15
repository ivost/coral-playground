
python3 ./py/classify.py \
  --model /test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels /test_data/inat_bird_labels.txt \
  --input /test_data/images/parrot.jpg