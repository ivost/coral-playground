
./label_image \
	--image ../../testdata/grace_hopper.bmp \
	--tflite_model ../../models/mobilenet/mobilenet_v1_1.0_224.tflite \
	--labels ../../models/mobilenet/mobilenet_v1_1.0_224.labels


./label_image \
	--image ../../testdata/grace_hopper.bmp \
	--tflite_model ../../models/mobilenet/mobilenet_v1_1.0_224_quant.tflite \
	--labels ../../models/mobilenet/mobilenet_v1_1.0_224_quant.labels

