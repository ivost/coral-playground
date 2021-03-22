from yolov4.tflite import YOLOv4

yolo = YOLOv4()

yolo.config.parse_names("./dataset/coco.names")
yolo.config.parse_cfg("../config/yolov4-tiny-relu-tpu.cfg")
yolo.summary()
yolo.load_tflite("./yolov4-tiny-int8_edgetpu.tflite")

yolo.inference("kite.jpg")
