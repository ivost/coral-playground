from yolov4.tf import YOLOv4, save_as_tflite, YOLODataset

"""
https://wiki.loliot.net/docs/lang/python/libraries/yolov4/python-yolov4-edge-tpu

edgetpu_compiler -sa yolov4-tiny-int8.tflite

"""
yolo = YOLOv4()

yolo.config.parse_names("coco.names")
yolo.config.parse_cfg("../config/yolov4-tiny-relu-tpu.cfg")

yolo.make_model()
yolo.load_weights("yolov4-tiny-relu.weights", weights_type="yolo")

dataset = YOLODataset(
    config=yolo.config,
    dataset_list="dataset/val2017.txt",
    image_path_prefix="dataset/val2017",
    training=False,
)

save_as_tflite(
    model=yolo.model,
    tflite_path="yolov4-tiny-int8.tflite",
    quantization="full_int8",
    dataset=dataset,
)
