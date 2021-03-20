# Lint as: python3
"""

-n 100 --verbose 0 --input /test_data/images/coco_val --model ./test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite --labels /test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite


"""

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import logging as log
import os
import sys

import time

from pathlib import Path

from common import util
from common.args import parse_args


version = "v.2021.3.19"


def main():
    t0 = time.perf_counter()
    args = init()
    log.info(f"Object detection benchmark {version}")

    assert os.path.exists(args.input)
    # out_dir = Path(args.input, "..", "results").absolute()
    # failed_dir = Path(out_dir, "not-recognized")
    # if not failed_dir.exists():
    #     os.makedirs(failed_dir)

    count = util.count_images(args)
    if count < args.count:
        args.count = count

    log.info(f"Loading {args.count} image(s) from {args.input} with total {count} image(s)")

    util.load_images(args)
    if len(args.files) == 0:
        log.info(f"empty input set")
        exit(0)

    log.info(f"Loaded {len(args.files)} image(s)")

    labels = read_label_file(args.labels) if args.labels else {}

    log.info(f"Initializing")
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    args.size = common.input_size(interpreter)
    log.info(f"Image preparation, size: {args.size}")
    images = util.preproces_images(args)

    # Note: The first inference on Edge TPU is slow because it includes
    # loading the model into Edge TPU memory.
    image = images[0]
    scale = (1.0, 1.0)
    log.debug(f"scale: {scale}, threshold: {args.threshold}")

    common.set_input(interpreter, image)
    interpreter.invoke()
    detect.get_objects(interpreter, args.threshold, scale)

    repeat = 1
    # accumulates inference time
    inference_duration = 0
    total = 0
    idx = 0
    failed = 0
    log.info(f"START - repeating {repeat} time(s)")
    for _ in range(repeat):
        for image in images:
            path = Path(args.files[idx]).absolute()
            total += 1
            idx += 1
            t1 = time.perf_counter()
            # inference
            common.set_input(interpreter, image)
            interpreter.invoke()
            objects = detect.get_objects(interpreter, args.threshold, scale)
            if not objects:
                failed += 1
                if args.verbose > 0:
                    print(f"No objects detected, image {path}")
                continue
            inference_duration += time.perf_counter() - t1
            # inference results
            if args.verbose > 1:
                log.debug(f"detected {len(objects)} objects")
                for obj in objects:
                    print(labels.get(obj.id, obj.id))
                    print('  id:    ', obj.id)
                    print('  score: ', obj.score)
                    print('  bbox:  ', obj.bbox)
            # if cid > 0:
            #     count += 1
            #     util.copy_to_dir(args, src_file_path=path, dest_dir_path=Path(out_dir, str(cid)))
            #     continue

            # util.copy_to_dir(args, src_file_path=path, dest_dir_path=failed_dir)

    dur = time.perf_counter() - t0
    avg = (inference_duration * 1000) / total
    log.info(f"  Total images: {total}, nothing detected on {failed}")
    log.info(f"Inference time: {inference_duration*1000:.0f} ms")
    log.info(f"       Average: {avg:.2f} ms")
    log.info(f"  Elapsed time: {dur*1000:.0f} ms")
    # log.info(f"  END - results are in {out_dir}")


def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)
    args = parse_args("classification")
    return args


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


if __name__ == '__main__':
    main()
