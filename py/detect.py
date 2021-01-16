# Lint as: python3
"""

"""

import argparse
import time

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


version = "v.2021.1.16"


def main():
    t0 = time.perf_counter()
    args = init()
    log.info(f"Object detection benchmark {version}")

    assert os.path.exists(args.input)
    out_dir = Path(args.input, "..", "results").absolute()
    failed_dir = Path(out_dir, "not-recognized")
    if not failed_dir.exists():
        os.makedirs(failed_dir)

    # by default max files is 10
    count = util.count_images(args)
    if count < args.count:
        args.count = count

    log.info(f"Loading {args.count} images from {args.input} with total {count} images")

    util.load_images(args)
    if len(args.files) == 0:
        log.info(f"empty input set")
        exit(0)

    log.info(f"Loaded {len(args.files)} images")

    labels = read_label_file(args.labels) if args.labels else {}

    log.info(f"Initializing")
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    args.size = common.input_size(interpreter)
    log.info(f"Image preparation")
    images = util.preproces_images(args)
    # Note: The first inference on Edge TPU is slow because it includes
    # loading the model into Edge TPU memory.
    image = images[0]
    #common.set_input(interpreter, image)

    tensor, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    log.debug(f"scale: {scale}, threshold: {args.threshold}")

    interpreter.invoke()

    objects = detect.get_objects(interpreter, args.threshold, scale)
    log.debug(f"detected {objects} objects")

    # classes = classify.get_classes(interpreter, args.top, args.confidence)
    #
    # repeat = 1
    # # accumulates inference time
    # inference_duration = 0
    # total = 0
    # idx = 0
    # failed = 0
    # log.info(f"START - repeating {repeat} time(s)")
    # for _ in range(repeat):
    #     for image in images:
    #         path = Path(args.files[idx]).absolute()
    #         total += 1
    #         idx += 1
    #         t1 = time.perf_counter()
    #         # inference
    #         common.set_input(interpreter, image)
    #         interpreter.invoke()
    #         classes = classify.get_classes(interpreter, args.top, args.confidence)
    #         inference_duration += time.perf_counter() - t1
    #         # inference results
    #         count = 0
    #         cid = 0
    #         for c in classes:
    #             if c.id not in IGNORE_IDS:
    #                 if cid == 0:
    #                     cid = c.id
    #                 if args.verbose > 0:
    #                     log.debug(f"{c.id:4d} - {labels.get(c.id, c.id)}, {c.score:5.2f}")
    #         if cid > 0:
    #             count += 1
    #             util.copy_to_dir(args, src_file_path=path, dest_dir_path=Path(out_dir, str(cid)))
    #             continue
    #
    #         failed += 1
    #         util.copy_to_dir(args, src_file_path=path, dest_dir_path=failed_dir)
    #
    # dur = time.perf_counter() - t0
    # avg = (inference_duration * 1000) / total
    # log.info(f"  Total images: {total}, not classified: {failed}")
    # log.info(f"Inference time: {inference_duration*1000:.0f} ms")
    # log.info(f"       Average: {avg:.2f} ms")
    # log.info(f"  Elapsed time: {dur*1000:.0f} ms")
    # log.info(f"  END - results are in {out_dir}")


def init():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)
    args = parse_args("classification")
    return args

    # for _ in range(repeat):
    #     # tensor, scale = common.set_resized_input(
    #     #     interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    #     # start2 = time.perf_counter()
    #     interpreter.invoke()
    #     objs = detect.get_objects(interpreter, args.threshold, scale)
    #     # inference_time = time.perf_counter() - start2
    #     # print('%.2f ms' % (inference_time * 1000))
    #
    #     if not objs:
    #         print('No objects detected')
    #
    #     if verbose:
    #         print('-------RESULTS--------')
    #         for obj in objs:
    #             print(labels.get(obj.id, obj.id))
    #             print('  id:    ', obj.id)
    #             print('  score: ', obj.score)
    #             print('  bbox:  ', obj.bbox)
    #
    # inference_time = time.perf_counter() - start
    # print('Total time for %d inferences: %.2f ms' % (repeat, inference_time * 1000))
    # print('Average: %.2f ms' % ((inference_time * 1000)/repeat))
    #
    # if args.output:
    #     image = image.convert('RGB')
    #     draw_objects(ImageDraw.Draw(image), objs, labels)
    #     image.save(args.output)
    #     # image.show()

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
