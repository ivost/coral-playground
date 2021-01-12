import logging as log
import os
import time
from os import listdir
from os.path import isfile, join


def count_images(args):
    path = os.path.abspath(args.input)
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return 1
    if not os.path.isdir(path):
        return 0
    count = 0
    max = args.start + args.number
    for f in listdir(path):
        if count >= max:
            break
        if isfile(join(path, f)):
            count += 1
    log.debug(f"image count {count}")
    return count


def load_images(args):
    path = os.path.abspath(args.input)
    files = []
    count = 0
    if os.path.isdir(path):
        log.debug(f"loading images from {path}")
        for f in listdir(path):
            if count == args.count:
                break
            if isfile(join(path, f)):
                count += 1
                files.append(join(path, f))
    else:
        log.debug(f"loading image {path}")
        files.append(path)
        count = 1
    assert args.count == count
    args.files = files

