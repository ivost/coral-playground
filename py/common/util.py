import logging as log
import os
import re
import shutil
import sys
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path

from PIL import Image


def count_images(args):
    return count_or_load_images(args, True)


def load_images(args):
    return count_or_load_images(args, False)


# todo: change regex to pathlib.glob
def count_or_load_images(args, count_only):
    path = os.path.abspath(args.input)
    pat = None
    args.files = []
    if not hasattr(args, 're_path') or args.re_path is None:
        args.re_path = None
    else:
        rex = args.re_path
        try:
            pat = re.compile(rex)
        except Exception as err:
            log.error(f"Invalid regex {rex} {err}")
            return 0

    if args.re_path is None:
        if not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            if not count_only:
                if args.verbose > 0:
                    log.debug(f"adding image {path}")
                args.files.append(path)
            return 1
        if not os.path.isdir(path):
            return 0

    count = 0
    limit = args.start + args.count
    idx = args.start
    for f in listdir(path):
        if not count_only and idx >= limit:
            break
        if Path(f).is_dir():
            continue
        if pat is None:
            fp = join(path, f)
            if Path(fp).is_dir():
                continue
            count += 1
            idx += 1
            if not count_only:
                if args.verbose > 1:
                    log.debug(f"adding image {count}/{limit}  {fp}")
                args.files.append(fp)
            continue
        # regex
        m = pat.match(f)
        if m is None:
            continue
        fp = join(path, f)
        count += 1
        idx += 1
        if not count_only:
            if args.verbose > 1:
                log.debug(f"adding image {count}/{limit}  {fp}")
            args.files.append(fp)

    if args.verbose > 1:
        log.debug(f"{count} images")
    return count


def preproces_images(args):
    result = []
    start = time.perf_counter()
    for file in args.files:
        if Path(file).is_dir():
            continue
        if args.verbose > 1:
            log.debug(f"file {file}")
        result.append(Image.open(file).convert('RGB').resize(args.size, Image.ANTIALIAS))

    duration = (time.perf_counter() - start) / 1000
    if duration > 2:
        log.debug(f"preprocessing took {duration} ms")
    return result


def copy_to_dir(args, src_file_path, dest_dir_path):
    if not dest_dir_path.exists():
        os.makedirs(dest_dir_path)
    dest = Path(dest_dir_path, src_file_path.name)
    if dest.exists():
        return False
    if args.verbose > 1:
        log.debug(f"Copying {src_file_path.name} to {dest_dir_path}")
    shutil.copy2(src_file_path, dest_dir_path)


def test():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)

    # log.debug("testing...")

    from py.common.args import parse_args
    args = parse_args("test")
    args.input = "/test_data/images"

    n = count_images(args)
    assert 0 <= n <= 100

    args.start = n - 2
    args.count = 5
    load_images(args)
    m = len(args.files)
    assert 0 <= m <= 5

    # must use raw string and valid regex "cat*.jpg" -> "cat.*\.jpg"
    args.re_path = R'cat.*\.jpg'
    args.count = 0
    args.number = 3
    m = count_images(args)
    load_images(args)
    m = len(args.files)
    assert 0 < m <= 3


if __name__ == '__main__':
    test()
