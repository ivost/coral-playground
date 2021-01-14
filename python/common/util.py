import logging as log
import os
import re
import sys
from os import listdir
from os.path import isfile, join

from python.common.args import parse_args


def count_images(args):
    return count_or_load_images(args, True)


def load_images(args):
    return count_or_load_images(args, False)


def count_or_load_images(args, countOnly):
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
            return 1
        if not os.path.isdir(path):
            return 0

    count = 0
    limit = args.start + args.number
    idx = args.start
    for f in listdir(path):
        if idx >= limit:
            break
        if pat is None:
            fp = join(path, f)
            count += 1
            idx += 1
            if not countOnly:
                args.files.append(fp)
            continue
        # regex
        m = pat.match(f)
        if m is None:
            continue
        fp = join(path, f)
        count += 1
        idx += 1
        if not countOnly:
            args.files.append(fp)

    return count


def test():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.DEBUG, stream=sys.stdout)

    # log.debug("testing...")

    args = parse_args("test")

    n = count_images(args)
    assert 0 <= n <= 10

    args.start = n - 2
    args.number = 5
    m = count_images(args)
    assert 0 <= m <= 5

    args.input = "/test_data/images"
    # must use raw string and valid regex "cat*.jpg" -> "cat.*\.jpg"
    args.re_path = R'cat.*\.jpg'
    args.start = 0
    args.number = 3
    m = count_images(args)
    assert 0 < m <= 3
    load_images(args)
    assert m == len(args.files)


if __name__ == '__main__':
    test()
