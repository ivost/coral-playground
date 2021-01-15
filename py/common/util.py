import logging as log
import os
import re
import sys
from os import listdir
from os.path import isfile, join


def count_images(args):
    return count_or_load_images(args, True)


def load_images(args):
    return count_or_load_images(args, False)


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
            return 1
        if not os.path.isdir(path):
            return 0

    count = 0
    limit = args.start + args.number
    idx = args.start
    for f in listdir(path):
        if not count_only and idx >= limit:
            break
        if pat is None:
            fp = join(path, f)
            count += 1
            idx += 1
            if not count_only:
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
            args.files.append(fp)

    return count


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
    args.number = 5
    load_images(args)
    m = len(args.files)
    assert 0 <= m <= 5

    # must use raw string and valid regex "cat*.jpg" -> "cat.*\.jpg"
    args.re_path = R'cat.*\.jpg'
    args.start = 0
    args.number = 3
    m = count_images(args)
    load_images(args)
    m = len(args.files)
    assert 0 < m <= 3


if __name__ == '__main__':
    test()
