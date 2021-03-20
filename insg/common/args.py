
import argparse


def parse_args(name):
    root = "/home/ivo/github/coral-playground/"
    mr = root+"models/mobilenet/"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", help="Required. Path to a image or folder with images.",
                        # env_var="INPUT",
                        default=root+"images",
                        type=str)
    parser.add_argument("-m", "--model", help="Required. Path to model",
                        default=mr+"mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
                        type=str)
    parser.add_argument('-l', '--labels',
                        default=mr+'inat_bird_labels.txt',
                        type=str)
    parser.add_argument("-o", "--output", help="Optional. Path to output directory.",
                        default=None,
                        type=str)
    parser.add_argument("-s", "--start",
                        help="Optional. Start index (when directory)",
                        default=0, type=int)
    parser.add_argument("-n", "--count",
                        help="Optional. Max number of images to process",
                        default=10, type=int)
    parser.add_argument("-c", "--confidence",
                        help="Optional. Min confidence",
                        default=0.4, type=float)
    parser.add_argument("-T", "--threshold",
                        help="Optional. Threshold",
                        default=0.4, type=float)
    parser.add_argument("-q", "--quiet",
                        help="Optional. If specified will show only perf",
                        action='store_true',
                        default=False)
    parser.add_argument("-t", "--top", help="Optional. Number of top results", default=3, type=int)
    parser.add_argument("-v", "--verbose",
                        help="Optional verbosity level. Use for debugging",
                        type=int,
                        default=0)

    args = parser.parse_args()
    # if args.number > 100:
    #     args.number = 100
    return args