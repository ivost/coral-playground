
import argparse


def parse_args(name):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", help="Required. Path to a image or folder with images.",
                        # env_var="INPUT",
                        default="/test_data/images",
                        type=str)
    parser.add_argument("-m", "--model", help="Required. Path to an tflite model",
                        # env_var="MODEL",
                        default="/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
                        type=str)
    parser.add_argument('-l', '--labels',
                        default='/test_data/inat_bird_labels.txt',
                        # env_var="LABELS",
                        type=str)
    parser.add_argument("-s", "--start",
                        help="Optional. Start index (when directory)",
                        default=0, type=int)
    parser.add_argument("-n", "--number",
                        help="Optional. Max number of images to process",
                        default=10, type=int)
    parser.add_argument("-c", "--confidence",
                        help="Optional. Min confidence",
                        default=0.6, type=float)
    parser.add_argument("-q", "--quiet",
                        help="Optional. If specified will show only perf",
                        action='store_true',
                        default=False)
    parser.add_argument("-tn", "--top", help="Optional. Number of top results", default=3, type=int)

    args = parser.parse_args()
    if args.number > 100:
        args.number = 100
    return args