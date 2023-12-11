from evaluation import Evaluation
import argparse

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--log_dir', type=str, default=None)
args = parser.parse_args()

evaluator = Evaluation(args.dir)
evaluator.evaluate()