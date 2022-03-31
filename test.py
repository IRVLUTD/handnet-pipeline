import argparse
import os

from dex_ycb_toolkit.coco_eval import COCOEvaluator


def main():

  coco_eval = COCOEvaluator('s0_test')
  os.makedirs('out/', exist_ok=True)
  coco_eval.evaluate('results.json', out_dir='out/', tasks=['bbox'])


if __name__ == '__main__':
  main()