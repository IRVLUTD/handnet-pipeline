import argparse
import os

from dex_ycb_toolkit.coco_eval import COCOEvaluator
from datasets3d.a2jdataset import A2JDataset


def main():
  data = A2JDataset()
  test = data[310549]


if __name__ == '__main__':
  main()