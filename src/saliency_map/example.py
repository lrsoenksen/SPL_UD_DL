#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from saliency_map import SaliencyMap
from utils import OpencvIo

if __name__ == "__main__":
    oi = OpencvIo()
    src = oi.imread(sys.argv[1])
    sm = SaliencyMap(src)
    oi.imshow_array([sm.map])
