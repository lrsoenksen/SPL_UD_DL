#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    image processing utilities.
"""

import sys
import os
import math
import itertools
import cv2 as cv
import numpy as np


class OpencvIo:
    def __init__(self):
        self.__util = Util()

    def imread(self, path, option=1):
        try:
            if not os.path.isfile(os.path.join(os.getcwd(), path)):
                raise IOError('File is not exist')
            src = cv.imread(path, option)
        except IOError:
            raise
        except:
            print('Arugment Error : Something wrong')
            sys.exit()
        return src

    def imshow(self, src, name='a image'):
        cv.imshow(name, src)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def imshow_array(self, images):
        name = 0
        for x in images:
            cv.imshow(str(name), np.uint8(self.__util.normalize_range(x)))
            name = name + 1
        cv.waitKey(0)
        cv.destroyAllWindows()

    def saliency_array2img(self, images):
        for x in images:
            saliency_img = np.uint8(self.__util.normalize_range(x))
        return saliency_img


class Util:
    def normalize_range(self, src, begin=0, end=255):
        dst = np.zeros((len(src), len(src[0])))
        amin, amax = np.amin(src), np.amax(src)
        for y, x in itertools.product(range(len(src)), range(len(src[0]))):
            if amin != amax:
                dst[y][x] = (src[y][x] - amin) * (end - begin) / (amax - amin) + begin
            else:
                dst[y][x] = (end + begin) / 2
        return dst

    def normalize(self, src):
        src = self.normalize_range(src, 0., 1.)
        amax = np.amax(src)
        maxs = []

        for y in range(1, len(src) - 1):
            for x in range(1, len(src[0]) - 1):
                val = src[y][x]
                if val == amax:
                    continue
                if val > src[y - 1][x] and val > src[y + 1][x] and val > src[y][x - 1] and val > src[y][x + 1]:
                    maxs.append(val)

        if len(maxs) != 0:
            src *= math.pow(amax - (np.sum(maxs) / np.float64(len(maxs))), 2.)

        return src
