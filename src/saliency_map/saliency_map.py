#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Saliency Map.
"""
import math
import itertools
import cv2 as cv
import numpy as np
from .utils import Util


class GaussianPyramid:
    def __init__(self, src):
        self.maps = self.__make_gaussian_pyramid(src)

    def __make_gaussian_pyramid(self, src):
        # gaussian pyramid | 0 ~ 8(1/256) . not use 0 and 1.
        maps = {'intensity': [],
                'colors': {'b': [], 'g': [], 'r': [], 'y': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}
        amax = np.amax(src)
        b, g, r = cv.split(src)
        for x in range(1, 9):
            b, g, r = list(map(cv.pyrDown, [b, g, r]))
            if x < 2:
                continue
            buf_its = np.zeros(b.shape)
            buf_colors = [np.zeros(b.shape) for _ in range(4)]  # b, g, r, y
            for y, x in itertools.product(range(len(b)), range(len(b[0]))):
                buf_its[y][x] = self.__get_intensity(b[y][x], g[y][x], r[y][x])
                buf_colors[0][y][x], buf_colors[1][y][x], buf_colors[2][y][x], buf_colors[3][y][x] = self.__get_colors(b[y][x], g[y][x], r[y][x], buf_its[y][x], amax)
            maps['intensity'].append(buf_its)
            for (color, index) in zip(sorted(maps['colors'].keys()), range(4)):
                maps['colors'][color].append(buf_colors[index])
            for (orientation, index) in zip(sorted(maps['orientations'].keys()), range(4)):
                maps['orientations'][orientation].append(self.__conv_gabor(buf_its, np.pi * index / 4))
        return maps

    def __get_intensity(self, b, g, r):
        return (np.float64(b) + np.float64(g) + np.float64(r)) / 3.

    def __get_colors(self, b, g, r, i, amax):
        b, g, r = [np.float64(x) if (x > 0.1 * amax) else 0. for x in [b, g, r]]
        nb, ng, nr = list(map(lambda x, y, z: max(x - (y + z) / 2., 0.), [b, g, r], [r, r, g], [g, b, b]))
        ny = max(((r + g) / 2. - math.fabs(r - g) / 2. - b), 0.)

        if i != 0.0:
            return [x / np.float64(i) for x in [nb, ng, nr, ny]]
        else:
            return nb, ng, nr, ny

    def __conv_gabor(self, src, theta):
        kernel = cv.getGaborKernel((8, 8), 4, theta, 8, 1)
        return cv.filter2D(src, cv.CV_32F, kernel)


class FeatureMap:
    def __init__(self, srcs):
        self.maps = self.__make_feature_map(srcs)

    def __make_feature_map(self, srcs):
        # scale index for center-surround calculation | (center, surround)
        # index of 0 ~ 6 is meaned 2 ~ 8 in thesis (Ich)
        cs_index = ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6))
        maps = {'intensity': [],
                'colors': {'bg': [], 'ry': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}

        for c, s in cs_index:
            maps['intensity'].append(self.__scale_diff(srcs['intensity'][c], srcs['intensity'][s]))
            for key in list(maps['orientations'].keys()):
                maps['orientations'][key].append(self.__scale_diff(srcs['orientations'][key][c], srcs['orientations'][key][s]))
            for key in list(maps['colors'].keys()):
                maps['colors'][key].append(self.__scale_color_diff(
                    (srcs['colors'][key[0]][c], srcs['colors'][key[0]][s]),
                    (srcs['colors'][key[1]][c], srcs['colors'][key[1]][s])
                ))
        return maps

    def __scale_diff(self, c, s):
        c_size = tuple(reversed(c.shape))
        return cv.absdiff(c, cv.resize(s, c_size, None, 0, 0, cv.INTER_NEAREST))

    def __scale_color_diff(self, xxx_todo_changeme, xxx_todo_changeme1):
        (c1, s1) = xxx_todo_changeme
        (c2, s2) = xxx_todo_changeme1
        c_size = tuple(reversed(c1.shape))
        return cv.absdiff(c1 - c2, cv.resize(s2 - s1, c_size, None, 0, 0, cv.INTER_NEAREST))


class ConspicuityMap:
    def __init__(self, srcs):
        self.maps = self.__make_conspicuity_map(srcs)

    def __make_conspicuity_map(self, srcs):
        util = Util()
        intensity = self.__scale_add(list(map(util.normalize, srcs['intensity'])))
        for key in list(srcs['colors'].keys()):
            srcs['colors'][key] = list(map(util.normalize, srcs['colors'][key]))
        color = self.__scale_add([srcs['colors']['bg'][x] + srcs['colors']['ry'][x] for x in range(len(srcs['colors']['bg']))])
        orientation = np.zeros(intensity.shape)
        for key in list(srcs['orientations'].keys()):
            orientation += self.__scale_add(list(map(util.normalize, srcs['orientations'][key])))
        return {'intensity': intensity, 'color': color, 'orientation': orientation}

    def __scale_add(self, srcs):
        buf = np.zeros(srcs[0].shape)
        for x in srcs:
            buf += cv.resize(x, tuple(reversed(buf.shape)))
        return buf


class SaliencyMap:
    def __init__(self, src):
        self.gp = GaussianPyramid(src)
        self.fm = FeatureMap(self.gp.maps)
        self.cm = ConspicuityMap(self.fm.maps)
        self.map = cv.resize(self.__make_saliency_map(self.cm.maps), tuple(reversed(src.shape[0:2])))

    def __make_saliency_map(self, srcs):
        util = Util()
        srcs = list(map(util.normalize, [srcs[key] for key in list(srcs.keys())]))
        return srcs[0] / 3. + srcs[1] / 3. + srcs[2] / 3.
