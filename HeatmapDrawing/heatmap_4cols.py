
from __future__ import division

import numpy as np
from matplotlib import cm
import matplotlib.mlab as ml

import pylab as pl
from pylab import imread

import pandas as pd


projectFolder = r'./data'
imFile = '%s/bg.png' % projectFolder
dataFile = '%s/data.csv' % projectFolder
outImage = './output.png'


def plot_heatmap(bgImFile, txt, outImage):
    mat = np.genfromtxt(txt, delimiter=',')
    im = imread(imFile)
    image_height, image_width = im.shape[:2]
    num_x = image_width / 5
    num_y = num_x / (image_width / image_height)
    x = np.linspace(0, image_width, num_x)
    y = np.linspace(0, image_height, num_y)

    figure = pl.figure(figsize=(10, 10), dpi=100)
    ax = pl.subplot(211)
    z = ml.griddata(mat[:, 0], mat[:, 1], mat[:, 2], x, y)
    cs = ax.contourf(x, y, z, alpha=0.6, zorder=1, cmap=cm.jet)
    pl.colorbar(cs)
    pl.plot(mat[:, 0], mat[:, 1], '+', alpha=0.6, markersize=1.5, zorder=3)
    ax.imshow(im, alpha=0.3, zorder=0)

    ax = pl.subplot(212)
    z = ml.griddata(mat[:, 0], mat[:, 1], mat[:, 3], x, y)
    cs = ax.contourf(x, y, z, alpha=0.6, zorder=1, cmap=cm.jet)
    pl.colorbar(cs)
    pl.plot(mat[:, 0], mat[:, 1], '+', alpha=0.6, markersize=1.5, zorder=3)
    ax.imshow(im, alpha=0.3, zorder=0)

    pl.savefig(outImage)
    pl.clf()
    pl.close(figure)

if __name__ == '__main__':
    plot_heatmap(imFile, dataFile, outImage)
