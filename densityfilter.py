#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from colour import Color
from matplotlib.patches import Polygon
import time
import random
import re
from random import randrange
import scipy.stats as st

from granatum_sdk import Granatum

def resetArray(minX, maxX, numsteps, numsigfigs):
    powten = 10**numsigfigs
    minX = minX*powten
    maxX = maxX*powten
    newMinX = np.floor(minX)
    newMaxX = np.ceil(maxX)
    stepsize = np.ceil((newMaxX-newMinX)/(numsteps-1.0))
    newMaxX = newMinX+stepsize*(numsteps-1.0)
    midNew = (newMaxX+newMinX)/2.0
    midOld = (maxX+minX)/2.0
    shift = np.round(midOld-midNew)
    return np.arange(newMinX+shift, newMaxX+stepsize+shift, step=stepsize)/powten

def main():
    tic = time.perf_counter()

    gn = Granatum()
    sample_coords = gn.get_import("coords")
    assay = gn.pandas_from_assay(gn.get_import("assay"))
    threshold = gn.get_arg("threshold")
    gridsize = gn.get_arg("gridsize")
    sigfigs = 2
    numticks = 6
    font = "Arial"

    coords = sample_coords.get("coords")
    coords = pd.DataFrame({"x": [a[0] for a in coords.values()], "y": [a[1] for a in coords.values()]}, index=coords.keys()) 
    dim_names = sample_coords.get("dimNames")
    random.seed(0)
    np.random.seed(0)

    target_dpi=300
    target_width=7.5 # inches
    target_height=6.5 # inches
    font_size_in_in=font/72.0 # inches
    font_size_in_px=font_size_in_in*target_dpi

    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1], s=5000/assay.shape[0])
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    xtickArray = resetArray(xmin, xmax, numticks, sigfigs)
    ytickArray = resetArray(ymin, ymax, numticks, sigfigs)
    plt.xlim(xtickArray[0], xtickArray[-1])
    plt.ylim(ytickArray[0], ytickArray[-1])
    plt.xticks(xtickArray, fontsize=font, fontname="Arial")
    plt.yticks(ytickArray, fontsize=font, fontname="Arial")
    plt.xlabel(dim_names[0], fontsize=font, fontname="Arial")
    plt.ylabel(dim_names[1], fontsize=font, fontname="Arial")

    ax = plt.gca()
    xs = np.arange(xmin, xmax, (xmax - xmin + 1.0)/gridsize)
    ys = np.arange(ymin, ymax, (ymax - ymin + 1.0)/gridsize)
    X, Y = np.meshgrid(xs, ys)
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = st.gaussian_kde(coords.T)
    Z = np.reshape(kernel(positions).T, X.shape)   # First get densities for plot
    thresh = Z.max() * threshold/100.0             # Absolute density threshold
    plt.contour(X, Y, Z, levels=[thresh], linewidths=0.1
    gn.add_current_figure_to_results(
        "Scatter-plot",
        dpi=target_dpi,
        width=target_width*target_dpi,
        height=target_height*target_dpi,
        savefig_kwargs={'bbox_inches': 'tight'})

    # Begin filtering
    assay = assay.loc[:, kernel(coords) > thresh]
    gn.export_statically(gn.assay_from_pandas(assay), 'Density filtered assay')

    gn.commit()



if __name__ == "__main__":
    main()
