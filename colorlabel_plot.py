# program kmeans.py

import sys, math, os
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt


if __name__ == '__main__':

    # read commandline variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input file name')
    parser.add_argument('--label', dest = 'label_file', required = True, help = 'file containing color label of input data')
    parser.add_argument('--o', dest = 'out_name', required = True, help = 'root name for output files')
    parser.add_argument('--color', dest = 'color_map', default = 'gist_rainbow', help = 'cmap parameter of pyplot')
    parser.add_argument('--1st', dest = 'axis_1st', default = '1', help = 'dimension index used as 1st axis')
    parser.add_argument('--2nd', dest = 'axis_2nd', default = '2', help = 'dimension index used as 2nd axis')
    parser.add_argument('--xmin', dest = 'x_min', default = '-1', help = 'lower limit of x axis in pyplot')
    parser.add_argument('--xmax', dest = 'x_max', default = '-1', help = 'upper limit of x axis in pyplot')
    parser.add_argument('--xdelta', dest = 'x_delta', default = '-1', help = 'ticks scale of x axis in pyplot')
    parser.add_argument('--ymin', dest = 'y_min', default = '-1', help = 'lower limit of y axis in pyplot')
    parser.add_argument('--ymax', dest = 'y_max', default = '-1', help = 'upper limit of y axis in pyplot')
    parser.add_argument('--ydelta', dest = 'y_delta', default = '-1', help = 'ticks scale in y axis in pyplot')


    args = parser.parse_args()
    
    fn_in       = args.in_file
    fn_label    = args.label_file
    fn_out      = args.out_name

    dim_axis    = [int(args.axis_1st) - 1, int(args.axis_2nd) - 1]
    color       = args.color_map
    xmin        = float(args.x_min)
    xmax        = float(args.x_max)
    xdelta      = float(args.x_delta)
    ymin        = float(args.y_min)
    ymax        = float(args.y_max)
    ydelta      = float(args.y_delta)

    temparray = np.loadtxt(fn_in)
    data = np.delete(temparray, [0], axis = 1)

    temparray = np.loadtxt(fn_label)
    templabel = np.delete(temparray, [0], axis = 1)
    label = np.squeeze(templabel)

    #print(data.shape)
    #print(label.shape)


    temp_min = np.min(data, axis = 0)
    temp_max = np.max(data, axis = 0)

#    print(dim_axis)
#    print(temp_min)
#    print(temp_max)

    if (int(xmin) == -1):
        xmin = temp_min[dim_axis[0]]
    if (int(xmax) == -1):
        xmax = temp_max[dim_axis[0]]
    if (int(ymin) == -1):
        ymin = temp_min[dim_axis[1]]
    if (int(ymax) == -1):
        ymax = temp_max[dim_axis[1]]

    fn_png = fn_out + '_label_plot.png'

    plt.rcParams['figure.dpi'] = 600

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('Dim '+str(dim_axis[0]))
    plt.ylabel('Dim '+str(dim_axis[1]))
    plt.tight_layout()
    mappable = plt.scatter(data[:,dim_axis[0]], data[:,dim_axis[1]], s = 0.2, c = label, cmap = color)
    cbar = fig.colorbar(mappable, ax=ax)

    plt.savefig(fn_png)
    
    sys.exit()