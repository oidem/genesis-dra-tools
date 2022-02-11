# program tica_interface.py

import sys, math, os
import matplotlib.pyplot as plt
import numpy as np
import pyemma
from pyemma.util.contexts import settings
import argparse
import time
import copy

if __name__ == '__main__':

    # read commandline variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input data after dimensionality reduction')
    parser.add_argument('--o', dest = 'out_name', default = 'feature_plot', help = 'root name of output file')
    parser.add_argument('--cfile', dest = 'color_file', required = True, help = 'file name containing feature values for coloring')
    parser.add_argument('--top', dest = 'top_file', default = '', help = 'input topology file such as ".pdb".')
    parser.add_argument('--trj', dest = 'type_trj', help = 'whether the datatype of your feature data is trajectory', action = 'store_true')
    parser.add_argument('--feature', dest = 'feature', default = 'all', help = 'feature you want to add to your trajectory input(default: all)')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--xmin', dest = 'x_min', required = True, help = 'minimum value of x axis', type = float)
    parser.add_argument('--xmax', dest = 'x_max', required = True, help = 'maximum value of x axis', type = float)
    parser.add_argument('--xtic', dest = 'x_del', required = True, help = 'step value of x axis', type = float)
    parser.add_argument('--ymin', dest = 'y_min', required = True, help = 'minimum value of y axis', type = float)
    parser.add_argument('--ymax', dest = 'y_max', required = True, help = 'maximum value of y axis', type = float)
    parser.add_argument('--ytic', dest = 'y_del', required = True, help = 'step value of y axis', type = float)
    parser.add_argument('--cmin', dest = 'c_min', required = True, help = 'minimum value of colormap', type = float)
    parser.add_argument('--cmax', dest = 'c_max', required = True, help = 'maximum value of colormap', type = float)
    parser.add_argument('--ctic', dest = 'c_del', required = True, help = 'step value of colormap', type = float)
    parser.add_argument('--axis', dest = 'axis_label', default = 'Dim', help = 'root label of axis. axis will be named "label"1 and "label"2. ')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--debug', dest = 'debug', help = 'debug option', action = 'store_true')

    # optional variables for transforming new data
    parser.add_argument('--fontsize', dest = 'font_size', default = '8', help = 'font size', type = float)
    parser.add_argument('--figsize', dest = 'fig_size', default = '5', help = 'size of x-dimension in inch (size of y-dimension will be determined by aspect ratio)', type = float)
    parser.add_argument('--dpi', dest = 'dpi', default = '300', help = 'dpi', type = int)
    parser.add_argument('--linewidth', dest = 'line_width', default = '0.3', help = 'line width of contour', type = float)
    parser.add_argument('--square', dest = 'do_square', help='ignore the aspect ratio of xrange/yange', action='store_true')
    parser.add_argument('--color', dest = 'color_map', default = 'jet', help = 'colormap of scatter plot')
    parser.add_argument('--tic_width', dest = 'tic_width', default = '2', help = 'width of ticks', type = float)
    parser.add_argument('--tic_length', dest = 'tic_length', default = '6', help = 'length of ticks', type = float)

    args = parser.parse_args()

    fn_in       = args.in_file
    fn_out      = args.out_name
    fn_color    = args.color_file
    top_file    = args.top_file
    trj         = args.type_trj
    feature     = args.feature
    is_multiple = args.is_multiple
    skip_output = args.skip_output
    xmin        = args.x_min
    xmax        = args.x_max
    xdel        = args.x_del
    ymin        = args.y_min
    ymax        = args.y_max
    ydel        = args.y_del
    cmin        = args.c_min
    cmax        = args.c_max
    cdel        = args.c_del
    alabel      = args.axis_label
    verbose     = args.verbose
    debug       = args.debug

    fontsize    = args.font_size
    fig_size    = args.fig_size
    dpi         = args.dpi
    linewidth   = args.line_width
    square      = args.do_square
    color       = args.color_map
    twidth      = args.tic_width
    tlength     = args.tic_length

    if (trj) and (top_file == ''):
        print('Error: you must input topology file by --top option, if the input datatype is "traj"!!!')
        sys.exit()

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    output fn_out: {}'.format(fn_out))
    print('    output fn_color: {}'.format(fn_color))
    print('    topology file: {}'.format(top_file))
    if (trj):
        print('    datatype of input: trj')
        print('    feature for coloring: {}'.format(feature))
    else:
        print('    datatype of input: other')
    if (skip_output):
        print('    skip output section?: yes')
    else:
        print('    skip output section?: no')
    print('    minimum value of x-axis: {}'.format(xmin))
    print('    maximum value of x-axis: {}'.format(xmax))
    print('    step value of x-axis: {}'.format(xdel))
    print('    minimum value of y-axis: {}'.format(ymin))
    print('    maximum value of y-axis: {}'.format(ymax))
    print('    step value of y-axis: {}'.format(ydel))
    print('    minimum value of colormap: {}'.format(cmin))
    print('    maximum value of colormap: {}'.format(cmax))
    print('    step value of pmf colormap: {}'.format(cdel))

    print('expert options')
    print('    font size: {}'.format(fontsize))
    print('    figure size (inch): {}'.format(fig_size))
    print('    dpi: {}'.format(dpi))
    print('    line width: {}'.format(linewidth))
    if (square):
        print('    ignore the aspect ratio of xrange/yrange?: yes')
    else:
        print('    ignore the aspect ratio of xrange/yrange?: no')
    print('    colormap: {}'.format(color))

    start = time.time()

    infiles = []
    if (is_multiple):
        with open(fn_in) as input:
            lines = input.readlines()
            for line in lines:
                infiles.append(line.replace('\n', ''))  # stripping line ending
    else:
        infiles.append(fn_in)

    nfile = len(infiles)

    if (verbose):
        print('')
        print('number of input files: {}'.format(nfile))

    data = []

    if (trj):
        feat = pyemma.coordinates.featurizer(top_file)
        if (feature == 'all'):
            feat.add_all()
        elif (feature == 'backbone_torsions'):
            feat.add_backbone_torsions()
        elif (feature == 'ca_torsions'):
            ca = feat.select_Ca()
            npair = ca.size - 3
            indices = np.zeros((npair, 4), dtype=np.int)
            for i in range(npair):
                for j in range(4):
                    indices[i, j] = ca[i+j]
            feat.add_dihedrals(indices, cossin = True, periodic = False)
        elif (feature == 'distances_ca'):
            feat.add_distances_ca()

        if (nfile != 1):
            data = pyemma.coordinates.load(infiles, feat)
        else:
            temparray = pyemma.coordinates.load(infiles, feat)
            data.append(temparray.copy())

        print('type of data: ', type(data))
        print('dimensions: ', len(data))
        print('type of data[0]', type(data[0]))
        print('shape of elements: ', data[0].shape)
        print('n_atoms: ', feat.topology.n_atoms)
    else:

        for infile in infiles:

            temparray = np.loadtxt(infile)
            data.append(np.delete(temparray, [0], axis=1))

    data_concatenated = np.concatenate(data)

    if (verbose):
        print('')
        print('shape of each input data:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)
        print('shape of concatenated data: ', data_concatenated.shape)

    colordata = []
    temparray = np.loadtxt(fn_color)
    colordata.append(np.delete(temparray, [0], axis=1))

    xsize = fig_size
    ysize = ((ymax - ymin) / (xmax - xmin)) * xsize
    if (square):
        ysize = xsize
    fig = plt.figure(figsize=(xsize, ysize))

    plt.switch_backend('agg')
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = (xsize, ysize)

    txmax = ((xmax // xdel) + 1) * xdel
    txmin = ((xmin // xdel) - 1) * xdel

    tymax = ((ymax // ydel) + 1) * ydel
    tymin = ((ymin // ydel) - 1) * ydel

    if (debug):
        print('')
        print('txmin: ', txmin)
        print('txmax: ', txmax)
        print('tymin: ', tymin)
        print('tymax: ', tymax)

    interval_x = np.arange(txmin, txmax, xdel)
    interval_y = np.arange(tymin, tymax, ydel)
    interval_c = np.arange(cmin, cmax, cdel)

    # set colormap
    cm = plt.cm.get_cmap(color)

    # plot scatter
    mappable = plt.scatter(*data_concatenated[:,:2].T, c=colordata, vmin=cmin, vmax=cmax, s=0.2, cmap=cm)

    # generate color bar
    if not square:
        cbar = fig.colorbar(mappable, ax=ax)

    if (not skip_output):

        fn_png = fn_out + '_feature_plot.png'

        # plot
        plt.xticks(interval_x)
        plt.yticks(interval_y)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.tick_params(width=twidth, length=tlength, colors='k')

        if (square):
            plt.axes().set_aspect('auto')
        else:
            plt.axes().set_aspect('equal')
        plt.xlabel(alabel+'1')
        plt.ylabel(alabel+'2')
        plt.tight_layout()
        plt.savefig(fn_png)

    interval = time.time() - start
    print('processing time : {}s'.format(interval))

    sys.exit()
