# program cluster_feature_plot.py   ! under construction


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
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input data used for clustering')
    parser.add_argument('--o', dest = 'out_name', default = 'feature_plot', help = 'root name of output file')
    parser.add_argument('--center', dest = 'center_file', required = True, help = 'file name for cluster centers')
    parser.add_argument('--cfile', dest = 'color_file', required = True, help = 'file name containing feature values for coloring')
    parser.add_argument('--top', dest = 'top_file', default = '', help = 'input topology file such as ".pdb".')
    parser.add_argument('--trj', dest = 'type_trj', help = 'whether the datatype of your feature data is trajectory', action = 'store_true')
    parser.add_argument('--feature', dest = 'feature', default = 'all', help = 'feature you want to add to your trajectory input(default: all)')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--cutoff', dest='cutoff', default='1.0', help='Lower limit of #samples in bins', type = float)
    parser.add_argument('--stepsize', dest='step_size', default='1.0', help='step size of gridding', type = float)
    parser.add_argument('--do_spline', dest = 'do_spline', help = 'whether you want to interpolate your pmf grid with spline function', action = 'store_true')
    parser.add_argument('--zoom', dest = 'n_zoom', default = '2', help = 'zoom factor of interpolation', type = float)
    parser.add_argument('--nspline', dest = 'n_spline', default = '3', help = 'number of spline order', type = int)
    parser.add_argument('--T', dest='temperature', default='293', help='temperature of the input system', type = float)
    parser.add_argument('--xmin', dest = 'x_min', required = True, help = 'minimum value of x axis', type = float)
    parser.add_argument('--xmax', dest = 'x_max', required = True, help = 'maximum value of x axis', type = float)
    parser.add_argument('--xtic', dest = 'x_del', required = True, help = 'step value of x axis', type = float)
    parser.add_argument('--ymin', dest = 'y_min', required = True, help = 'minimum value of y axis', type = float)
    parser.add_argument('--ymax', dest = 'y_max', required = True, help = 'maximum value of y axis', type = float)
    parser.add_argument('--ytic', dest = 'y_del', required = True, help = 'step value of y axis', type = float)
    parser.add_argument('--cmin', dest = 'c_min', required = True, help = 'minimum value of colormap', type = float)
    parser.add_argument('--cmax', dest = 'c_max', required = True, help = 'maximum value of colormap', type = float)
    parser.add_argument('--pmfmin', dest = 'pmf_min', required = True, help = 'minimum value of pmf contour', type = float)
    parser.add_argument('--pmfmax', dest = 'pmf_max', required = True, help = 'maximum value of pmf contour', type = float)
    parser.add_argument('--pmfdel', dest = 'pmf_del', required = True, help = 'step value of pmf contour', type = float)
    parser.add_argument('--xcyclic', dest = 'xcyclic', help='x grid is cyclic or not', action='store_true')
    parser.add_argument('--ycyclic', dest = 'ycyclic', help='y grid is cyclic or not', action='store_true')
    parser.add_argument('--ctic', dest = 'c_del', required = True, help = 'step value of colormap', type = float)
    parser.add_argument('--axis', dest = 'axis_label', default = 'Dim', help = 'root label of axis. axis will be named "label"1 and "label"2. ')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')

    # optional variables for transforming new data
    parser.add_argument('--fontsize', dest = 'font_size', default = '8', help = 'font size', type = float)
    parser.add_argument('--figsize', dest = 'fig_size', default = '5', help = 'size of x-dimension in inch (size of y-dimension will be determined by aspect ratio)', type = float)
    parser.add_argument('--dpi', dest = 'dpi', default = '300', help = 'dpi', type = int)
    parser.add_argument('--linewidth', dest = 'line_width', default = '0.3', help = 'line width of contour', type = float)
    parser.add_argument('--square', dest = 'do_square', help='ignore the aspect ratio of xrange/yange', action='store_true')
    parser.add_argument('--color', dest = 'color_map', default = 'jet', help = 'colormap of pmf plot')
    parser.add_argument('--color_center', dest = 'color_center', default = 'jet', help = 'colormap of center plot')

    args = parser.parse_args()

    fn_in       = args.in_file
    fn_out      = args.out_name
    fn_center   = args.center_file
    fn_color    = args.color_file
    top_file    = args.top_file
    trj         = args.type_trj
    feature     = args.feature
    is_multiple = args.is_multiple
    skip_output = args.skip_output
    cutoff      = args.cutoff
    stepsize    = args.step_size
    spline      = args.do_spline
    nzoom       = args.n_zoom
    nspline     = args.n_spline
    temperature = args.temperature
    xmin        = args.x_min
    xmax        = args.x_max
    xdel        = args.x_del
    ymin        = args.y_min
    ymax        = args.y_max
    ydel        = args.y_del
    cmin        = args.c_min
    cmax        = args.c_max
    cdel        = args.c_del
    pmfmin      = args.pmf_min
    pmfmax      = args.pmf_max
    pmfdel      = args.pmf_del
    xcyclic     = args.xcyclic
    ycyclic     = args.ycyclic
    alabel      = args.axis_label
    verbose     = args.verbose

    fontsize    = args.font_size
    fig_size    = args.fig_size
    dpi         = args.dpi
    linewidth   = args.line_width
    square      = args.do_square
    color       = args.color_map

    if (trj) and (top_file == ''):
        print('Error: you must input topology file by --top option, if the input datatype is "traj"!!!')
        sys.exit()

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    output fn_out: {}'.format(fn_out))
    print('    output fn_center: {}'.format(fn_center))
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
    print('    cutoff value: {}'.format(cutoff))
    print('    step size of gridding: {}'.format(stepsize))
    print('    temperature: {}'.format(temperature))
    print('    minimum value of x-axis: {}'.format(xmin))
    print('    maximum value of x-axis: {}'.format(xmax))
    print('    step value of x-axis: {}'.format(xdel))
    print('    minimum value of y-axis: {}'.format(ymin))
    print('    maximum value of y-axis: {}'.format(ymax))
    print('    step value of y-axis: {}'.format(ydel))
    print('    minimum value of colormap: {}'.format(cmin))
    print('    maximum value of colormap: {}'.format(cmax))
    print('    step value of  colormap: {}'.format(cdel))
    print('    minimum value of pmf contour: {}'.format(pmfmin))
    print('    maximum value of pmf contour: {}'.format(pmfmax))
    print('    step value of pmf contour: {}'.format(pmfdel))
    if (xcyclic):
        print('    is the x-axis is cyclic coordinate?: yes')
    else:
        print('    is the x-axis is cyclic coordinate?: no')
    if (ycyclic):
        print('    is the y-axis is cyclic coordinate?: yes')
    else:
        print('    is the y-axis is cyclic coordinate?: no')
    print('    label of each axis: {}'.format(alabel))

    print('expert options')
    if (spline):
        print('    interpolate your input pmf grid?: yes')
    else:
        print('    interpolate your input pmf grid?: no')
    print('    zoom factor: {}'.format(nzoom))
    print('    order of spline interpolation: {}'.format(nspline))
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

    # parameter setting
    nbin_max = 1000000
    fe_max = 10000000.0
    #numx = int((xmax-xmin) / xdel)
    #numy = int((ymax-ymin) / ydel)
    RT = 8.314 * temperature / 1000.0
    RT_kcal = 8.314 * temperature / 4.184 / 1000.0
    beta = (1000.0 * 4.184) / (8.314 * temperature)

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

    ndata = data_concatenated.shape[0]
    xdata_min, ydata_min = data_concatenated.min(axis=0)
    xdata_max, ydata_max = data_concatenated.max(axis=0)

    if (verbose):
        print('')
        print('shape of elements: {}'.format(data.shape))
        print('minimum data value along x-axis: {}'.format(xdata_min))
        print('maximum data value along x-axis: {}'.format(xdata_max))
        print('minimum data value along y-axis: {}'.format(ydata_min))
        print('maximum data value along y-axis: {}'.format(ydata_max))

    numx = int((xmax - xmin) / stepsize)
    numy = int((ymax - ymin) / stepsize)

    if (verbose):
        print('')
        print('number of grids along x-axis (before interpolation): {}'.format(numx))
        print('number of grids along y-axis (before interpolation): {}'.format(numy))

    ## Calculate number of samples in each bin.
    ndata_in_bin = np.zeros([numx, numy], int)
    for i in range(ndata):
        cv1 = float(data[i, 0])
        cv2 = float(data[i, 1])
        idx1 = int((cv1 - xmin) / stepsize)
        idx2 = int((cv2 - ymin) / stepsize)
        if (xcyclic):
            if (idx1 >= numx):
                idx1 -= numx
        if (ycyclic):
            if (idx2 >= numy):
                idx2 -= numy
        ndata_in_bin[idx1,idx2] += 1

    zoomed = ndata_in_bin
    numx_zoom = numx
    numy_zoom = numy

    if (spline):
        zoomed = scipy.ndimage.zoom(ndata_in_bin, zoom = nzoom, order = nspline)
        numx_zoom = int(nzoom * numx)
        numy_zoom = int(nzoom * numy)

    if (verbose):
        print('')
        print('number of grids along x-axis (after interpolation): {}'.format(numx_zoom))
        print('number of grids along y-axis (after interpolation): {}'.format(numy_zoom))

    ## Calculate the free energy in each bin
    fe = np.zeros([numx_zoom, numy_zoom], float)
    fe_min = fe_max
    for j in range(numy_zoom):
        for i in range(numx_zoom):
            if (zoomed[i, j] <= 0):
                fe[i,j] = fe_max
            else:
                if zoomed[i, j] > cutoff:
                    fe[i, j] = - RT_kcal * np.log(zoomed[i, j])
            if (fe[i, j] < fe_min):
                fe_min = fe[i,j]
    fe -= fe_min

    if (verbose):
        print('')
        print('shape of grids before interpolation: {}'.format(ndata_in_bin.shape))
        print('shape of grids after interpolation: {}'.format(zoomed.shape))
        print('shape of interpolated PMF grids: {}'.format(fe.shape))

    if (spline):
        zoom_factor = 1 / nzoom
    else:
        zoom_factor = 1

    xmin_zoom = float(xmin) + 0.5 * stepsize * zoom_factor
    x_list = []
    for i in range(numx_zoom):
        x_list.append(xmin_zoom + i * stepsize * zoom_factor)
    x = np.array(x_list)

    ymin_zoom = float(ymin) + 0.5 * stepsize * zoom_factor
    y_list = []
    for i in range(numy_zoom):
        y_list.append(ymin_zoom + i * stepsize * zoom_factor)
    y = np.array(y_list)

    x_mesh, y_mesh = meshgrid(x, y)

    if (verbose):
        print('')
        print('shape of array x: {}'.format(x.shape))
        print('shape of array y: {}'.format(y.shape))
        print('shape of meshgrid x: {}'.format(x_mesh.shape))
        print('shape of meshgrid y: {}'.format(y_mesh.shape))

    centerdata = []
    temparray = np.loadtxt(fn_center)
    centerdata.append(np.delete(temparray, [0], axis=1))

    colordata = []
    temparray = np.loadtxt(fn_color)
    colordata.append(np.delete(temparray, [0], axis=1))

    plt.switch_backend('agg')
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['figure.dpi'] = dpi

    xsize = fig_size
    ysize = ((ymax - ymin) / (xmax - xmin)) * xsize
    if (square):
        ysize = xsize
    fig = plt.figure(figsize=(xsize, ysize))
    ax = fig.add_subplot(1,1,1)

    interval_x = np.arange(xmin, xmax, xdel)
    interval_y = np.arange(ymin, ymax, ydel)
    interval_c = np.arange(cmin, cmax, cdel)

    # set colormap
    cm = plt.cm.get_cmap(color)

    # plot scatter
    mappable = ax.scatter(*data_concatenated[:,:2].T, c=colordata, vmin=cmin, vmax=cmax, s=0.2, cmap=cm)

    # generate color bar
    cbar = fig.colorbar(mappable, ax=ax)

    if (not skip_output):

        fn_png = fn_out + '_feature_plot.png'

        # plot
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.xticks(interval_x)
        plt.yticks(interval_y)
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
