import itertools

import skimage.measure
import skimage.io
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from microfilm.microplot import microshow
from microfilm import colorify
from microfilm.dataset import Nparray
from .utils import create_rectangle

def get_roi_cm(roi_path=None, roi_im=None):
    """
    Find the center of mass of a roi defined as a single label in an image.

    Parameters
    ----------
    roi_path: str
        path to roi image
    roi_im = 2d array
        image containing roi

    Returns
    -------
    cm: 1d array
        center of mass of roi

    """
    
    if roi_path is None:
        if roi_im is None:
            raise Exception("You have to provide an image if you don't provide a path")              
    else:
        roi_im = skimage.io.imread(roi_path)
    
    roi_im = skimage.measure.label(roi_im)
    roi_props = skimage.measure.regionprops_table(roi_im, properties=('label','centroid'))
    cm = np.stack([roi_props['centroid-0'].astype(np.uint16),roi_props['centroid-1'].astype(np.uint16)],axis=1)
    
    return cm


def save_labelled_roi(file_path, roi_path=None, roi_im=None):
    """
    Save an image of the labelled roi

    Parameters
    ----------
    file_path: str or Path
    roi_path: str
        path to roi image
    roi_im = 2d array
        image containing roi

    Returns
    -------

    """
    
    if roi_path is None:
        if roi_im is None:
            raise Exception("You have to provide an image if you don't provide a path")              
    else:
        roi_im = skimage.io.imread(roi_path)
    
    roi_im = skimage.measure.label(roi_im).astype(np.uint8)
    skimage.io.imsave(file_path, roi_im, check_contrast=False)


def create_concentric_mask(center, im_dims, sector_width=10, num_sectors=10):
    """
    Create a labelled mask of disk split in concentric rings.

    Parameters
    ----------
    center: list or array
        2d position of center of disk(s)
    im_dims: list
        image size
    sector_width: int
        ring thickness
    num_sectors: int
        number of rings to define

    Returns
    -------
    concentric_labels: 3d array
        labelled image with concentric rings
        each plane [x,:,:] corresponds to a roi

    """
    
    if isinstance(center, list):
        if isinstance(center[0], list):
            center = np.array(center)
        else:
            center = np.array([center])

    yy, xx = np.meshgrid(np.arange(im_dims[1]),np.arange(im_dims[0]))
    
    concentric_labels = np.zeros((center.shape[0], im_dims[0], im_dims[1]), dtype=np.uint16)
    
    for r in range(center.shape[0]):
        roi_mask = np.zeros(im_dims, dtype=np.bool_)
        concentric_masks = [roi_mask]

        for ind, i in enumerate(np.arange(sector_width, sector_width*num_sectors+1, sector_width)):

            temp_roi = np.sqrt((xx - center[r,0])**2 + (yy - center[r,1])**2) < i
            concentric_masks.append((ind+1)*(temp_roi*~roi_mask))
            roi_mask = temp_roi

        concentric_labels[r,:,:] = np.sum(np.array(concentric_masks),axis=0)
    
    return concentric_labels

def create_sector_mask(center, im_dims, angular_width=20, max_rad=50, ring_width=None):
    """
    Create a labelled mask of a disk or ring split in angular sectors. If radius is
    provided, the mask is a ring otherwise a disk.

    Parameters
    ----------
    center: list
        2d position of center of disk
    im_dims: list
        image size
    angular_width: int
        size of angular sectors in degrees
    max_rad: float
        disk radius
    ring_width: int
        ring width


    Returns
    -------
    sector_labels: 3d array
        labelled image with concentric rings
        each plane [x,:,:] corresponds to a roi

    """

    if isinstance(center, list):
        if isinstance(center[0], list):
            center = np.array(center)
        else:
            center = np.array([center])

    yy, xx = np.meshgrid(np.arange(im_dims[1]),np.arange(im_dims[0]))

    sector_labels = np.zeros((center.shape[0], im_dims[0], im_dims[1]), dtype=np.uint16)
    for r in range(center.shape[0]):
        angles = np.arctan2(xx-center[r,0],yy-center[r,1])
        angles %= (2*np.pi)
        rad_mask = np.sqrt((xx - center[r,0])**2 + (yy - center[r,1])**2)
        
        if ring_width is None:
            rad_mask = rad_mask < max_rad
        else:
            rad_mask = (rad_mask < max_rad) & (rad_mask > max_rad-ring_width)

        sector_masks = [rad_mask*(ind+1)*((angles >= np.deg2rad(i)) *(angles < np.deg2rad(i+angular_width)))
                        for ind, i in enumerate(np.arange(0,360-angular_width+1,angular_width))]
        sector_labels[r,:,:] = np.sum(np.array(sector_masks),axis=0)

    return sector_labels

def roi_reorder(sectors, new_order):
    """
    Given a roi with dimensions RxWxH where R are rois,
    reorder the roi order.

    Parameters
    ----------
    sectors : 3D array
        numpy array with dimensions RxHxW where R=roi,
        H=height, W=width
    new_order : list
        list of indices in correct order

    Returns
    -------
    reordered: 3D array
        numpy array with dimensions RxHxW
        with reorders rois
    """

    reordered = sectors[new_order,:,:]
    return reordered

def get_cmap_labels(im_label, cmap_name='cool', alpha=1):
    """Create list of L colors where L is the number of labels in the image"""

    #cmap_original = plt.get_cmap(cmap_name)
    cmap_original = colorify.cmaps_def(cmap_name=cmap_name)
    colors = cmap_original(np.linspace(0,1,im_label.max()+1))
    # background should be transparent and black so that it disappears
    # when used allone (transparent) or in composite mode
    colors[0,:] = 0
    colors[1::,-1] = alpha
    cmap = matplotlib.colors.ListedColormap(colors)
    
    return colors, cmap
    
def nan_labels(im_label):
    """Return a label image where background is set to nan for transparency"""
    
    im_label_nan = im_label.astype(float)
    im_label_nan[im_label_nan==0] = np.nan
    
    return im_label_nan

def sub_time_lapse_generator(time_image, channels, min_time=0, max_time=None, step=1):
    """
    Create a frame generator for a dataset only for a subset of time frames.

    Parameters
    ----------
    time_image: array or microfilm.dataset
        should be a 3D TxHxW numpy array or any microfilm.dataset object
    channels: str or list of str
        name(s) of channels to analyze
    min_time: int
        first time point to consider
    max_time: int
        last time point to consider
    step: int
        step between time points

    Returns
    -------
    time_image_part: generator
        frame generator
    max_time: int
        index of last frame

    """

    if isinstance(time_image, np.ndarray):
        time_image = Nparray(time_image)
    
    if max_time is None:
        max_time = time_image.max_time

    time_gen = time_image.frame_generator(channel=channels)
    time_image_part = itertools.islice(time_gen, min_time, max_time, step)

    return time_image_part, max_time

def measure_intensities(time_image, im_labels, channels, min_time=0, max_time=None, step=1):
    """
    Measure average intensity in a time-lapse image using a labelled image

    Parameters
    ----------
    time_image: array or microfilm.dataset
        should be a 3D CxTxHxW numpy array or any microfilm.dataset object
    im_labels: 3d array
        labelled image with dimension RxHxW where R is for rois
    channels: str or list of str
        name(s) of channels to analyze
    min_time: int
        first time point to consider
    max_time: int
        last time point to consider
    step: int
        step between time points

    Returns
    -------
    signal: 4d xarray
        signal array with dimensions TxSxCxR where
        T=time, S=splits, C=channels, R=rois

    """

    time_image_part, max_time = sub_time_lapse_generator(
        time_image=time_image, channels=channels, min_time=min_time, max_time=max_time, step=step)
    
    num_channels = len(channels) if isinstance(channels, list) else 1

    signal = np.zeros((len(range(min_time, max_time, step)), im_labels.max(), num_channels, im_labels.shape[0]))

    for t, im_np in enumerate(time_image_part):
        for r in range(im_labels.shape[0]):
            measures = skimage.measure.regionprops_table(im_labels[r], 
                                                            intensity_image=im_np, properties=('label','mean_intensity'))        
            if im_np.ndim == 3:
                signal[t,:,:,r] = np.stack([measures['mean_intensity-'+str(k)] for k in range(im_np.shape[2])], axis=1)
            else:
                signal[t,:,0,r] = measures['mean_intensity']
    
    signal = xr.DataArray(signal, dims=("time", "sector", "channel", "roi"))
    
    return signal

def scan_intensities(
    time_image, channels, src, dst, min_time=0, max_time=None, step=1, box_width=1,
    reduce_func=np.mean, line_kwargs={'mode':'reflect'}):
    """
    Measure intensity along a line from scr to dst with a thickness of linewidth
    for a given channel(s) and time-frames.

    Parameters
    ----------
    time_image: array or microfilm.dataset
        should be a 3D TxHxW numpy array or any microfilm.dataset object
    channels: str or list of str
        name(s) of channels to analyze
    src: array_like, shape (2, )
        The coordinates of the start point of the scan line.
    dst: array_like, shape (2, )
        The coordinates of the end point of the scan line. The destination point
        is included in the profile, in contrast to standard numpy indexing.
    min_time: int
        first time point to consider
    max_time: int
        last time point to consider
    step: int
        step between time points
    box_width: int, optional
        Width of the scan, perpendicular to the line
    reduce_func: callable, optional
        Function used to calculate the aggregation of pixel values perpendicular
        to the profile_line direction when linewidth > 1. If set to None the unreduced array will be returned.
    line_kwargs: dict
        Keyword arguments passed to skimage.measure.profile_line

    Returns
    -------
    signal: 3d xarray
        signal array with dimensions LxTxC where
        C=channels, T=frames, L=line length
    rectangle: 2d array
        coordinates of the box

    """

    if line_kwargs is None:
        line_kwargs = {}

    # compute box for plotting
    rectangle = create_rectangle(src, dst, box_width)

    time_image_part, max_time = sub_time_lapse_generator(
        time_image=time_image, channels=channels, min_time=min_time, max_time=max_time, step=step)
    
    num_channels = len(channels) if isinstance(channels, list) else 1

    scan0 = skimage.measure.profile_line(
        image=time_image.load_frame(channel_name=time_image.channel_name[0], frame=0), 
        src=src, dst=dst, linewidth=box_width, reduce_func=reduce_func, **line_kwargs)

    signal = np.zeros((num_channels, len(range(min_time, max_time, step)), len(scan0)))

    for t, im_np in enumerate(time_image_part):

        scan = skimage.measure.profile_line(
            image=im_np, src=src, dst=dst, linewidth=box_width, reduce_func=reduce_func, **line_kwargs)

        if scan.ndim == 2:
            signal[:,t,:] = np.swapaxes(scan, 0, 1)
        else:
            signal[0,t,:] = np.swapaxes(scan, 0, 1)
    
    signal = xr.DataArray(signal, dims=("channel", "time", "profile"))
    
    return signal, rectangle

def plot_signals(signal, channel=0, roi=0, color_array=None, ax=None):
    """
    Plot extracted signal with specific colormap
    
    Parameters
    ----------
    signal: 4d xarray or Numpy array with same dimension order
        signal array with dimensions TxSxCxR where
        T=time, S=splits, C=channels, R=rois
    color_array: 2d array
        N x 4 array of RGBA colors where N >= sectors
    ax: Matplotlib axis

    Returns
    -------
    fig: Matplotlib figure
    
    """

    if isinstance(signal, np.ndarray):
        signal = xr.DataArray(signal, dims=("time", "sector", "channel", "roi"))
        
    signal_sel = signal.sel(channel=channel, roi=roi)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = ax.figure
    for i in range(signal_sel.shape[1]):
        if color_array is not None:
            ax.plot(signal_sel[:,i], color=color_array[i])
        else:
            ax.plot(signal_sel[:,i])
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Intensity', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    fig.tight_layout()
    
    return fig
    
def plot_sectors(
    image, sectors, channel=None, time=0, roi=None,
    cmap=None, im_cmap='gray', alpha=0.5, show_labels=False, **micro_kwargs):
    """
    Plot image and overlayed sectors with a given colormap
    
    Parameters
    ----------
    image: dataset object
        image to be plotted
    sectors: 3d array
        labelled image of sectors RxHxW, first dimension can contains multiple rois
    channel: str
        name of channel to plot, defaults to first channel
    time: int
        frame to plot, defaults to t=0
    roi: int
        index of roi to plot, if None, all are plotted
    cmap: list or single Matplotlib colormap or colormap name str
        colormap for each split mask corresponding to a roi
    im_cmap: Matplotlib colormap
        colormap for image
    alpha: float
        transparency of rois [0,1]
    show_labels: bool
        show roi indices as title, default False
    micro_kwargs: dict
        additional parameters are passed to microshow

    Returns
    -------
    microim: Microimage object
    
    """
    
    if im_cmap is None:
        im_cmap = plt.get_cmap('gray')
    
    if cmap is None:
        cmap = ['viridis' for x in range(sectors.shape[0])]
    if not isinstance(cmap, list):
        cmap = [cmap]

    if channel is None:
        channel = image.channel_name[0]

    im = image.load_frame(channel,time)
    if roi is not None:
        sectors = sectors[[roi]]

    # compute cmap for each roi sector map. cmaps are transparent outside the roi
    num_roi = sectors.shape[0]
    cmaps = []
    for i in range(num_roi):
        if isinstance(cmap[i], str):
            temp_cmap = get_cmap_labels(sectors[i], cmap_name=cmap[i])
            cmaps.append(temp_cmap[1])
        else:
            cmaps.append(cmap[i])
    
    # creata an image by alpha superposing each mask on top of image
    # as masks are transparent outside roi, the underlying image doesn't
    # become increasingly opaque
    microim = microshow(
        images=[im, *sectors], cmaps=[im_cmap, *cmaps],
        proj_type='alpha', alpha=alpha,
        channel_names=['image']+[str(i) for i in range(num_roi)],
        channel_label_show=show_labels, **micro_kwargs)
    
    return microim

def save_signal(signal, name='mycsv.csv', channels=None):
    """
    Save the sector signal in a CSV file with a given name

    Parameters
    ----------
    signal: 4d xarray
        signal array with dimensions TxSxCxR where
        T=time, S=splits, C=channels, R=rois
    name: str
        file name for export (should end in .csv)

    Returns
    -------
    
    """
    
    signal_df = signal.to_dataframe(name='intensity').reset_index()

    if channels is not None:
        for ind, c in enumerate(channels):
            signal_df.loc[signal_df['channel'] == ind, 'channel'] = c

    signal_df.to_csv(name, index=False)

    '''if signal.ndim == 3:
        format = 'long'
        if channels is None:
            channels = ['channel-'+str(i) for i in range(signal.shape[2])]

    def reshape_df(signal_array, int_name='intensity'):
        df = pd.DataFrame(signal_array)
        df = df.reset_index().rename({'index': 'time'}, axis='columns')
        df = pd.melt(df, id_vars='time', var_name='sector', value_name=int_name)
        return df

    if format == 'wide':
        signal_df = pd.DataFrame(signal)
        signal_df.to_csv(name, index=False)
    elif format == 'long':
        if signal.ndim == 2:
            signal_df = reshape_df(signal)
        else:
            dfs = []
            for k in range(signal.shape[2]):
                temp_df = reshape_df(signal[:, :, k], int_name='intensity')
                temp_df['channel'] = channels[k]
                dfs.append(temp_df)
            signal_df = pd.concat(dfs)
            
        signal_df.to_csv(name, index=False)'''