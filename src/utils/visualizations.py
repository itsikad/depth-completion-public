from PIL import Image
from typing import Any, Union, Optional, Tuple, List, Dict

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

from .projections import sparse_back_projection


def torch_to_array(
    tensor: Tensor
) -> np.ndarray:

    """
    Converts a pytorch tensor to numpy arrow.
    """

    assert type(tensor) == Tensor, 'Input must be a pytorch tensor'

    # Convert tensor to numpy array
    return tensor.detach().cpu().numpy().squeeze()


def convert_depth_colormap(
    depth: Union[Tensor, np.ndarray],
    d_min: Optional[Union[Tensor, np.ndarray, Any]] = None,
    d_max: Optional[Union[Tensor, np.ndarray, Any]] = None
) -> np.ndarray:

    """
    Converts a depth map to depth color map

    Arguments:
        depth :  a (1,H,W) or (H,W) tensor or numpy array
    
        d_min : minimum depth value, can be any numeric, default: 0

        d_max : maximum depth value, can be any numeric, default: depth.max()

    Return:
        Depth colormap as numpy array (H,W,C)
    """

    # Convert to numpy
    if isinstance(depth, Tensor):
        depth = torch_to_array(depth)
    
    # Scale
    if d_min is None:
        d_min = 0.
    
    if d_max is None:
        d_max = depth.max()
    
    depth = (depth - d_min) / (d_max - d_min)

    # Create colormap
    cmap = plt.cm.jet
    depth_colormap = cmap(depth)[:, :, :3]  # H, W, C

    return depth_colormap


def plot_grayscale(
    img: Union[Tensor, np.ndarray],
    fig: Optional[Figure] = None,
    subplotspec: Optional[SubplotSpec] = None,
    title: Optional[str] = None
) -> Figure:

    # Convert tensor to numpy array
    if isinstance(img, Tensor):
        img = torch_to_array(img)
    
    # Permute (1,H,W) to (H,W,1) if neccessary 
    if img.shape[0] == 1:
        img = img.transpose(1,2,0)

    # Scale/clip to (0,1)
    if img.max() > 1.0:
        img = img / img.max()
    
    if img.min() < 0.:
        img = np.clip(img, a_min=0., a_max=1.)

    # Create figure if not passed
    if fig is None:
        fig = plt.figure()

    # Set subplotspec if not passed   
    if subplotspec is None:
        gs = GridSpec(1,1, figure=fig)
        subplotspec = gs[0]

    ax = fig.add_subplot(subplotspec)
    ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    if title is not None:
        ax.set_title(title)

    return fig


def plot_rgb(
    img: Union[Tensor, np.ndarray],
    fig: Optional[Figure] = None,
    subplotspec: Optional[SubplotSpec] = None,
    title: Optional[str] = None
) -> Figure:

    """
    Generate RGB plot.

    Arguments:
        img : a (C,H,W) tensor or numpy array

        fig (optional) : a matplotlib Figure object to add the plot into

        subplotspec (optional) : a matplotlib SubplotSpec object that determines the add axis position in the plot

        title (optional): a string, title for the current figure/axis

    Returns:
        fig : a matplotlib figure
              If fig is not None, add the plot to fig at subplotspec (or if none at (0,0))
              otherwise, returns a new matplotlib figure object.
    """

    # Convert tensor to numpy array
    if isinstance(img, Tensor):
        img = torch_to_array(img)
    
    # Permute (C,H,W) to (H,W,C) if neccessary 
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)

    # Create figure if not passed
    if fig is None:
        fig = plt.figure()

    # Set subplotspec if not passed   
    if subplotspec is None:
        gs = GridSpec(1,1, figure=fig)
        subplotspec = gs[0]

    ax = fig.add_subplot(subplotspec)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    if title is not None:
        ax.set_title(title)

    return fig


def plot_depth_map(
    depth: Union[Tensor, np.ndarray],
    fig: Optional[Figure] = None,
    subplotspec: Optional[SubplotSpec] = None,
    title: Optional[str] = None
) -> Figure:

    """
    Generate depth map plot

    Arguments:
        depth : (H,W) or (1,H,W) tensor or numpy array

        fig (optional) : a matplotlib Figure object to add the plot into

        subplotspec (optional) : a matplotlib SubplotSpec object that determines the add axis position in the plot

        title (optional): a string, title for the current figure/axis

    Returns:
        fig : a matplotlib figure
              If fig is not None, add the plot to fig at subplotspec (or if none at (0,0))
              otherwise, returns a new matplotlib figure object.
    """

    # Convert tensor to numpy array
    if isinstance(depth, Tensor):
        depth = torch_to_array(depth)

    # Convert to colormap
    depth_cmap = convert_depth_colormap(depth)

    return plot_rgb(img=depth_cmap, fig=fig, subplotspec=subplotspec, title=title)


def plot_depth_error(
    pred: Union[Tensor, np.ndarray],
    target: Union[Tensor, np.ndarray],
    fig: Optional[Figure] = None,
    subplotspec: Optional[SubplotSpec] = None,
    title: Optional[str] = None
) -> Figure:
    
    """
    Generate depth error colormap.

    Arguments:
        pred: a (1,H,W) or (H,W) tensor or numpy array
        
        target: a (1,H,W) or (H,W) tensor or numpy array, must be the same type of pred
        
        fig (optional) : a matplotlib Figure object to add the plot into

        subplotspec (optional) : a matplotlib Gridspec object that determines the add axis position in the plot

        title (optional): a string, title for the current figure/axis

    Returns:
        fig : a matplotlib figure
        If fig is not None, add the plot to fig at subplotspec (or if none at (0,0))
        otherwise, returns a new matplotlib figure object.
    """

    assert type(pred) == type(target), f'Pred and target must be of the same type, got {type(pred)} and {type(target)} respectively'

    mask = target > 0
    err = abs((pred - target) * mask)

    return plot_depth_map(depth=err, fig=fig, subplotspec=subplotspec, title=title)


def plot_overlay_rgb_depth_error(
    img: Union[Tensor, np.ndarray], 
    pred: Union[Tensor, np.ndarray],
    target: Union[Tensor, np.ndarray],
    percent: Optional[float] = 0.2,
    fig: Optional[Figure] = None,
    subplotspec: Optional[SubplotSpec] = None,
    title: Optional[str] = None
) -> Figure:

    """
    Generate RGB image and the top (percent)% errors defined by percent.

    Arguments:
        img: a (C,H,W) tensor or numpy array

        pred: a (1,H,W) or (H,W) tensor or numpy array
        
        target: a (1,H,W) or (H,W) tensor or numpy array, must be the same type of pred

        percent: (float) determines the top (percent)% of errors to plot
        
        fig (optional) : a matplotlib Figure object to add the plot into

        subplotspec (optional) : a matplotlib Gridspec object that determines the add axis position in the plot

        title (optional): a string, title for the current figure/axis

    Returns:
        fig : a matplotlib figure
              If fig is not None, add the plot to fig at subplotspec (or if none at (0,0))
              otherwise, returns a new matplotlib figure object.
    """

    assert type(img) == type(pred) == type(target), f'RGB, pred and target must be of the same type, got {type(img)}, {type(pred)} and {type(target)} respectively'

    if isinstance(img, Tensor):
        img = torch_to_array(img)
        pred = torch_to_array(pred)
        target = torch_to_array(target)
    
    mask = target > 0
    err = abs((pred - target) * mask)
    threshold = np.percentile(err[target>0], 1-percent)
    mask = err < threshold

    img = img * mask  #  zero high error

    return plot_rgb(img=img, fig=fig, subplotspec=subplotspec, title=title)
    

def plot_3d_depthmap(
    depth: Union[Tensor, np.ndarray],
    intrinsics: Union[Tensor, np.ndarray],
    fig: Optional[Figure] = None,
    subplotspec: Optional[SubplotSpec] = None,
    title: Optional[str] = None
) -> Figure:

    """
    Generate 3D depth map

    Arguments:
        depth: a (1,H,W) or (H,W) tensor or numpy array

        intrinsics : (3,3) tensor/array containing the camera intrinsics matrix
        
        fig (optional) : a matplotlib Figure object to add the plot into

        subplotspec (optional) : a matplotlib Gridspec object that determines the add axis position in the plot

        title (optional): a string, title for the current figure/axis

    Returns:
        fig : a matplotlib figure
              If fig is not None, add the plot to fig at subplotspec (or if none at (0,0))
              otherwise, returns a new matplotlib figure object.
    """

    if isinstance(depth, Tensor):
        depth = torch_to_array(depth)
        intrinsics = torch_to_array(intrinsics)
    
    # Create colormap
    depth_cmap = convert_depth_colormap(depth)

    # Create mask
    mask = (depth > 0)

    # Create point cloud
    pc_xyz, _ = sparse_back_projection(sdepth=depth, intrinsics=intrinsics)

    # Extract features
    pc_rgb = depth_cmap[mask]

    # Create figure if not passed
    if fig is None:
        fig = plt.figure()

    # Set subplotspec if not passed   
    if subplotspec is None:
        gs = GridSpec(1,1, figure=fig)
        subplotspec = gs[0]

    ax = fig.add_subplot(subplotspec, projection='3d')

    ax.scatter(xs=pc_xyz[:,0], ys=pc_xyz[:,2], zs=pc_xyz[:,1], c=pc_rgb)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if title is not None:
        ax.set_title(title)

    return fig


def gen_output_analysis_figure(
    pred: Union[Tensor, np.array],
    target: Union[Tensor, np.array], 
    img: Union[Tensor, np.array],
    title: Optional[str] = None
) -> Figure:

    """
    Generate 3D depthmap

    Arguments:
        pred : a (1,H,W) or (H,W) tensor or numpy array of predicted depth map

        target : a (1,H,W) or (H,W) tensor or numpy array of the target depth map

        img : a (C,H,W) tensor or numpy array containing an RGB image
        
        title (optional) : a string, title for the current figure/axis

    Returns:
        fig : a matplotlib figure
    """


    if isinstance(pred, Tensor):
        pred = torch_to_array(pred)

    if isinstance(target, Tensor):
        target = torch_to_array(target)

    if isinstance(img, Tensor):
        img = torch_to_array(img)

    # Create figure
    fig = plt.figure(figsize=(20,10), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(3,2)

    fig = plot_depth_map(depth=pred, fig=fig, subplotspec=gs[0,:], title='Predicted depth map')
    fig = plot_depth_map(depth=target, fig=fig, subplotspec=gs[1,0], title='Target depth map')
    fig = plot_rgb(img=img, fig=fig, subplotspec=gs[1,1], title='RGB Image')
    fig = plot_depth_error(pred=pred, target=target, fig=fig, subplotspec=gs[2,0], title='Depth error map')
    fig = plot_overlay_rgb_depth_error(img=img, pred=pred, target=target, percent=0.2, fig=fig, subplotspec=gs[2,1], title='20% percentile overlay')

    return fig