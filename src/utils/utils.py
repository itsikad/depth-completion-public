from typing import List, Tuple, Dict, MutableSequence, Union

import numpy as np

import torch
from torch import Tensor


def get_padding(kernel_size: Tuple[int, int]) -> Tuple:
    """
        Returns the padding such that the spatial dimensions 
        of the convolution output are identical to the input dimensions.
        Assuming stride=1, dilation=1

        Arguments:
            kernel_size : the kernel size given as a tuple of two integers, e.g. (int,int) or [int,int], etc.

        Return:
            the padding per dimension
    """
    
    return tuple([x // 2 for x in kernel_size])


def to_int_list(
    m: Union[int, MutableSequence, None]
) -> Tuple[List, int]:

    if m is None:
        num_scales = None
        m = None
    elif isinstance(m, MutableSequence):
        num_scales = len(m)
        m = [int(x) for x in m]
    elif isinstance(m, int):
        num_scales = 1
        m = [m]
    else:
        raise ValueError('Expected input to be a list, int or None but got {} '.format(type(m)))

    return m, num_scales


def array_to_tensor(
    array: Union[np.ndarray, None]
) -> Tensor:

    return None if array is None else torch.tensor(array, dtype=torch.float32)


class DepthCompCollationFn:
    """
    A custom collation function.
    """

    def __init__(
        self,
        data
    ) -> None:
        """
        A custom batching class which collates data tensors regularly by stacking along
        new dimension (0) and indexing tensors by adding sample index to first dimension 
        and concatenating to allow easy indexing and save revaluating .nonzero elements

        Arguments:
            data : a list of tuples where each tuple represents the inputs that correponds to a single sample
                data[i][0] : (C,H,W) tensor containing the RGB image of the i-th sample
                data[i][1] : (C,H,W) tensor containing the RGB image of a nearby frame of the i-th sample
                data[i][2] : (1,H,W) tensor containing a grayscale image of the i-th sample
                data[i][3] : (1,H,W) tensor containing a grayscale image of a nearby frame of the i-th sample
                data[i][4] : (1,H,W) tensor containing the depth image as float, depth in meters
                data[i][5] : (1,H,W) bolean mask tensor, True for valid sparse depth input pixels and False otherwise
                data[i][6] : list of (1,Ns) tensor per scale that contains the points indices of the i-th sample
                data[i][7] : list of (1,Ns,Ks) tensor per scale that contains the neighbors indices, range [0,HW-1] of the i-th sample
                data[i][8] : list of (3,Ns,Ks) tensor per scale that contain the 3d displacement vector between a point and its neighbors of the i-th sample
                data[i][9] : (3,) translation vector
                data[i][10] : (3,3) rotation matrix
                data[i][11] : (3,3) intrinsics matrix
                data[i][12] : (1,H,W) tensor containing the Depth ground truth (TARGET) of the i-th sample
                data[i][13] : (1,H,W) bolean mask tensor, True for valid depth pixels and False otherwise
                
        Return:
            A dict with keys: rgb, rgb_near, gray_curr, gray_near, sdepth, sdepth_mask, pc_idx, nbrs_idx, nbrs_disp, t_vec, r_mat, intrinsics, target, target_mask
        """

        transposed_data = list(zip(*data))

        self.rgb = self._stack_tensors(transposed_data[0])  # (B,C,H,W)
        self.rgb_near = self._stack_tensors(transposed_data[1])  # (B,C,H,W)
        self.gray_curr = self._stack_tensors(transposed_data[2])  # (B,1,H,W)
        self.gray_near = self._stack_tensors(transposed_data[3])  # (B,1,H,W)
        self.sdepth = self._stack_tensors(transposed_data[4])  # (B,1,H,W)
        self.sdepth_mask = self._stack_tensors(transposed_data[5])  # (B,1,H,W)
        self.translation = self._stack_tensors(transposed_data[9])  # (B,3)
        self.rotation = self._stack_tensors(transposed_data[10])  # (B,3,3)
        self.intrinsics = self._stack_tensors(transposed_data[11])  # (B,3,3)
        self.target = self._stack_tensors(transposed_data[12]) # (B,1,H,W)
        self.target_mask = self._stack_tensors(transposed_data[13])  # (B,1,H,W)
        
        if data[0][6] is None:
            self.pc_idx, self.nbrs_idx, self.nbrs_disp = torch.empty(0), torch.empty(0), torch.empty(0)
        else:
            self.pc_idx, self.nbrs_idx, self.nbrs_disp = self._stack_pointcloud(
                data=transposed_data[6:9],
                num_scales=len(data[0][6])
                )

    def _stack_tensors(
        self,
        tensors
    ) -> Tensor:

        if tensors[0] is None:
            return torch.empty(0)
        else:
            return torch.stack(tensors, 0)

    def _stack_pointcloud(
        self,
        data,
        num_scales
    ) -> Tuple[List,...]:

        pc_idx = []
        nbrs_idx = []
        nbrs_disp = []
        for s in range(num_scales):
            pc_idx.append(torch.stack(list(zip(*data[0]))[s]))
            nbrs_idx.append(torch.stack(list(zip(*data[1]))[s]))
            nbrs_disp.append(torch.stack(list(zip(*data[2]))[s]))

        return pc_idx, nbrs_idx, nbrs_disp


    def pin_memory(
        self
    ) -> Dict:

        self.rgb = self.rgb.pin_memory()
        self.rgb_near = self.rgb_near.pin_memory()
        self.gray_curr = self.gray_curr.pin_memory()
        self.gray_near = self.gray_near.pin_memory()
        self.sdepth = self.sdepth.pin_memory()
        self.sdepth_mask = self.sdepth_mask.pin_memory()
        self.pc_idx = [x.pin_memory() for x in self.pc_idx]
        self.nbrs_idx = [x.pin_memory() for x in self.nbrs_idx]
        self.nbrs_disp = [x.pin_memory() for x in self.nbrs_disp]
        self.translation = self.translation.pin_memory()
        self.rotation = self.rotation.pin_memory()
        self.intrinsics = self.intrinsics.pin_memory()
        self.target_mask = self.target_mask.pin_memory()
        self.target = self.target.pin_memory()

        return {
            'rgb': self.rgb,
            'rgb_near': self.rgb_near,
            'gray_curr': self.gray_curr,
            'gray_near': self.gray_near,
            'sdepth': self.sdepth,
            'sdepth_mask': self.sdepth_mask,
            'pc_idx': self.pc_idx,
            'nbrs_idx': self.nbrs_idx,
            'nbrs_disp': self.nbrs_disp,
            'translation': self.translation,
            'rotation': self.rotation,
            'intrinsics': self.intrinsics,
            'target_mask': self.target_mask,
            'target': self.target
        }


def depth_completion_collate_fn(batch):

    """
    Wrapper for custom collation class for depth completion. 
    """

    return DepthCompCollationFn(batch)


class ObjectBuilder:
    def __init__(self):
        self._constructors = {}

    def register_constructor(self, name, constructor):
        self._constructors[name] = constructor

    def create(self, name, **kwargs):
        constructor = self._constructors.get(name)
        if not constructor:
            raise ValueError(name)
        return constructor(**kwargs)