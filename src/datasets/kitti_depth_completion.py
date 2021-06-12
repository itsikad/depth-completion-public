import os
import glob
import pickle
import os.path as osp
from random import choice
from typing import Union, Optional, Tuple, Dict
from collections.abc import Sequence
from omegaconf import DictConfig

import numpy as np
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl

from utils import kitti_downloader
import utils.transforms
from utils.projections import sparse_depthmap_to_multi_scale_pointcloud, get_pose_pnp, transform_intrinsics
from utils.utils import depth_completion_collate_fn, to_int_list, array_to_tensor


_KITTI_TRAIN = 'train'
_KITTI_VALID = 'val'
_KITTI_VALID_SEL = 'val_selection_cropped'
_KITTI_TEST = 'test_depth_completion_anonymous'


def load_rgb(
    path: str
) -> Image.Image:

    return Image.open(path)


def load_rgb_near(
    path: str,
    max_frame_diff: int = 3
) -> Image.Image:

    fpath, fname = os.path.split(path)
    frame_id = int(fname.split('.')[0])

    candidates = [i for i in range(-max_frame_diff, max_frame_diff+1) if i != 0]

    count = 0
    while True:
        random_offset = choice(candidates)
        path_near = osp.join(fpath, f'{frame_id + random_offset:010d}.png')
        if osp.exists(path_near):
            break
        assert count < 20, f'cannot find a nearby frame in 20 trials for {path}'
        count += 1

    return load_rgb(path_near)


def close_pil_imgs(
    imgs: Union[Sequence, Image.Image]
) -> None:

    def close_img(img) -> None:
        if img is not None:
            img.close()

    if isinstance(imgs, Sequence):
        for img in imgs:
            close_img(img)
    else:
        close_img(imgs)


def load_depth(
    path: str
) -> Image.Image:

    img_np = np.array(Image.open(path), dtype=int)

    # Verify 16bit sparse and groundtruth depth maps
    assert np.max(img_np>255), f'Expecting a 16bit depth map at: {path}'

    # Convert to meters and convert back 
    # to PIL image to comply with torchvision transform input type
    return Image.fromarray(img_np.astype(np.float) / 256.)


def string_to_intrinsics(
    p_rect_line: str
) -> np.ndarray:

    """
    Retreives the camera intrinsics matrix from a string.
    
    Arguments:
        p_rect_line : a string corresponds to p_rect_0x line in calibration file
        
    Return:
        intrinsics : (3,3) matrix, camera intrinsics
    """
    
    mat_str = p_rect_line.split(":")[1].split(" ")[1:]  # strip the 'p_rect_0x: ' prefix and split elements
    mat_np = np.reshape(np.array([float(p) for p in mat_str]), (3,4)).astype(np.float32)  # convert to 3x4 numpy array
    intrinsics = mat_np[:3, :3]  # keep only the camera intrinsics part

    return intrinsics


def load_train_intrinsics(
    calibration_dir: str
) -> Dict:

    """
    Loads camera intrinsics matrix for training/validation splits.
    
    Arguments: 
        calibration_dir : KITTI depth completion calibration files path.
                          Expects camera calibration files to be at calibration_dir/yyyy_mm_dd/
    
    Return:
        intrinsics_dict : a dictionary of intrinsics matrices keyed by date+camera_idx, 
                          e.g., 2011_09_26_02  (02 for left camera) or 2011_09_26_03 (03 for right camera).
    """
    
    # Check directory and files exist
    if not osp.isdir(calibration_dir):
        raise ValueError(f'Missin calibration directory at {calibration_dir}')
    
    if not len(os.listdir(calibration_dir)) > 0:
        raise ValueError(f'Missing calibration files at {calibration_dir}')

    # Load intrinsics
    intrinsics_dict = dict()
    for date in os.listdir(calibration_dir):
        calib_file_path = osp.join(calibration_dir, date, 'calib_cam_to_cam.txt')

        with open(calib_file_path, 'r') as f:
            cam_02_str, cam_03_str = list(map(f.readlines().__getitem__, [25,33]))

        intrinsics_02 = string_to_intrinsics(cam_02_str)
        intrinsics_03 = string_to_intrinsics(cam_03_str)

        intrinsics_dict.update({date + '_02' : intrinsics_02, date + '_03' : intrinsics_03})

    return intrinsics_dict


def load_test_intrinsics_dict(
    calibration_file_path: str
) -> Tensor:

    """
    Loads camera intrinsics matrix for test/validation selection splits.
    
    Arguments: 
        calibration_file_path : calibration file path.
    
    Return:
        intrinsics : (3,3) intrinsics matrix
    """
    
    # check that file exist

    if not osp.isfile(calibration_file_path):
        raise ValueError(f'Missing calibration file in: {calibration_file_path}')

    with open(calibration_file_path, 'r') as f:
        intrinsics_str = f.readlines()[0].replace(' \n','').split(' ')  # file contains a single line

    intrinsics = np.reshape(np.array([float(p) for p in intrinsics_str]), (3,3)).astype(np.float32)

    return intrinsics


class KITTIDepthCompletionDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_config: Union[Dict, DictConfig],
        dataloader_config: Union[Dict, DictConfig],
    ) -> None:

        super().__init__()

        self.dataloader_config = dataloader_config
        self.dataset_config = dataset_config
        self.dataset_config.dataset_root = osp.expanduser(self.dataset_config.dataset_root)
        
        self.splits = [_KITTI_TRAIN, _KITTI_VALID, _KITTI_VALID_SEL, _KITTI_TEST]
        self.collate_fn = depth_completion_collate_fn
        
        # Transforms
        self.color_transforms = self._get_color_transforms()
        self.train_transforms = self._get_geometric_transforms(split=_KITTI_TRAIN)
        self.val_transforms = self._get_geometric_transforms(split=_KITTI_VALID)

    def prepare_data(
        self
    ) -> None:
    
        """
        Saves a list of all valid samples into file located in <root>/processed/<split_name>.txt
        Each line consists the files from which to construct a single sample:
            train/val splits : (RGB file path, sparse depth file path, intrinsics key, target file path)

            validation selection split: (RGB file path, sparse depth file path, intrinsics file path, target file path)

            test split: (RGB file path, sparse depth file path, intrinsics file path) 
                        
            the i-th element in the list corresponds to the i-th sample in the dataset
        """

        # Check if dataset exists, otheriwse download and unzip
        if not osp.isdir(self.dataset_config.dataset_root):
            kitti_downloader.download_kitti(
                root_dir=self.dataset_config.data_root,
                keep_zip=True
            )

        # Check if processed directory exists
        processed_dir = osp.join(self.dataset_config.dataset_root, 'processed')
        if not osp.isdir(processed_dir):
            os.mkdir(processed_dir)

        for split_name in self.splits:
            split_processed_file = osp.join(processed_dir, split_name + '.pickle')
            # Check if processed datasets already exist
            if not osp.isfile(split_processed_file):
                split_root = osp.join(self.dataset_config.dataset_root, split_name)

                # Create processed dataset file
                dataset = []
                if split_name in [_KITTI_TRAIN, _KITTI_VALID]:
                    # train / validation splits
                    rgb_list = glob.glob(split_root + '/*/image/*/*.png')
                    for rgb_fpath in rgb_list:
                        depth_fpath = rgb_fpath.replace('/image/', '/proj_depth/velodyne_raw/')
                        target_fpath = rgb_fpath.replace('/image/', '/proj_depth/groundtruth/')
                        intrinsics_key = rgb_fpath.split('_drive')[0][-10:] + '_' + rgb_fpath.split('image_')[1][:2]
                        if osp.isfile(depth_fpath) and osp.isfile(target_fpath):
                            dataset.append((depth_fpath, rgb_fpath, intrinsics_key, target_fpath))
                elif split_name == _KITTI_VALID_SEL:
                    # validation selection splits
                    rgb_list = glob.glob(split_root + '/image/*.png')
                    for rgb_fpath in rgb_list:
                        depth_fpath = rgb_fpath.replace('/image/', '/velodyne_raw/').replace('sync_image', 'sync_velodyne_raw')
                        target_fpath = rgb_fpath.replace('/image/', '/groundtruth_depth/').replace('sync_image', 'sync_groundtruth_depth')
                        intrinsics_fpath = rgb_fpath.replace('/image/', '/intrinsics/').replace('png', 'txt')
                        if osp.isfile(depth_fpath) and osp.isfile(target_fpath) and osp.isfile(intrinsics_fpath):
                            dataset.append((depth_fpath, rgb_fpath, intrinsics_fpath, target_fpath))
                elif split_name == _KITTI_TEST:
                    # test split
                    rgb_list = glob.glob(split_root + '/image/*.png')
                    for rgb_fpath in rgb_list:
                        depth_fpath = rgb_fpath.replace('/image/', '/velodyne_raw/')
                        intrinsics_fpath = rgb_fpath.replace('/image/', '/intrinsics/').replace('png', 'txt')
                        if osp.isfile(depth_fpath) and osp.isfile(intrinsics_fpath) :
                            # set sparse depth input as target in test mode for compatibility (no ground truth)
                            dataset.append((depth_fpath, rgb_fpath, intrinsics_fpath))
                else:
                    raise ValueError('Unknown split')

                # Check that dataset is not empty
                assert len(dataset)>0, f'Found 0 images in subfolders of: {split_root} \n'

                # Save to file
                processed_fpath = osp.join(processed_dir, split_name + '.pickle')
                with open(processed_fpath, 'wb') as f:
                    pickle.dump(dataset, f)

    def setup(
        self,
        stage: Optional[str] = None
    ) -> None:

        if stage == 'fit':
            self.train_dataset = KITTIDepthDataset(
                split=_KITTI_TRAIN,
                geometric_transforms=self.train_transforms,
                color_transforms=self.color_transforms,
                **self.dataset_config)

            self.val_dataset = KITTIDepthDataset(
                split=_KITTI_VALID,
                geometric_transforms=self.val_transforms,
                color_transforms=self.color_transforms,
                **self.dataset_config)

        if stage == 'test':
            self.test_dataset = KITTIDepthDataset(
                split=_KITTI_VALID_SEL,
                geometric_transforms=self.val_transforms,
                color_transforms=None,
                **self.dataset_config)

        if stage == 'predict' or stage is None:
            self.predict_dataset = KITTIDepthDataset(
                split=_KITTI_TEST,
                geometric_transforms=self.val_transforms,
                color_transforms=None,
                **self.dataset_config)

    def train_dataloader(
        self
    ) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            shuffle=True,
            **self.dataloader_config)
    
    def val_dataloader(
        self
    ) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.dataloader_config)

    def test_dataloader(
        self
    ) -> DataLoader:

        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.dataloader_config)

    def predict_dataloader(
        self
    ) -> DataLoader:

        return DataLoader(
            self.predict_dataset,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.dataloader_config)

    def _get_color_transforms(
        self
    ) -> transforms.Compose:

        return transforms.Compose([utils.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)])

    def _get_geometric_transforms(
        self,
        split: str
    ) -> transforms.Compose:

        transforms_list = []

        # Horizontal flip (training only)
        if self.dataset_config.horizontal_flip and split == _KITTI_TRAIN:
            transforms_list.append(utils.transforms.RandomHorizontalFlip(p=0.5))
        
        # Cropping
        if self.dataset_config.crop_type == 'bottom':
            crop_transform = utils.transforms.BottomCrop
        else:
            crop_transform = utils.transforms.CenterCrop

        transforms_list.append(crop_transform(output_size=(352,1216)))
        
        return transforms.Compose(transforms_list)


class KITTIDepthDataset(Dataset):

    """
    Returns a KITTI Depth Completion benchmark dataset
    """

    def __init__(
        self, 
        dataset_root: str, 
        split: str, 
        crop_type: str,
        geometric_transforms: Optional[transforms.Compose] = None,
        color_transforms: Optional[transforms.Compose] = None,
        use_rgb: Optional[bool] = True,
        use_rgb_near: Optional[bool] = False,
        use_grayscale:  Optional[bool] = False,
        use_pointcloud: Optional[bool] = False,
        use_intrinsics: Optional[bool] = False,
        use_pose: Optional[bool] = False,
        first_scale: Optional[int] = None,
        n: Optional[Union[list, int]] = None,
        k: Optional[Union[list, int]] = None,
        leaf_size: Optional[int] = 25,
        **kwargs
    ) -> None:

        """
        Initializes a KITTI Depth copmletion benchmark dataset.
        
        Arguments:
            root : path to dataset root directory, expects splits subfolder train, val, val_selection_cropped

            split : split name [_KITTI_TRAIN, _KITTI_VALID, _KITTI_VALID_SEL, _KITTI_TEST]

            crop_type : can be either 'bottom' or 'center' for BottomCrop and CenterCrop transform respectively

            geometric_transforms : transforms applied on all image-like inputs (PIL) including RGB and 
                                   sparse depth and targets before converting to numpy (crop, flip, etc)

            color_transforms : transforms applied on RGB inputs only after geometric transforms and before converting to numpy

            use_rgb: RGB image is not loaded if set to False (default: True)

            use_grayscale: Convert RGB image to grayscale
        
            use_rgb_near: if set to True, loads RGB image of nearby frame (max difference is 3 frames) (default: False)
        
            use_pointcloud: if set to True, also converts depth map to pointcloud

            use_intrinsics: Loads intrinsics matrix per frame if set to True (default: False)

            first_scale : if depth_map is set to 'pointcloud', then determines the scale of the first point cloud, set to 0 or 1
                          for example, setting init_scale = 0, the pointcloud in the first scale will be generated from the origin image scale (B,C,H,W),
                          alternatively, setting init_scale = 1 the pointcloud in the first scale will be generated after applying a maxpool2d with 2x2 kernel and stride 2 (B,C,H//2,W//2),

            n : if depth_map is set to 'pointcloud', a list that determines the number of points to keep per scale or an int for a single scale

            k : if depth_map is set to 'pointcloud', a list that determines the number of nearest neighbors to keep per scale or an int for a single scale

            leaf_size : KNN algorithm parameter, defaults to 25
        """

        # Load dataset list
        self.root = dataset_root
        self.split = split
        self.processed_fpath = self.root + '/processed/' + split + '.pickle'
        with open(self.processed_fpath, 'rb') as f:
            self.dataset = pickle.load(f)

        # Configure dataset
        self.crop_type = crop_type
        self.is_train_val = split in [_KITTI_TRAIN, _KITTI_VALID]
        self.use_rgb = use_rgb or use_grayscale
        self.use_rgb_near = use_rgb_near and split == _KITTI_TRAIN
        self.use_grayscale = use_grayscale
        self.use_pointcloud = use_pointcloud
        self.use_intrinsics = use_intrinsics or use_pointcloud
        self.use_pose = use_pose and split == _KITTI_TRAIN
        self.first_scale = first_scale
        self.leaf_size = leaf_size
        self.n, self.num_scales = to_int_list(n)
        self.k, _ = to_int_list(k)

        # Transforms
        self.geometric_transforms = geometric_transforms
        self.color_transforms = color_transforms if color_transforms is not None else utils.transforms.IdentityTransform()  # Defaults to identity transform
        self.grayscale_transform = utils.transforms.Grayscale(num_output_channels=1) if self.use_grayscale else None
        self.to_numpy = utils.transforms.ToNumpy()
        self.to_tensor = utils.transforms.ToTensor()

        # Load intrinsic matrices for train and validation sets
        # Test and val_selection_cropped sets has intrinsics matrix per sample
        if self.is_train_val:
            self._calibration_dir = osp.join(self.root, 'intrinsics')
            self._intrinsics_dict = load_train_intrinsics(self._calibration_dir)
        
    
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(
        self,
        idx: int
    ) -> Tuple:

        """
        Return the idx-th sample and applies transforms.

        Argument:
            idx : index of requested sample

        Return:
            rgb : a (C,H,W) tensor containing the RGB image or None (if use_rgb is False)
            
            rgb_near : a (C,H,W) tensor containing the RGB image of nearby frame or None (if use_rgb_near is False)

            gray_curr : current frame in grayscale
            
            gray_near : nearby frame in grayscale

            sdepth : (1,H,W) tensor containing the depth image as float ,depth in meters

            sdepth_mask : 1,H,W) bolean mask tensor, True for valid sdepth pixels and False otherwise

            pc_idx : list of (Ns,) tensor per scale that contains the points indices, range [0,HW-1] or None (if use_pointcloud is False)

            nbrs_idx : list of (Ns,Ks) tensor per scale that contains the neighbors indices, range [0,HW-1] or None (if use_pointcloud is False)

            nbrs_disp : list of (3,Ns,Ks) tensor per scale that contain the 3d displacement vector between a point and its neighbors or None (if use_pointcloud is False)

            intrinsics : (3,3) intrinsics matrix or None if use_intrinsics is False

            target : (1,H,W) tensor containing the depth ground truth (for test split, returns the sprase depth input)

            target_mask : (1,H,W) bolean mask tensor, True for valid depth pixels and False otherwise

        """
        
        # Load PIL images
        rgb_img, rgb_near_img, sdepth_img, target_img, intrinsics = self.loader(idx)
        orig_img_size = rgb_img.size

        # Transform
        
        rgb_img, rgb_near_img, sdepth_img, target_img = self.geometric_transforms((rgb_img, rgb_near_img, sdepth_img, target_img))
        if self.use_intrinsics:
            intrinsics = transform_intrinsics(
                intrinsics,
                orig_img_size=orig_img_size,
                cropped_img_size=rgb_img.size,
                crop_type=self.crop_type
            )


        # Generate grayscale
        gray_curr_img, gray_near_img = self.grayscale_transform((rgb_img, rgb_near_img)) if self.use_grayscale else (None, None)

        # Convert PIL images to numpy arrays
        # apply operations available on numpy arrays only
        rgb, rgb_near, gray_curr, gray_near, sdepth, target = self.to_numpy((rgb_img, rgb_near_img, gray_curr_img, gray_near_img, sdepth_img, target_img))
        close_pil_imgs((rgb_img, rgb_near_img, gray_curr_img, gray_near_img, sdepth_img, target_img))

        # Find translation and rotation
        # return the same image and no motion when PnP fails
        success, rotation, translation = get_pose_pnp(gray_curr, gray_near, sdepth, intrinsics) if self.use_pose else (False, None, None)
    
        if not success and self.use_pose:
            rgb_near = rgb
            gray_near = gray_curr

        # Convert to Tensors
        rgb, rgb_near, gray_curr, gray_near, sdepth, target = self.to_tensor((rgb, rgb_near, gray_curr, gray_near, sdepth, target))
        intrinsics = array_to_tensor(intrinsics)
        rotation = array_to_tensor(rotation)
        translation = array_to_tensor(translation)

        # Apply color transform (train only)
        if self.split == _KITTI_TRAIN:
            rgb, rgb_near = self.color_transforms((rgb, rgb_near))

        # Create masks
        sdepth_mask = (sdepth > 0.)
        target_mask = (target > 0.) if self.split != _KITTI_TEST else None

        # Create pointcloud
        if self.use_pointcloud:
            pc_idx, nbrs_idx, nbrs_disp = sparse_depthmap_to_multi_scale_pointcloud(sdepth=sdepth, intrinsics=intrinsics, first_scale=self.first_scale, num_points=self.n, num_neighbors=self.k, leaf_size=self.leaf_size)
        else:
            pc_idx = None
            nbrs_idx = None
            nbrs_disp = None

        return rgb, rgb_near, gray_curr, gray_near, sdepth, sdepth_mask, pc_idx, nbrs_idx, nbrs_disp, translation, rotation, intrinsics, target, target_mask

    def loader(
        self,
        idx: int
    ) -> Tuple:

        """
        Loads the idx-th sample.

        Argument:
            idx : sample index

        Return:
            A tuple (rgb, depth, intrinsics, target) where:

            rgb : PIL Image object containing the RGB image
            
            rgb_near : PIL Image object containing a nearby frame's RGB image

            depth : PIL Image object containing sparse depth input

            target : PIL Image object containing ground truth depth target

            intrinsics : for train/val splits a key to  intrinsics matrix,
                         for test/val_sel splits path of file containing the intrinsics file

        """

        sample = self.dataset[idx]
        depth = load_depth(sample[0])
        rgb = load_rgb(sample[1]) if self.use_rgb else None
        rgb_near = load_rgb_near(sample[1]) if self.use_rgb_near else None
        intrinsics = self._get_intrinsics(sample[2]) if self.use_intrinsics else None
        target = load_depth(sample[-1]) if self.split != _KITTI_TEST else None

        return rgb, rgb_near, depth, target, intrinsics
        
    def _get_intrinsics(
        self,
        path_or_key: str
    ) -> Tensor:

        """
        Load intrinsics matrices
        
        Arguments:

            path_or_key : string
                train/val splits : key to intrinsics_dict (<date>_<camera index>)
                val_sel/test splits : path to calibration file
                
        Return:
            intrinsics: (3,3) camera intrinsics

        """

        if self.is_train_val:
            intrinsics = self._intrinsics_dict[path_or_key]
        else:
            intrinsics = load_test_intrinsics_dict(path_or_key)

        return intrinsics