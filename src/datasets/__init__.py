from utils.utils import ObjectBuilder

from .kitti_depth_completion import KITTIDepthCompletionDataModule

builder = ObjectBuilder()
builder.register_constructor('kitti_depth_completion', KITTIDepthCompletionDataModule)
