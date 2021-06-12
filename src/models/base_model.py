import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__()

    
    def val_post_process(self, pred, *args, **kwargs):
        """
        Override this method to add post validation processing
        """

        return pred
