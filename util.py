import logging
import torch
import random
import numpy as np
import os

'''
Logger
'''
logger = logging.getLogger("basic_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

__all__ = ["logger"]




'''
Seeding (from https://github.com/HilaManor/AudioEditingCode/blob/codeclean/code/utils.py)
'''
def set_reproducability(seed: int, extreme: bool = False) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Extreme options
        if extreme:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        # Even more extreme options
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")