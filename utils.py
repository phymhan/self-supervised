import os
import sys
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import ast
import argparse
import torch
import wandb

def get_name_from_args(args):
    name = getattr(args, 'name', Path(args.log_dir).name if hasattr(args, 'log_dir') else 'unknown')
    return name

def setup_wandb_run_id(log_dir, resume=False):
    # NOTE: if resume, use the existing wandb run id, otherwise create a new one
    os.makedirs(log_dir, exist_ok=True)
    file_path = Path(log_dir) / 'wandb_run_id.txt'
    if resume:
        assert file_path.exists(), 'wandb_run_id.txt does not exist'
        with open(file_path, 'r') as f:
            run_id = f.readlines()[-1].strip()  # resume from the last run
    else:
        run_id = wandb.util.generate_id()
        with open(file_path, 'a+') as f:
            f.write(run_id + '\n')
    return run_id

class logging_file(object):
    def __init__(self, path, mode='a+', time_stamp=True, **kwargs):
        self.path = path
        self.mode = mode
        if time_stamp:
            # self.path = self.path + '_' + time.strftime('%Y%m%d_%H%M%S')
            # self.write(f'{time.strftime("%Y%m%d_%H%M%S")}\n')
            self.write(f'{datetime.now()}\n')
    
    def write(self, line_to_print):
        with open(self.path, self.mode) as f:
            f.write(line_to_print)

    def flush(self):
        pass

    def close(self):
        pass

    def __del__(self):
        pass
