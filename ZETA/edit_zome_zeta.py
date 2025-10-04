import pandas as pd
import logging
from util import logger
import argparse
import os
import torch
import numpy as np
import subprocess

import pickle as pkl
import time

import sys
zeta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ZETA/AudioEditingCode/code'))
if zeta_path not in sys.path:
    sys.path.insert(0, zeta_path)

from ZETA.AudioEditingCode.code.load_model import load_zeta
from ZETA.AudioEditingCode.code.main_run_edit import edit_zeta, edit_sdedit
from types import SimpleNamespace


def clean_emphasized(string):
    string = string.strip("()")
    items = string.replace("\"", "").split(",")
    return " ".join(item.strip() for item in items)


def edit_excerpt(df_row, inference_speed, ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device, args_zeta):
    if df_row["download_status"]==True:
        audio_input = str(df_row["audio"])
        audio_input = audio_input.replace("\\", "/")
        df_row["audio"] = audio_input
        tgt_prompt = str(df_row["editing_prompt"])

        wav_name = str(df_row["ytid"])
        
        with torch.no_grad():
            start = time.time()

            args_zeta.init_aud = audio_input
            args_zeta.target_prompt = [tgt_prompt]

            if args_zeta.mode == "sdedit":
                edit_sdedit(ldm_stable, skip, device, args_zeta)
            else:
                edit_zeta(ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device, args_zeta)

            end = time.time()
            inference_speed.append(end-start)

            
        idx = int(df_row.name)
        
        if idx % 25 == 24:
            logger.info(f"----------------CURRENT IDX: {idx}-----------------")


def parse_args():
    parser = argparse.ArgumentParser(description="ZoME Bench editing.") 
    parser.add_argument('--zome_pkl', type=str, default="ZoME/processed_zome.pkl", help='pkl file for dataset')
    parser.add_argument('--model_id', type=str, default="AudioLCM", help='model type in ["AudioLCM","AudioLDM"]')
    parser.add_argument('--config_path', type=str, default="configs/audiolcm.yaml")
    parser.add_argument('--vocoder_path', type=str, default="vocoder")
    parser.add_argument('--model_path', type=str, default="audiolcm.ckpt")
    parser.add_argument('--abs_path', type=str, default='/home/matt3c/projects/def-csubakan-ab/matt3c/AudioLCM', help='absolute path of project folder in remote machine')
    # default : /home/matt3c/projects/def-csubakan-ab/matt3c/AudioLCM c:/Users/Matt/Documents/CODE/MILA/CM_Inversion/AudioLCM/
    parser.add_argument('--eval', action='store_true', help='eval with LPAPS and CLAP scores')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if os.path.isdir(args.abs_path):
        cwd = args.abs_path
        os.chdir(cwd)
        file_handler = logging.FileHandler(os.path.join(cwd,'logs.log'), mode='w')
        logger.addHandler(file_handler)
    else:
        cwd = os.getcwd()
        file_handler = logging.FileHandler(os.path.join(cwd,'logs.log'), mode='w')
        logger.addHandler(file_handler)

    # read ZoME dataset file
    logger.info("Read pkl file")
    zome_df = pd.read_pickle(args.zome_pkl)
    
    if not args.eval:
        logger.info("Loading model")
        args_zeta = SimpleNamespace(
            device_num=0,
            seed=42,
            model_id="cvssp/audioldm-l-full",
            init_aud="",
            cfg_src=[3.0],
            cfg_tar=[12.0],
            num_diffusion_steps=200,
            target_prompt=[""],
            source_prompt=[""],
            target_neg_prompt=[""],
            tstart=[100],
            results_path="ZoME/ZoME_edited_SDEDIT",
            cutoff_points=None,
            mode="sdedit",
            fix_alpha=0.1,
            wandb_name=None,
            wandb_group=None,
            wandb_disable=True,
            eta=1.0,
            numerical_fix=True,
            test_rand_gen=False
        )

        ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device = load_zeta(args_zeta)

        logger.info("Beginning wav edition")
        inference_speed = []
        
        zome_df.apply(lambda row: edit_excerpt(row, inference_speed, ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device, args_zeta), axis=1)

        mean_speed = np.mean(inference_speed)
        min_speed = np.min(inference_speed)
        max_speed = np.max(inference_speed)
        std_speed = np.std(inference_speed)
        with open(f'inference_speed_{args.model_id}.txt', 'w') as f :        
            f.write(f"\n Inference speed : {args.model_id} \n")
            f.write(f"Mean Inference Time: {mean_speed} seconds\n")
            f.write(f"Min Inference Time: {min_speed} seconds\n")
            f.write(f"Max Inference Time: {max_speed} seconds\n")
            f.write(f"Standard Deviation: {std_speed} seconds\n")

    else:
        pass



if __name__ == "__main__":
    main()