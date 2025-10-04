import argparse
import calendar
import matplotlib.pyplot as plt
import os
import time
import torch
import torchaudio
import warnings
import wandb
from torch import inference_mode

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from models import load_model
from utils import set_reproducability, load_audio, get_spec


HF_TOKEN = None  # Needed for stable audio open. You can leave None when not using it


def parse_args():
    parser = argparse.ArgumentParser(description='Run text-based audio editing.')
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=None, help="GPU device number")
    parser.add_argument("--model_id", type=str, choices=["cvssp/audioldm-s-full-v2",
                                                         "cvssp/audioldm-l-full",
                                                         "cvssp/audioldm2",
                                                         "cvssp/audioldm2-large",
                                                         "cvssp/audioldm2-music",
                                                         'declare-lab/tango-full-ft-audio-music-caps',
                                                         'declare-lab/tango-full-ft-audiocaps',
                                                         "stabilityai/stable-audio-open-1.0"
                                                         ],
                        default="cvssp/audioldm2-music", help='Audio diffusion model to use')

    parser.add_argument("--init_aud", type=str, help='Audio to invert and extract PCs from')
    parser.add_argument("--cfg_src", type=float, nargs='+', default=[3],
                        help='Classifier-free guidance strength for forward process')
    parser.add_argument("--cfg_tar", type=float, nargs='+', default=[12],
                        help='Classifier-free guidance strength for reverse process')
    parser.add_argument("--num_diffusion_steps", type=int, default=200,
                        help="Number of diffusion steps. TANGO and AudioLDM2 are recommended to be used with 200 steps"
                             ", while AudioLDM is recommeneded to be used with 100 steps")
    parser.add_argument("--target_prompt", type=str, nargs='+', default=[""],
                        help="Prompt to accompany the reverse process. Should describe the wanted edited audio.")
    parser.add_argument("--source_prompt", type=str, nargs='+', default=[""],
                        help="Prompt to accompany the forward process. Should describe the original audio.")
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help="Negative prompt to accompany the inversion and generation process")
    parser.add_argument("--tstart", type=int, nargs='+', default=[100],
                        help="Diffusion timestep to start the reverse process from. Controls editing strength.")
    parser.add_argument("--results_path", type=str, default="results", help="path to dump results")

    parser.add_argument("--cutoff_points", type=float, nargs='*', default=None)
    parser.add_argument("--mode", default="ours", choices=['ours', 'ddim'],
                        help="Run our editing or DDIM inversion based editing.")
    parser.add_argument("--fix_alpha", type=float, default=0.1)

    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true', default=True)

    args = parser.parse_args()
    args.eta = 1.
    args.numerical_fix = True
    args.test_rand_gen = False
    return args

def load_zeta(args):
    if args.model_id == "stabilityai/stable-audio-open-1.0" and HF_TOKEN is None:
        raise ValueError("HF_TOKEN is required for stable audio model")

    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    model_id = args.model_id
    cfg_scale_src = args.cfg_src
    cfg_scale_tar = args.cfg_tar

    # same output
    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    if args.mode == 'ours':
        image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
            f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
            f'skip_{int(args.num_diffusion_steps) - int(args.tstart[0])}_{time_stamp_name}'
    else:
        if args.tstart != args.num_diffusion_steps:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'skip_{int(args.num_diffusion_steps) - int(args.tstart[0])}_{time_stamp_name}'
        else:
            image_name_png = f'cfg_e_{"-".join([str(x) for x in cfg_scale_src])}_' + \
                f'cfg_d_{"-".join([str(x) for x in cfg_scale_tar])}_' + \
                f'{args.num_diffusion_steps}timesteps_{time_stamp_name}'

    wandb.login(key='')
    wandb_run = wandb.init(project="AudInv", entity='', config={},
                           name=args.wandb_name if args.wandb_name is not None else image_name_png,
                           group=args.wandb_group,
                           mode='disabled' if args.wandb_disable else 'online',
                           settings=wandb.Settings(_disable_stats=True))
    wandb.config.update(args)

    eta = args.eta  # = 1
    if len(args.tstart) != len(args.target_prompt):
        if len(args.tstart) == 1:
            args.tstart *= len(args.target_prompt)
        else:
            raise ValueError("T-start amount and target prompt amount don't match.")
    args.tstart = torch.tensor(args.tstart, dtype=torch.int)
    skip = args.num_diffusion_steps - args.tstart

    ldm_stable = load_model(model_id, device, args.num_diffusion_steps, token=HF_TOKEN)


    return(ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device)