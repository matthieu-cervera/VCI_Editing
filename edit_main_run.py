import logging
from util import logger, set_reproducability
import argparse
import os
import sys

import torch
import numpy as np
import pickle as pkl
import random
import csv
import time
from omegaconf import OmegaConf
from types import SimpleNamespace
from audio_tools.tools import get_duration


def parse_args():
    parser = argparse.ArgumentParser(description="Edit all audios from given audio set") 
    parser.add_argument('--audio_set', type=str, default="MedleyDB/audio", help='folder with original audios')
    parser.add_argument('--results_dir', type=str, default="MedleyDB/edited", help='folder to write edited audios')
    parser.add_argument('--prompts', type=str, default="MedleyDB/MedleyDB.csv", help='csv file with the prompts and the wav names')
    parser.add_argument('--model_name', type=str, default="VCI_AudioLDM2", help='model type in ["VCI_AudioLCM","VCI_AudioLDM","VCI_AudioLDM2", "ZETA","DDIM","SDEDIT", "MusicGen"]')
    parser.add_argument('--model_id', type=str, default=None, help='precise model_id (e.g. cvssp/audioldm2-music)')
    parser.add_argument('--hparams', type=str, default="hparams_edit.yaml", help='path to YAML file with hyperparameters')
    parser.add_argument('--abs_path', type=str, default="VCI_Editing", help='absolute path of project folder in remote machine')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--comment', type=str, default=None, help='facultative comment on the run or the model')
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if os.path.isdir(args.abs_path):
        cwd = args.abs_path
        os.chdir(cwd)
        file_handler = logging.FileHandler(os.path.join(cwd,'logs_edit.log'), mode='w')
        logger.addHandler(file_handler)
    else:
        cwd = os.getcwd()
        file_handler = logging.FileHandler(os.path.join(cwd,'logs_edit.log'), mode='w')
        logger.addHandler(file_handler)

    if args.seed is not None:
        set_reproducability(args.seed, extreme=False)

    # Load HParams
    hparams = OmegaConf.load(args.hparams)

    # Store prompts 
    prompts = dict()
    with open(args.prompts, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            audio_name = row["filename"]
            audio_path = os.path.join(args.audio_set, audio_name)
            if os.path.isfile(audio_path):
                target_prompt = row.get("prompt", "").strip()
                source_prompt = row.get("source_prompt", "").strip()
                words_target = target_prompt.split(" ")
                words_source = source_prompt.split(" ")
                mutual = " ".join([w for w in words_target if w in words_source])
                local = " ".join([w for w in words_target if w not in words_source])
                nb = row.get("nb")  
                if nb is not None:
                    nb = int(nb)
                else:
                    nb = ""
                audio_name = audio_name + str(nb)
                prompts[audio_name] = source_prompt, target_prompt, local, mutual, nb
            else:
                print(f"[Warning] File not found: {audio_path}")
    
    inference_speed = []

    if args.model_name in ["VCI_AudioLCM", "VCI_AudioLDM", "VCI_AudioLDM2"]:
        hparams = hparams[args.model_name]
        logger.info(f"Loading Model : {args.model_name}")
        from InferenceLCM import load_wrapper, VCI_Edit

        wrapper, device = load_wrapper(args.model_name, args.model_id)

        if hparams.phi is not None:
            phi = torch.Tensor([hparams.phi]).to(wrapper.device)
        else:
            phi = None

        logger.info(f"Beginning editing of {len(prompts)} audios")
        for audio_input, value in prompts.items():
            source_prompt, target_prompt, local, mutual, nb = value

            audio_input = audio_input.split(".wav")[0] + ".wav"


            audio_input = os.path.join(args.audio_set, audio_input)
            wav_name = os.path.basename(audio_input).split('.')[0]


            start = time.time()
            with torch.no_grad():
                audio_edited, _ = VCI_Edit(wrapper, audio_input, source_prompt, target_prompt, local=local, mutual=mutual, wav_name=wav_name, 
                                              result_dir=args.results_dir, nb_consistency_steps=hparams.nb_consistency_steps, save_wav=True, 
                                              phi=phi, attention_control=hparams.attention_control, guidance_scale=hparams.cfg, 
                                              src_guidance_scale=hparams.src_cfg, thresh_e=hparams.thresh_e, thresh_m=hparams.thresh_m, 
                                              tau_s=hparams.tau_s, tau_c=hparams.tau_c, nb=nb)
            end = time.time()
            inference_speed.append(end-start)
    
    elif args.model_name in ["ZETA", "SDEDIT", "DDIM"]:
        directory_zeta = os.path.join(os.getcwd(), "ZETA","AudioEditingCode", "code")
        sys.path.insert(0, directory_zeta)
        zeta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ZETA/AudioEditingCode/code'))
        if zeta_path not in sys.path:
            sys.path.insert(0, zeta_path)
        
        hparams = hparams[args.model_name]

        logger.info(f"Loading Model : {args.model_name}")
        from ZETA.AudioEditingCode.code.load_model import load_zeta
        from ZETA.AudioEditingCode.code.main_run_edit import edit_zeta, edit_sdedit

        if args.model_id is None:
            model_id = "cvssp/audioldm2-music"
        else:
            model_id = args.model_id

        args_zeta = SimpleNamespace(
            device_num=0,
            seed=args.seed,
            model_id=model_id,
            init_aud="",
            cfg_src=[hparams.src_cfg],
            cfg_tar=[hparams.cfg],
            num_diffusion_steps=hparams.num_inference_steps,
            target_prompt=[""],
            source_prompt=[""],
            target_neg_prompt=[""],
            tstart=[hparams.tstart],
            results_path=args.results_dir,
            cutoff_points=None,
            mode=hparams.mode,
            fix_alpha=0.1,
            wandb_name=None,
            wandb_group=None,
            wandb_disable=True,
            eta=1.0,
            numerical_fix=True,
            test_rand_gen=False
        )

        ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device = load_zeta(args_zeta)

        logger.info(f"Beginning editing of {len(prompts)} audios")
        for audio_input, value in prompts.items():
            source_prompt, target_prompt, _, _ , nb = value #prompts[audio_input]   
            audio_input = audio_input.split(".wav")[0] + ".wav"       

            args_zeta.init_aud = os.path.join(args.audio_set, audio_input)
            args_zeta.target_prompt = [target_prompt]
            args_zeta.source_prompt = [source_prompt]

            start = time.time()
            if args_zeta.mode == "sdedit":
                edit_sdedit(ldm_stable, skip, device, args_zeta, nb)
            else:
                edit_zeta(ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device, args_zeta, nb)

            end = time.time()
            inference_speed.append(end-start)

    elif args.model_name in ["MusicGen"]:
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        import torchaudio
            
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



        model = MusicGen.get_pretrained("facebook/musicgen-melody")
        model.set_generation_params(duration=10)
        # model.to(device)

        for audio_input, value in prompts.items():
            source_prompt, target_prompt, local, mutual, nb = value #prompts[audio_input]
            audio_input = audio_input.split(".wav")[0] + ".wav"

            audio_input = os.path.join(args.audio_set, audio_input)
            wav_name = os.path.basename(audio_input).split('.')[0]

            start = time.time()
            
            with torch.no_grad():
                audio, sr = torchaudio.load(audio_input)
                audio_duration = get_duration(audio_input)

                model.set_generation_params(duration=audio_duration)

                if audio.shape[0] != 1: # stereo to mono  (2,wav_len) -> (1,wav_len)
                    audio = audio.mean(0,keepdim=True)
                    audio = audio.to(device)

                audio = audio.unsqueeze(0)  # add batch dimension
                edited_output = model.generate_with_chroma(
                    descriptions=[target_prompt],
                    melody_wavs=audio,
                    melody_sample_rate=sr,
                    progress=True
                )

                audio_write(os.path.join(args.results_dir, wav_name + f"_edited{nb}"),edited_output[0].cpu(), model.sample_rate)
                
            end = time.time()
            inference_speed.append(end-start) 

    else: 
        raise NotImplementedError(f"Error: Model not implemented. You provided {args.model_name} model name but model name should be one of [AudioLCM, AudioLDM, AudioLDM2, ZETA, DDIM, SDEDIT, MusicGen]")


    mean_speed = np.mean(inference_speed)
    min_speed = np.min(inference_speed)
    max_speed = np.max(inference_speed)
    std_speed = np.std(inference_speed)
    with open(f'{args.model_name}.txt', 'a') as f :
        f.write(f"\n================= {args.model_name} execution =================\n")
        if args.comment is not None :
            f.write(f"comment : {args.comment}\n")
        f.write(f"audio saved in {args.results_dir}\n")
        f.write(f"model hparams used : {hparams}\n")
        f.write(f"Inference speed : {args.model_name} \n")
        f.write(f"Mean Inference Time: {mean_speed} seconds\n")
        f.write(f"Min Inference Time: {min_speed} seconds\n")
        f.write(f"Max Inference Time: {max_speed} seconds\n")
        f.write(f"Standard Deviation: {std_speed} seconds\n")



if __name__ == "__main__":
    main()