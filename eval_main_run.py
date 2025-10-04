import logging
from util import logger
import argparse
import os
import torch
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Edited Audio Evaluation") 
    parser.add_argument('--audio_set', type=str, default="MedleyDB/audio", help='folder with original audios')
    parser.add_argument('--eval_set', type=str, default="MedleyDB/edited", help='folder with edited audios')
    parser.add_argument('--prompts', type=str, default="MedleyDB/MedleyDB.csv", help='csv file with the prompts and the wav names')
    parser.add_argument('--model_name', type=str, default="", help='model name (results are written in scores_model_name.txt)')
    parser.add_argument('--long_excerpts', action='store_true', help='flag if you are evaluating excerpts longer than 10s for CLAP model')
    parser.add_argument('--abs_path', type=str, default='VCI_Editing', help='absolute path of project folder in remote machine')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if os.path.isdir(args.abs_path):
        cwd = args.abs_path
        os.chdir(cwd)
        file_handler = logging.FileHandler(os.path.join(cwd,'logs_eval.log'), mode='w')
        logger.addHandler(file_handler)
    else:
        cwd = os.getcwd()
        file_handler = logging.FileHandler(os.path.join(cwd,'logs_eval.log'), mode='w')
        logger.addHandler(file_handler)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Fréchet Audio Distance
    from eval.fad import FrechetAudioDistance

    frechet_clap = FrechetAudioDistance(
            ckpt_dir="eval/",
            model_name="clap",
            sample_rate=48000,
            submodel_name="music_audioset",  # for CLAP only
            verbose=False,
            enable_fusion=False,            # for CLAP only
        )
        
    FAD_clap_score = frechet_clap.score(args.audio_set, args.eval_set)

    # CLAP and LPAPS
    from eval.clap_lpaps import CLAP_LPAPS

    clap_model_name = 'music_audioset_epoch_15_esc_90.14.pt'
    clap_model_path = 'eval/'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    clap_lpaps = CLAP_LPAPS(device, clap_model_path, clap_model_name, long_excerpts=args.long_excerpts)
    clap_scores, lpaps_scores = clap_lpaps.score(args.audio_set, args.eval_set, args.prompts)
    clap_scores = np.array(clap_scores)
    CLAP_score = np.mean(clap_scores)
    LPAPS_score = np.mean(lpaps_scores)

    # MuLan similarity
    from eval.mulan_sim import MuLAN_sim

    mulan_sim = MuLAN_sim()
    mulan_sim_scores = mulan_sim.score(args.eval_set, args.prompts)
    mulan_sim_score = np.mean(mulan_sim_scores)
    
    # TOP 1 CQT-PCC 
    from eval.cqt_pcc import CQT_PCC_nn

    cqt_pcc = CQT_PCC_nn(sr=16000)
    cqt_pcc_scores = cqt_pcc.score(args.audio_set, args.eval_set)
    CQT_PCC_score = np.mean(cqt_pcc_scores)

    # Audio Aesthetics
    from eval.meta_audiobox import AudioboxAesthetics

    audiobox_aesthetics = AudioboxAesthetics()
    original_ae, edited_ae, ae_abs_diff = audiobox_aesthetics.score(args.audio_set, args.eval_set)
    audiobox_ae = sum(ae_abs_diff.values())

    # PRINT & SAVE SCORES
    print(f"\n SCORES : {args.eval_set} \n")
    print(f"nb of evaluated excerpts : {len(clap_scores)}\n")  
    print(f"==============Edited scores==============\nLPAPS : {LPAPS_score}\nCLAP : {CLAP_score}\nFAD (clap) : {FAD_clap_score}\nMuLAN : {mulan_sim_score}\nCQT-PCC : {CQT_PCC_score}")
    print(f"Audiobox-AE : {audiobox_ae}")
    print(f"\nAudiobox-AE details : original {original_ae}\n edited {edited_ae} \n abs difference {ae_abs_diff}")

    with open(f'scores_{args.model_name}.txt', 'w') as f :        
        f.write(f"\n SCORES : {args.eval_set} \n")
        f.write(f"nb of evaluated excerpts : {len(clap_scores)}\n")  
        f.write(f"==============Edited scores==============\nLPAPS : {LPAPS_score}\nCLAP : {CLAP_score}\nFAD (clap) : {FAD_clap_score}\nMuLAN : {mulan_sim_score}\nCQT-PCC : {CQT_PCC_score}")
        f.write(f"Audiobox-AE : {audiobox_ae}")
        f.write(f"\nAudiobox-AE details : original {original_ae}\n edited {edited_ae} \n abs difference {ae_abs_diff}")


if __name__ == "__main__":
    main()