'''
inspired by https://github.com/HilaManor/AudioEditingCode/blob/codeclean/evals/utils.py
utilitary functions for both CLAP & LPAPS (w clap emb) scores
'''

import numpy as np
import torch
import os
import torchaudio
from typing import Optional
from tqdm import tqdm
from eval.meta_clap_consistency import CLAPTextConsistencyMetric
from eval.lpaps import LPAPS
import csv



def compute_lpaps_with_windows(aud1: torch.Tensor, aud1_sr: int, aud2: torch.Tensor, aud2_sr: int, model: LPAPS,
                               windows_size1: Optional[int] = None, windows_size2: Optional[int] = None,
                               overlap: float = 0.1, method: str = 'mean', device: str = 'cuda:0') -> float:
    """Calculate the LPAPS score for the given audio files, windowed. If windows_size1 or windows_size2 is None, it will default to 10 seconds.

    :param torch.Tensor aud1: The first audio file to compute LPAPS for
    :param int aud1_sr: The sample rate of the first audio file
    :param torch.Tensor aud2: The second audio file to compute LPAPS for
    :param int aud2_sr: The sample rate of the second audio file
    :param LPAPS model: The LPAPS model to use
    :param Optional[int] windows_size1: Window size in seconds for the first audio file. Defaults to 10 seconds (None)
    :param Optional[int] windows_size2: Window size in seconds for the second audio file. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined LPAPS score
    """

    if windows_size1 is None:
        windows_size1 = int(aud1_sr * 10)
    if windows_size2 is None:
        windows_size2 = int(aud2_sr * 10)

    scores = []
    for i, j in zip(range(0, aud1.shape[-1], int(windows_size1 * (1 - overlap))),
                    range(0, aud2.shape[-1], int(windows_size2 * (1 - overlap)))):
        window1 = aud1[:, i:i + windows_size1]
        window2 = aud2[:, j:j + windows_size2]
        scores.append(model(window1.unsqueeze(0).to(device), window2.unsqueeze(0).to(device),
                      torch.tensor([aud1_sr], device=device),
                      torch.tensor([aud2_sr], device=device)).item())

    if method == 'mean':
        func = np.mean
    elif method == 'median':
        func = np.median
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    else:
        raise ValueError(f'Unknown method: {method}')
    return func(scores)

def compute_clap_with_windows(aud: torch.Tensor, aud_sr: int, prompt: str, model: CLAPTextConsistencyMetric,
                              windows_size: Optional[int] = None, overlap: float = 0.1,
                              method: str = 'mean', device: torch.device = 'cuda:0') -> float:
    """Calculate the CLAP score for the given audio file and prompt, windowed. If windows_size is None, it will default to 10 seconds. 

    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int aud_sr: The sample rate of the audio file
    :param str prompt: The prompt to compute CLAP relative to
    :param CLAPTextConsistencyMetric model: The CLAP model to use
    :param Optional[int] windows_size: Window size in seconds. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined CLAP score
    """
    if windows_size is None:
        windows_size = int(aud_sr * 10)
    scores = []
    for i in range(0, aud.shape[-1], int(windows_size * (1 - overlap))):
        window = aud[:, i:i + windows_size]
        model.update(window.unsqueeze(0).to(device), [prompt], torch.tensor([aud_sr], device=device))
        scores.append(model.compute())
        model.reset()
    if method == 'mean':
        func = np.mean
    elif method == 'median':
        func = np.median
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    else:
        raise ValueError(f'Unknown method: {method}')
    return func(scores)

def calc_clap_win(clap_model: CLAPTextConsistencyMetric, aud: torch.Tensor, sr: int, target_prompt: str,
                  win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the CLAP score between an audio file and a prompt, with optional windowing

    :param CLAPTextConsistencyMetric clap_model: An initialized CLAP model to use
    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int sr: The sample rate of the audio file
    :param str target_prompt: The prompt to compute CLAP relative to
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The CLAP score
    """
    if win_length is None:
        clap_model.update(aud.unsqueeze(0).to(device), [target_prompt], torch.tensor([sr], device=device))
        tmp = clap_model.compute()
        clap_model.reset()
        return tmp
    else:
        return compute_clap_with_windows(
            aud, sr, target_prompt, clap_model, device=device,
            windows_size=win_length * sr, overlap=overlap, method=method)

def calc_lpaps_win(lpaps_model: LPAPS, aud1: torch.Tensor, aud2: torch.Tensor, sr1: int, sr2: int,
                   win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the LPAPS score between two audio files, with optional windowing

    :param LPAPS lpaps_model: An initialized LPAPS model to use
    :param torch.Tensor aud1: First audio file
    :param torch.Tensor aud2: Second audio file
    :param int sr1: Sample rate of the first audio file
    :param int sr2: Sample rate of the second audio file
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The LPAPS score
    """
    if win_length is None:
        return lpaps_model(aud1.unsqueeze(0).to(device),
                           aud2.unsqueeze(0).to(device),
                           torch.tensor([sr1], device=device),
                           torch.tensor([sr2], device=device)).item()
    else:
        return compute_lpaps_with_windows(aud1, sr1, aud2, sr2, lpaps_model,
                                          windows_size1=win_length * sr1,
                                          windows_size2=win_length * sr2,
                                          overlap=overlap, method=method, device=device)


def load_eval_models(device, clap_checkpoint_path, clap_ckpt_name="music_audioset_epoch_15_esc_90.14.pt", req_grad=False):

    clap_model = CLAPTextConsistencyMetric(model_path=os.path.join(clap_checkpoint_path, clap_ckpt_name),
                                            model_arch='HTSAT-base' if 'fusion' not in clap_ckpt_name else 'HTSAT-tiny',
                                            enable_fusion='fusion' in clap_ckpt_name
                                            ).to(device)

    clap_model.eval()

    lpaps_model = LPAPS(net='clap', device=device,
                    net_kwargs={'model_arch': 'HTSAT-base' if 'fusion' not in clap_ckpt_name
                                else 'HTSAT-tiny',
                                'chkpt': clap_ckpt_name,
                                'enable_fusion': 'fusion' in clap_ckpt_name},
                    checkpoint_path=clap_checkpoint_path, req_grad=req_grad)
    
    lpaps_model.eval()   
    return clap_model, lpaps_model

class CLAP_LPAPS:

    def __init__(self, device, clap_model_path = "eval", clap_model_name = 'music_audioset_epoch_15_esc_90.14.pt', long_excerpts=False, req_grad=False):
        '''
        long_excerpts = True when audio excerpts are >10s
        when False, we should crop the audios resampled 48kHz for them to be less than 10s (sometimes the audios from AudioCaps are just a little more than 10s)
        '''
        self.clap_model, self.lpaps_model = load_eval_models(device, clap_model_path, clap_model_name, req_grad=req_grad)
        self.prompts = {}
        self.device = device
        self.target_prompt = ""
        self.long_excerpts = long_excerpts

    def handle_prompts(self, audio_dir, prompt_path, audio_column_name='filename', prompt_column_name="prompt"):
        '''
        Fill the self.prompts dict to make prompts easier to access
        audio_dir : the directory of the base audios
        prompt_path : the path of the csv file containing the prompts and the audio filenames
        {x}_column_name : the column name of x in the csv file
        '''
        with open(prompt_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                audio_name = row[audio_column_name]
                nb = row.get("nb")  
                if nb is not None:
                    nb = int(nb)
                else:
                    nb = ""
                audio_name = audio_name.split(".wav")[0] + "_edited" + str(nb) + ".wav"

                audio_path = os.path.join(audio_dir, audio_name)
                if os.path.isfile(audio_path):
                    prompt = row.get(prompt_column_name, "").strip()
                    
                    self.prompts[audio_name] = prompt
                else:
                    print(f"[Warning] File not found: {audio_path}")



    def score(self, orig_path, edit_path, tgt_prompts_path):

        # tgt prompt at self.prompts[audio]
        self.handle_prompts(edit_path, tgt_prompts_path)
        
        audio_edited = os.listdir(edit_path)
        lpaps_scores = []
        clap_scores = []


        for audio_name in tqdm(audio_edited):
            base_audio_name = audio_name.split("_edited")[0] + ".wav"
            base_audio_path = os.path.join(orig_path, base_audio_name)
            edit_audio_path = os.path.join(edit_path, audio_name)

            base_audio, base_sr = torchaudio.load(base_audio_path)
            try:
                edit_audio, edit_sr = torchaudio.load(edit_audio_path)
            except:
                edit_audio_path = edit_audio_path.split(".")[0]+"_edited.wav"
                edit_audio, edit_sr = torchaudio.load(edit_audio_path)

            if not self.long_excerpts:  # resample to 48kHz and ensure that audios are <10s
                base_audio = torchaudio.functional.resample(base_audio, base_sr, 48000)
                base_audio = base_audio[:,:480000]
                edit_audio = torchaudio.functional.resample(edit_audio, edit_sr, 48000)
                edit_audio = edit_audio[:,:480000]
                win = None
            else:
                base_audio = torchaudio.functional.resample(base_audio, base_sr, 48000)
                edit_audio = torchaudio.functional.resample(edit_audio, edit_sr, 48000)
                win = 10

            lpaps_scores.append(calc_lpaps_win(self.lpaps_model, base_audio, edit_audio, 48000, 48000, win, 'mean', 0.1, self.device)) 
            clap_scores.append(calc_clap_win(self.clap_model, edit_audio, 48000, self.prompts[audio_name], win, 'mean', 0.1, self.device))


        return clap_scores, lpaps_scores
