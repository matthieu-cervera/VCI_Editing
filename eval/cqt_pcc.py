'''
Top-k Constant Q transform - Pearson correlation coefficient
see https://github.com/KinWaiCheuk/nnAudio
'''
import librosa
import torchaudio
import os
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from nnAudio.features import CQT2010 # version nnAudio==0.3.1

class CQT_PCC_nn:
    def __init__(self, sr):  
        self.sr = sr
        self.args = {
            'n_bins': 128,
            'bins_per_octave': 24, 
        }

        self.CQT = CQT2010(self.sr, **self.args)

    def constant_q_transform(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] != 1: # stereo to mono  (2,wav_len) --> (1,wav_len)
            audio = audio.mean(0,keepdim=True)
        wav = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr)
        cqt = self.CQT(wav)
        return cqt
    
    def get_top1(self, cqt, max_len):
        top1 = np.argmax(cqt[0], axis=0)
        top1 = top1[:max_len]
        return top1

    def score(self, orig_path, edit_path):
        audio_inputs = os.listdir(edit_path)
        cqt_pcc = []
        for audio_name in tqdm(audio_inputs):
            base_audio_name = audio_name.split("_edited")[0] + ".wav"
            base_audio_path = os.path.join(orig_path, base_audio_name)
            edit_audio_path = os.path.join(edit_path, audio_name)

            cqt_base = self.constant_q_transform(base_audio_path)
            
            try:
                cqt_edit = self.constant_q_transform(edit_audio_path)
            except:
                edit_audio_path = edit_audio_path.split(".")[0]+"_edited.wav"
                cqt_edit = self.constant_q_transform(edit_audio_path)
            
            max_len = min(cqt_base.shape[-1], cqt_edit.shape[-1])

            top1_base = self.get_top1(cqt_base, max_len)
            top1_edit = self.get_top1(cqt_edit, max_len)
            pearson = pearsonr(top1_base, top1_edit)
            if pearson.pvalue<=0.05:
                cqt_pcc.append(float(pearson.statistic))

        return cqt_pcc
