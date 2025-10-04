'''
Audiobox-Aesthetics metric 
see https://github.com/facebookresearch/audiobox-aesthetics
'''

from audiobox_aesthetics.infer import initialize_predictor
import torchaudio
import os
from tqdm import tqdm
import numpy as np

class AudioboxAesthetics:

    def __init__(self):
        self.predictor = initialize_predictor()
        self.ae_abs_diff = 0
        
    def compute_ae(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        return self.predictor.forward([{"path":wav, "sample_rate": sr}])


    def score(self, orig_path, edit_path):
        audio_inputs = os.listdir(orig_path)
        audio_edited = os.listdir(edit_path)
        ae = {'CE': [], 'CU': [], 'PC': [], 'PQ': []}
        ae_orig = {'CE': [], 'CU': [], 'PC': [], 'PQ': []}
        ae_abs_diff = {'CE': [], 'CU': [], 'PC': [], 'PQ': []}
        orig = {'CE': 0, 'CU': 0, 'PC': 0, 'PQ': 0}
        result = {'CE': 0, 'CU': 0, 'PC': 0, 'PQ': 0}
        abs_diff = {'CE': 0, 'CU': 0, 'PC': 0, 'PQ': 0}
        for audio_name in tqdm(audio_edited):
            base_audio_name = audio_name.split("_edited")[0] + ".wav"
            base_audio_path = os.path.join(orig_path, base_audio_name)
            edit_audio_path = os.path.join(edit_path, audio_name)

            ae_base = self.compute_ae(base_audio_path)[0]
            
            try:
                ae_edit = self.compute_ae(edit_audio_path)[0]
            except:
                edit_audio_path = edit_audio_path.split(".")[0]+"_edited.wav"
                ae_edit = self.compute_ae(edit_audio_path)[0]
            
            for key in ae.keys():
                ae[key].append(ae_edit[key])
                ae_orig[key].append(ae_base[key])
                ae_abs_diff[key].append(abs(ae_edit[key]-ae_base[key]))
        
        for key in ae.keys():
            result[key] = np.mean(ae[key]).item()
            orig[key] = np.mean(ae_orig[key]).item()
            abs_diff[key] = np.mean(ae_abs_diff[key]).item()

        return orig, result, abs_diff