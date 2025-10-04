'''
Cosine Similarity computed on MuLan embeddings
see https://github.com/tencent-ailab/MuQ
'''

from muq import MuQMuLan
import torch
import os
import torchaudio
from tqdm import tqdm 
import csv

class MuLAN_sim:
    def __init__(self):
        self.mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").eval()
        self.mulan_sr = 24000
        self.prompts = {}

    def load_audio(self, path):
        audio, sr = torchaudio.load(path)

        if sr != self.mulan_sr:
            audio = torchaudio.functional.resample(audio, sr, self.mulan_sr)

        if audio.shape[0] != 1: # stereo to mono  (2,wav_len) -> (1,wav_len)
            audio = audio.mean(0,keepdim=True)

        return audio
    
    def handle_prompts(self, audio_dir, prompt_path, audio_column_name='filename', prompt_column_name="prompt"):
        '''
        Fill the self.prompts dict to make prompts easier to access.
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



    def score(self, edit_path, prompts):
        self.handle_prompts(edit_path, prompts)
        audios= os.listdir(edit_path)
        cosine_sims = []
        for audio_name in tqdm(audios):
            audio_path = os.path.join(edit_path, audio_name)
            audio = self.load_audio(audio_path)

            target_text = self.prompts[audio_name]

            with torch.no_grad():
                audio_embed = self.mulan(wavs = audio) 
                text_embed = self.mulan(texts = target_text)

            sim = self.mulan.calc_similarity(audio_embed, text_embed)
            cosine_sims.append(sim.item())

        return cosine_sims
