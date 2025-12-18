# Virtual Consistency for Audio Editing

#### Matthieu Cervera, Francesco Paissan, Mirco Ravanelli, Cem Subakan

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2509.17219)

We provide the implementation of *Virtual Consistency for Audio Editing* in this repository. Visit our [demo page](https://matthieu-cervera-9e056d.gitlab.io/vci_editing) to listen to edited audio examples.

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Multiple Edits](#multiple-edits)
- [Evaluation](#evaluation)
- [Supplementary Information](#supplementary-information)
- [Acknowledgements](#acknowledgements)

## Requirements
### Virtual Consistency
```bash
python -m pip install -r requirements.txt
```
### Reference Methods
```bash
python -m pip install -r ZETA/AudioEditingCode/requirements.txt
python -m pip install audiocraft
```

## Quick Start
We provide an example of how you can quickly edit an audio sample:

```python
import torch
from InferenceLCM import VCI_Edit, load_wrapper
from util import set_reproducability

set_reproducability(42) # results quality can be very seed dependant

source_prompt="a recording of piano music"
target_prompt="a recording of trumpet music"
audio_input="piano_source.wav"

model_type = "VCI_AudioLDM2"

wrapper, device = load_wrapper(model_type)

with torch.no_grad():
    audio_edited, edited_path = VCI_Edit(wrapper, audio_input, source_prompt, target_prompt, 
                                        local="", mutual="", wav_name="trumpet", 
                                        nb_consistency_steps=20, result_dir="edited", 
                                        phi=torch.Tensor([0.82]).to(device), guidance_scale=20.0)  
```

## Multiple Edits

Use the `edit_main_run.py` script to generate multiple audio edits given a batch of text prompts. Prompts should be in a .csv file with 3 or 4 columns `{filename; prompt; source_prompt; nb}`. `nb` should be used if you aim to edit the same audio file with multiple prompts.

```bash
python edit_main_run.py --audio_set="MedleyDB/audio" 
                        --results_dir="MedleyDB/edited" 
                        --prompts="MedleyDB/MedleyDB.csv" 
                        --model_name="VCI_AudioLDM2" 
                        --hparams="hparams_edit.yaml 
```

## Evaluation

For CLAP and LPAPS, you need to download the checkpoint `music_audioset_epoch_15_esc_90.14.pt` ([HugginFace](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt)) and put it in the `./eval/` folder.

To compute the scores of the edited excerpts, use the `eval_main_run.py` script :
```bash
python eval_main_run.py --audio_set="MedleyDB/audio" 
                        --eval_set="MedleyDB/edited" 
                        --prompts="MedleyDB/MedleyDB.csv" 
                        --model_name="VCI"
                        --long_excerpts
```

You should use the `--long_excerpts` argument if you're evaluating audios that are >10s.

## Supplementary Information

You can run VCI using *AudioLDM2*, *AudioLDM* and *AudioLCM*. We recommend using *AudioLDM2*. <br>
<sub>For *AudioLCM*, you need to download the weights from [Huggingface](https://huggingface.co/liuhuadai/AudioLCM) (`audiolcm.ckpt`, `epoch=000032.ckpt`, `maa2.ckpt`, `BigVGAN vocoder`, `t5-v1_1-large`, `bert-base-uncased`, `CLAP_weights_2022.pth`).</sub>


## Acknowledgements
This implementation is heavily based or inspired by code from [AudioLCM](https://github.com/Text-to-Audio/AudioLCM), [InfEdit](https://github.com/sled-group/InfEdit), [ZETA](https://github.com/HilaManor/AudioEditingCode) and [AudioLDM2](https://github.com/haoheliu/AudioLDM2). We'd like to thank them for their contributions.

### Datasets
Thanks to previous work (Liu et al. ; Bittner et al. ; Manor et al. ) , we used [ZoME Bench](https://huggingface.co/datasets/liuhuadai/ZoME-Bench) dataset(Academic Free License v3.0) and [MedleyDB](https://medleydb.weebly.com/), [MedleyMDprompts](https://github.com/HilaManor/AudioEditingCode/tree/codeclean/MedleyMDPrompts) (Attribution 4.0 International License) audios and prompts.


### Eval
The eval part uses code from the following Github repos:
[CLAP](https://github.com/LAION-AI/CLAP) (CC0 1.0 Universal License),
[facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) (MIT License),
[Audiobox](https://github.com/facebookresearch/audiobox-aesthetics) (Attribution 4.0 International License),
[FAD](https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py) (MIT License),
[MuLanQ](https://github.com/tencent-ailab/MuQ) (MIT License),
[CQT - nnAudio](https://github.com/KinWaiCheuk/nnAudio) (MIT License).
