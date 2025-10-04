import argparse, os, sys, glob
import pathlib
directory = pathlib.Path(os.getcwd())
print(directory)
sys.path.append(str(directory))
import torch
from ldm.models.diffusion.scheduling_lcs import LCMSampler as lcmsampler
import soundfile
from models import AudioLCMWrapper, AudioLDMWrapper, AudioLDM2Wrapper
from util import logger

class GenSamples:
    def __init__(self, sampler, wrapper, outpath, save_mel = False, save_wav = True, original_inference_steps=None) -> None:
        self.sampler = sampler
        self.wrapper = wrapper
        self.outpath = outpath
        self.save_mel = save_mel
        self.save_wav = save_wav
        self.original_inference_steps = original_inference_steps

    def gen_edit(self, audio_input, source_prompt, target_prompt, nb_consistency_steps=8, wav_name = None):
        
        if self.wrapper.model_name == "AudioLDM2":
            emb_source = self.wrapper.encode_text([source_prompt])
            emb_target = self.wrapper.encode_text([target_prompt])
            c_source = emb_source.embedding_class_labels
            c_target = emb_target.embedding_class_labels

        elif self.wrapper.model_name == "AudioLDM":
            emb_source = None
            emb_target = None
            c_source, _, _ = self.wrapper.encode_text_improved([source_prompt])
            c_target, _, _ = self.wrapper.encode_text_improved([target_prompt])

        else:
            emb_source = None
            emb_target = None
            c_source= self.wrapper.encode_text(source_prompt)
            c_target= self.wrapper.encode_text(target_prompt)

        # encoding audio input
        x_0, shape = self.wrapper.vae_encode(audio_input)

        x_source_t, x_target_t, x_target_0_pred = self.sampler.VC_sampling_edit(S=nb_consistency_steps,
                                            x_0=x_0,  
                                            source_cond=c_source,
                                            target_cond=c_target,
                                            x_T=None,
                                            batch_size=1,
                                            shape=shape,
                                            verbose=False,
                                            emb_source=emb_source,      # Useful for AudioLDM2 cond
                                            emb_target=emb_target       # Useful for AudioLDM2 cond
                                            )

        
        edited_spec = self.wrapper.vae_decode(x_target_0_pred)
        target_spec = self.wrapper.vae_decode(x_target_t)

        if self.save_wav:
            wav = self.wrapper.mel_to_wav(edited_spec)
            wav_path = os.path.join(self.outpath,wav_name)
            soundfile.write(wav_path, wav, 16000)
            
        else:
            wav = self.wrapper.mel_to_wav_grad(edited_spec) # preserving gradient
            wav_path = None 

        return wav, wav_path



def load_wrapper(model_type, model_id=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"computing on {device}")

    if model_type == "VCI_AudioLCM":
        wrapper = AudioLCMWrapper("AudioLCM", device)
        return wrapper, device
    
    elif model_type == "VCI_AudioLDM2":
        if model_id is not None:
            wrapper = AudioLDM2Wrapper(model_id, device)
        else:
            wrapper = AudioLDM2Wrapper("cvssp/audioldm2-music", device)
        return wrapper, device
    
    elif model_type == "VCI_AudioLDM":
        if model_id is not None:
            wrapper = AudioLDMWrapper(model_id, device)
        else:
            wrapper = AudioLDMWrapper("cvssp/audioldm-s-full-v2", device)
        return wrapper, device
    
    else:
        raise ValueError("Model type not supported. Please choose a model type in VCI_AudioLCM, VCI_AudioLDM, VCI_AudioLDM2.")


def VCI_Edit(wrapper, input_path, source_prompt, target_prompt, local=None, mutual=None, wav_name="test",
             result_dir = "results", nb_consistency_steps=8, save_wav=True, phi=None,
                guidance_scale=1.0, src_guidance_scale=1.0, attention_control=False,
             thresh_e=0.3, thresh_m=0.5, tau_s=0.7, tau_c=0.7, clap=None, nb=""):
  
    wrapper.register_schedule()

    input_audio, Tshape = wrapper.wav_to_mel(input_path)

    if attention_control:
        controller, local_blend = wrapper.set_attention_control(input_audio, Tshape, source_prompt, target_prompt, local, mutual, nb_consistency_steps, 
                                                                thresh_e=thresh_e, thresh_m=thresh_m, tau_s=tau_s, tau_c=tau_c) 

    else:
        print("-------No controller---------")
        controller = None
        local_blend = None

    sampler = lcmsampler(wrapper, controller, store_latents=False, phi=phi, guidance_scale=guidance_scale, 
                         src_guidance_scale=src_guidance_scale, local_blend=local_blend, clap=clap)

    os.makedirs(result_dir, exist_ok=True)

    generator = GenSamples(sampler, wrapper, result_dir, save_mel = False, save_wav = save_wav, 
                           original_inference_steps=wrapper.original_inference_steps)

    if wrapper.model_id == "AudioLCM":
        with wrapper.model.ema_scope():
            edited, edited_path = generator.gen_edit(input_audio, source_prompt, target_prompt, nb_consistency_steps=nb_consistency_steps, 
                                                                 wav_name=wav_name+f"_edited{nb}.wav")
    else:
        edited, edited_path = generator.gen_edit(input_audio, source_prompt, target_prompt, nb_consistency_steps=nb_consistency_steps, 
                                                             wav_name=wav_name+f"_edited{nb}.wav")

        
    print(f"Edited sample ready")

    return edited, os.path.join(result_dir,wav_name+f"_edited{nb}.wav")

    

                

