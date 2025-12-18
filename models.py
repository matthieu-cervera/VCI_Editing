import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from functools import partial
from audio_tools.stft import TacotronSTFT
from audio_tools.tools import wav_to_fbank, get_duration
from audio_tools.NAT_Mel import MelNet, load_mono_audio
from util import logger
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import make_beta_schedule
# from attention_control import AttentionControlLCM, AttentionControlLDM, AttentionControlLDM2
# import seq_aligner
import torchaudio

class PromptEmbeddings(NamedTuple):
    embedding_hidden_states: torch.Tensor
    embedding_class_labels: torch.Tensor
    boolean_prompt_mask: torch.Tensor


def load_model_from_config(config, ckpt = None, verbose=True):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = pl_sd["state_dict"]
        
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        print(f"Note chat no ckpt is loaded !!!")

    model.eval()
    return model

def load_consistency_model(config_path = "configs/audiolcm.yaml", model_path = "./model/000184.ckpt"):
    '''
        Load the AudioLCM model.

        :param str config_path: yaml file with configs for all models that compose AudioLCM

    '''
    config = OmegaConf.load(config_path)

    print("-------loading model---------")
    logger.info("-------loading model---------")
    model = load_model_from_config(config, model_path)
    print("-------model loaded---------")
    logger.info("-------model loaded---------")

    return model



class ModelWrapper(torch.nn.Module):
    def __init__(self, model_id: str,
                 device: torch.device,
                 double_precision: bool = False,
                 token: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.device = device
        self.double_precision = double_precision
        self.token = token

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        '''
        save schedule values into Wrapper class
        '''
        
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

    def get_fn_STFT(self) -> torch.nn.Module:
        pass

    def get_sr(self) -> int:
        return 16000

    def vae_encode(self, x):
        pass

    def vae_decode(self, x):
        pass

    def mel_to_wav(self, spec):
        pass

    def encode_text(self, prompt):
        pass

    def unet_forward(self, x, t, c, w):
        pass

    def set_attention_control(self, input_audio, Tshape, source_prompt, target_prompt, local, mutual, nb_consistency_steps, 
                              thresh_e, thresh_m, tau_s, tau_c, compute_second_mask=False):
        pass


class AudioLCMWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from vocoder.bigvgan.models import VocoderBigVGAN
        config_path="configs/audiolcm.yaml"
        model_path="audiolcm.ckpt"
        vocoder_path="vocoder"

        self.model_name = "AudioLCM"

        self.model = load_consistency_model(config_path=config_path, model_path=model_path).to(self.device)
        self.vocoder = VocoderBigVGAN(vocoder_path, self.device)
        self.original_inference_steps = 50

    def get_fn_STFT(self) -> torch.nn.Module:
        hparams = {
        'fft_size': 1024,
        'audio_num_mel_bins': 80,
        'audio_sample_rate': 16000,
        'hop_size': 256,
        'win_size': 1024,
        'fmin': 0,
        'fmax': 8000
        }   
        return MelNet(hparams)
    
    def wav_to_mel(self, audio_path):
        fn_stft = self.get_fn_STFT()
        wav, sr = load_mono_audio(audio_path)
        mel_audio = fn_stft(wav).to(self.device)
        T_shape = mel_audio.shape[2]//2

        return mel_audio, T_shape

    def vae_encode(self, x):
        T_shape = x.shape[2]//2
        x_0 = self.model.encode_first_stage(x)
        x_0 = self.model.get_first_stage_encoding(x_0).detach()
        
        if self.model.channels>0:
            shape = [self.model.channels, 20, T_shape]
        else:
            shape = [20, T_shape]

        return x_0, shape

    def vae_decode(self, x):
        return self.model.decode_first_stage(x)

    def mel_to_wav(self, spec):
        wav = self.vocoder.vocode(spec)    
        return wav

    def encode_text(self, prompt):
        prompt = dict(ori_caption=prompt,struct_caption=f'<{prompt}& all>')
        for k,v in prompt.items():
            prompt[k] = 1 * [v]
        e_prompt = self.model.get_learned_conditioning(prompt)

        return(e_prompt)
    
    def unet_forward(self, x, t, c, w):
        return self.model.apply_model(x, t, c, self.model.unet, w_cond=w)
    
    # def set_attention_control(self, input_audio, Tshape, source_prompt, target_prompt, local, mutual, nb_consistency_steps, 
    #                           thresh_e, thresh_m, tau_s, tau_c, compute_second_mask=False):
        
    #     text_encoder = self.model.cond_stage_model.t5_transformer
    #     text_tokenizer = self.model.cond_stage_model.t5_tokenizer
    #     clap_encoder = self.model.cond_stage_model.caption_encoder
    #     clap_tokenizer = self.model.cond_stage_model.clap_tokenizer

    #     source_prompt = dict(ori_caption=source_prompt,struct_caption=f'<{source_prompt}& all>')
    #     target_prompt = dict(ori_caption=target_prompt,struct_caption=f'<{target_prompt}& all>')


    #     try:
    #         mapper,alphas, ms, alpha_e, alpha_m= seq_aligner.get_refinement_mapper([source_prompt["ori_caption"], target_prompt["ori_caption"]],
    #                                                                             [[local, mutual]], text_tokenizer, text_encoder, self.device)
            
    #         mapper_clap,alphas_clap, ms_clap, alpha_e_clap, alpha_m_clap= seq_aligner.get_refinement_mapper([source_prompt["ori_caption"], target_prompt["ori_caption"]],
    #                                                                             [[local, mutual]], clap_tokenizer, clap_encoder, self.device, encoder_type="CLAP")

    #         w_blended_src = torch.cat([alpha_m, alpha_m_clap], dim=1).squeeze(0)
    #         w_blended_tgt = torch.cat([alpha_e, alpha_e_clap], dim=1).squeeze(0)
    #         mapper = torch.cat([mapper, mapper_clap + self.model.cond_stage_model.max_length], dim=1)
    #         alignment_fn = mapper.squeeze(0).tolist()
    #         local_blend = AttentionControlLCM.LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, compute_second_mask=compute_second_mask, save_inter=False)
    #         alphas = torch.cat([alphas, alphas_clap], dim=1).squeeze(0)
    #         ms = torch.cat([ms, ms_clap], dim=1).squeeze(0)
    #         local_blend.set_map(ms,alphas,w_blended_tgt,w_blended_src,154)  

    #     except Exception as e:
    #         logger.info(f"Exception during blended words computing : {e}")
    #         alignment_fn = [i for i in range(154)]
    #         w_blended_src = torch.zeros((154))
    #         w_blended_tgt = torch.zeros((154))
    #         local_blend = None 

    #     print("-------Building attention controller---------")
    #     controller = AttentionControlLCM.AttentionManipulator(input_audio, tau_s=tau_s, tau_c= tau_c, total_steps=nb_consistency_steps, alignment_fn=alignment_fn, 
    #                                       T_shape=Tshape, w_blended_src=w_blended_src, w_blended_tgt=w_blended_tgt)
        
    #     print("-------passing controller into unet---------")
    #     AttentionControlLCM.register_attention_control(self.model, controller)

    #     return controller, local_blend


class AudioLDMWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from diffusers import AudioLDMPipeline
        self.model = AudioLDMPipeline.from_pretrained(self.model_id, token=self.token).to(self.device)
        self.model_name = "AudioLDM"
        self.original_inference_steps = 200

    def get_fn_STFT(self) -> torch.nn.Module:
        return TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )
    
    def wav_to_mel(self, audio_path):
        fn_stft = self.get_fn_STFT()
        duration = get_duration(audio_path)
        mel_audio, _, _ = wav_to_fbank(audio_path, fn_STFT=fn_stft, target_length=int(duration * 102.4)) 
        mel_audio = mel_audio.unsqueeze(0)
        T_shape = mel_audio.shape[1]//2

        return mel_audio, T_shape

    def vae_encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = x.to(self.device)
        x_0 = self.model.vae.encode(x).latent_dist.mode()
        shape = [x_0.shape[i] for i in range(1,4)]

        return (x_0 * self.model.vae.config.scaling_factor).float(), shape

    def vae_decode(self, x):
        return self.model.vae.decode(1 / self.model.vae.config.scaling_factor * x).sample.squeeze(1)

    def mel_to_wav(self, spec):
        spec = spec.squeeze(0)
        wav = self.model.vocoder(spec).detach().cpu()
        
        return wav
    
    def mel_to_wav_grad(self, spec):
        spec = spec.squeeze(0)
        wav = self.model.vocoder(spec)
        
        return wav
    
    def encode_text_2(self, prompts: List[str], **kwargs) -> Tuple[None, Optional[torch.Tensor], None]:
        text_input = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_input.input_ids
        attention_mask = text_input.attention_mask
        untruncated_ids = self.model.tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] \
                and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.model.tokenizer.batch_decode(
                untruncated_ids[:, self.model.tokenizer.model_max_length - 1: -1])
            print("The following part of your input was truncated because CLAP can only handle sequences up to"
                  f" {self.model.tokenizer.model_max_length} tokens: {removed_text}")

        with torch.no_grad():
            text_encoding = self.model.text_encoder(text_input.input_ids.to(self.device),
                                                    attention_mask=attention_mask.to(self.device))[0]
        text_encoding = F.normalize(text_encoding, dim=-1)
        text_encoding = text_encoding.to(dtype=self.model.text_encoder.dtype, device=self.device)

        return None, text_encoding, None

    def encode_text(self, prompts):
        tokenized = self.model.tokenizer(prompts, padding="max_length", max_length=self.model.tokenizer.model_max_length,
            truncation=True, return_tensors="pt").to(self.device)
        e_prompt = self.model.text_encoder(**tokenized).text_embeds

        return(e_prompt)
    
    def encode_text_improved(self, target_prompt):
        text_embeddings_hidden_states, text_embeddings_class_labels, text_embeddings_boolean_prompt_mask = self.encode_text_2(target_prompt)
        uncond_embedding_hidden_states, uncond_embedding_class_lables, uncond_boolean_prompt_mask = self.encode_text_2([""])

        text_emb = PromptEmbeddings(embedding_hidden_states=text_embeddings_hidden_states,
                                boolean_prompt_mask=text_embeddings_boolean_prompt_mask,
                                embedding_class_labels=text_embeddings_class_labels)
        uncond_emb = PromptEmbeddings(embedding_hidden_states=uncond_embedding_hidden_states,
                                  boolean_prompt_mask=uncond_boolean_prompt_mask,
                                  embedding_class_labels=uncond_embedding_class_lables) 
        
        self.uncond_emb = uncond_emb
        
        return text_embeddings_class_labels, text_emb, uncond_emb
    
    @torch.no_grad()
    def unet_forward(self, x, t,  class_labels, encoder_hidden_states=None, encoder_attention_mask=None):
        return self.model.unet(x, t, encoder_hidden_states=None, class_labels=class_labels).sample
    
    
    # def set_attention_control(self, input_audio, Tshape, source_prompt, target_prompt, local, mutual, nb_consistency_steps, thresh_e, thresh_m, tau_s, tau_c, compute_second_mask=False):
    #     print("------- building attention controller ---------")
    #     try:
    #         local_blend = AttentionControlLDM.LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, save_inter=False)
    #         cross_replace_steps=tau_c
    #         self_replace_steps=tau_s

    #         controller = AttentionControlLDM.AttentionRefine([source_prompt, target_prompt],[[local, mutual]],
    #                         nb_consistency_steps,
    #                         0,
    #                         cross_replace_steps=cross_replace_steps,
    #                         self_replace_steps=self_replace_steps,
    #                         tokenizer=self.model.tokenizer,
    #                         encoder = self.model.text_encoder,
    #                         local_blend=local_blend
    #                         )
    #         print("------- passing controller into unet ---------")
    #         AttentionControlLDM.register_attention_control(self.model, controller)
    #     except Exception as e:
    #         print(f"------- error during attention control set-up : {e} ---------")
    #         controller, local_blend = None, None

    #     return controller, local_blend
    
    # def remove_attention_control(self):
    #     AttentionControlLDM.unregister_attention_control(self.model)
    

class AudioLDM2Wrapper(ModelWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from diffusers import AudioLDM2Pipeline
        
        self.model = AudioLDM2Pipeline.from_pretrained(self.model_id, token=self.token).to(self.device)
        self.model_name = "AudioLDM2"
        self.original_inference_steps = 200

    def get_fn_STFT(self) -> torch.nn.Module:
        return TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )
    
    def wav_to_mel(self, audio_path):
        fn_stft = self.get_fn_STFT()
        duration = get_duration(audio_path)
        mel_audio, _, _ = wav_to_fbank(audio_path, fn_STFT=fn_stft, target_length=int(duration * 102.4)) 
        mel_audio = mel_audio.unsqueeze(0)
        T_shape = mel_audio.shape[1]//2

        return mel_audio, T_shape

    def vae_encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = x.to(self.device)
        if x.shape[2] % 4:
            x = torch.nn.functional.pad(x, (0, 0, 4 - (x.shape[2] % 4), 0))

        x_0 = self.model.vae.encode(x).latent_dist.mode()
        shape = [x_0.shape[i] for i in range(1,4)]

        return (x_0 * self.model.vae.config.scaling_factor).float(), shape

    def vae_decode(self, x):
        return self.model.vae.decode(1 / self.model.vae.config.scaling_factor * x).sample.squeeze(1)

    def mel_to_wav(self, spec):
        spec = spec.squeeze(0)
        wav = self.model.mel_spectrogram_to_waveform(spec.detach().float()).detach()
        
        return wav
    
    def mel_to_wav_grad(self, spec):
        spec = spec.squeeze(0)
        wav = self.model.mel_spectrogram_to_waveform(spec)
        
        return wav

    def encode_text(self, prompts):
        from transformers import RobertaTokenizer, RobertaTokenizerFast

        tokenizers = [self.model.tokenizer, self.model.tokenizer_2]
        text_encoders = [self.model.text_encoder, self.model.text_encoder_2]
        prompt_embeds_list = []
        attention_mask_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompts,
                padding="max_length", # if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)) else True,
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] \
                    and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                print(f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                      f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                      )

            text_input_ids = text_input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                if text_encoder.config.model_type == "clap":
                    prompt_embeds = text_encoder.get_text_features(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    prompt_embeds = prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    attention_mask = attention_mask.new_ones((len(prompts), 1))
                else:
                    prompt_embeds = text_encoder(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    prompt_embeds = prompt_embeds[0]

            prompt_embeds_list.append(prompt_embeds)
            attention_mask_list.append(attention_mask)

        projection_output = self.model.projection_model(
            hidden_states=prompt_embeds_list[0],
            hidden_states_1=prompt_embeds_list[1],
            attention_mask=attention_mask_list[0],
            attention_mask_1=attention_mask_list[1],
        )
        projected_prompt_embeds = projection_output.hidden_states
        projected_attention_mask = projection_output.attention_mask

        generated_prompt_embeds = self.model.generate_language_model(
            projected_prompt_embeds,
            attention_mask=projected_attention_mask,
            max_new_tokens=None,
        )
        prompt_embeds = prompt_embeds.to(dtype=self.model.text_encoder_2.dtype, device=self.device)
        attention_mask = (
            attention_mask.to(device=self.device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=self.device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.model.language_model.dtype, device=self.device)

        text_emebdding = PromptEmbeddings(embedding_hidden_states=generated_prompt_embeds,
                                boolean_prompt_mask=attention_mask,
                                embedding_class_labels=prompt_embeds)

        return text_emebdding 
    
    
    @torch.no_grad()
    def unet_forward(self, sample, timestep, encoder_hidden_states=None, class_labels=None, encoder_attention_mask=None):
        encoder_hidden_states_1 = class_labels
        class_labels = None
        encoder_attention_mask_1 = encoder_attention_mask
        encoder_attention_mask = None
        return self.model.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                               encoder_hidden_states_1=encoder_hidden_states_1, class_labels=class_labels, 
                               encoder_attention_mask=encoder_attention_mask, encoder_attention_mask_1=encoder_attention_mask_1).sample

    
    
    # def set_attention_control(self, input_audio, Tshape, source_prompt, target_prompt, local, mutual, nb_consistency_steps, thresh_e, thresh_m, tau_s, tau_c, compute_second_mask=False):
    #     print("-------Building attention controller---------")
    #     try:
    #         local_blend = AttentionControlLDM2.LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, save_inter=False)
    #         cross_replace_steps=tau_c/nb_consistency_steps
    #         self_replace_steps=tau_s/nb_consistency_steps

    #         controller = AttentionControlLDM2.AttentionRefine([source_prompt, target_prompt],[[local, mutual]],
    #                         nb_consistency_steps,
    #                         0,
    #                         cross_replace_steps=cross_replace_steps,
    #                         self_replace_steps=self_replace_steps,
    #                         tokenizer=self.model.tokenizer_2,
    #                         encoder = self.model.text_encoder_2,
    #                         clap_tokenizer=self.model.tokenizer,
    #                         clap_encoder=self.model.text_encoder,
    #                         local_blend=local_blend,

    #                         )
    #         print("-------passing controller into unet---------")
    #         AttentionControlLDM2.register_attention_control(self.model, controller)
    #     except Exception as e:
    #         print(f"-------error during attention control set-up : {e}---------")
    #         controller, local_blend = None, None

    #     return controller, local_blend
    
    # def remove_attention_control(self):
    #     AttentionControlLDM2.unregister_attention_control(self.model)
    
   