"""CONSISTENCY SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import List, Optional, Tuple, Union, NamedTuple
from einops import rearrange

class PromptEmbeddings(NamedTuple):
    embedding_hidden_states: torch.Tensor
    embedding_class_labels: torch.Tensor
    boolean_prompt_mask: torch.Tensor


class LCMSampler(object):
    def __init__(self, wrapper, controller=None, store_latents=False, phi=None,
                 guidance_scale=1.0, src_guidance_scale=1.0, local_blend=None, **kwargs):
        super().__init__()
        self.wrapper = wrapper
        self.ddpm_num_timesteps = 1000 
        self.original_inference_steps = wrapper.original_inference_steps
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, self.ddpm_num_timesteps)[::-1].copy().astype(np.int64))
        self.custom_timesteps = False
        self.timestep_scaling = 10.0
        self.prediction_type = 'epsilon'
        self.controller = controller
        self.store_latents = store_latents
        self.latents = []
        self.epsilons = []
        self.phi = phi
        self.guidance_scale = guidance_scale
        self.src_guidance_scale = src_guidance_scale
        self.local_blend = local_blend


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda") and torch.cuda.is_available():
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_discretize="uniform", verbose=True):
       
        alphas_cumprod = self.wrapper.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, f'alphas {alphas_cumprod.shape[0]} have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.wrapper.device)
        self.register_buffer('betas', to_torch(self.wrapper.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.wrapper.alphas_cumprod_prev))

       
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
        
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        original_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        strength: int = 1.0,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps, which will be used to generate a linearly-spaced timestep
                schedule (which is different from the standard `diffusers` implementation). We will then take
                `num_inference_steps` timesteps from this schedule, evenly spaced in terms of indices, and use that as
                our final timestep schedule. If not set, this will default to the `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps on the training/distillation timestep
                schedule is used. If `timesteps` is passed, `num_inference_steps` must be `None`.
        """
        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        # 1. Calculate the LCM original training/distillation timestep schedule.
        original_steps = (
            original_inference_steps if original_inference_steps is not None else self.original_inference_steps
        )

        if original_steps > self.ddpm_num_timesteps:
            raise ValueError(
                f"`original_steps`: {original_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.ddpm_num_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.ddpm_num_timesteps} timesteps."
            )
        # import ipdb
        # ipdb.set_trace()
        # LCM Timesteps Setting
        # The skipping step parameter k from the paper.
        k = self.ddpm_num_timesteps // original_steps
        # LCM Training/Distillation Steps Schedule
        # Currently, only a linearly-spaced schedule is supported (same as in the LCM distillation scripts).
        lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * k - 1

        # 2. Calculate the LCM inference timestep schedule.
        if timesteps is not None:
            # 2.1 Handle custom timestep schedules.
            train_timesteps = set(lcm_origin_timesteps)
            non_train_timesteps = []
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

                if timesteps[i] not in train_timesteps:
                    non_train_timesteps.append(timesteps[i])

            if timesteps[0] >= self.ddpm_num_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.ddpm_num_timesteps}."
                )

            # Raise warning if timestep schedule does not start with self.config.num_train_timesteps - 1
            if strength == 1.0 and timesteps[0] != self.ddpm_num_timesteps - 1:
                logger.warning(
                    f"The first timestep on the custom timestep schedule is {timesteps[0]}, not"
                    f" `self.ddpm_num_timesteps - 1`: {self.ddpm_num_timesteps - 1}. You may get"
                    f" unexpected results when using this timestep schedule."
                )

            # Raise warning if custom timestep schedule contains timesteps not on original timestep schedule
            if non_train_timesteps:
                logger.warning(
                    f"The custom timestep schedule contains the following timesteps which are not on the original"
                    f" training/distillation timestep schedule: {non_train_timesteps}. You may get unexpected results"
                    f" when using this timestep schedule."
                )

            # Raise warning if custom timestep schedule is longer than original_steps
            if len(timesteps) > original_steps:
                logger.warning(
                    f"The number of timesteps in the custom timestep schedule is {len(timesteps)}, which exceeds the"
                    f" the length of the timestep schedule used for training: {original_steps}. You may get some"
                    f" unexpected results when using this timestep schedule."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.num_inference_steps = len(timesteps)
            self.custom_timesteps = True

            # Apply strength (e.g. for img2img pipelines) (see StableDiffusionImg2ImgPipeline.get_timesteps)
            init_timestep = min(int(self.num_inference_steps * strength), self.num_inference_steps)
            t_start = max(self.num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start * self.order :]
            # TODO: also reset self.num_inference_steps?
        else:
            # 2.2 Create the "standard" LCM inference timestep schedule.
            if num_inference_steps > self.ddpm_num_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.ddpm_num_timesteps`:"
                    f" {self.ddpm_num_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.ddpm_num_timesteps} timesteps."
                )

            skipping_step = len(lcm_origin_timesteps) // num_inference_steps

            if skipping_step < 1:
                raise ValueError(
                    f"The combination of `original_steps x strength`: {original_steps} x {strength} is smaller than `num_inference_steps`: {num_inference_steps}. Make sure to either reduce `num_inference_steps` to a value smaller than {int(original_steps * strength)} or increase `strength` to a value higher than {float(num_inference_steps / original_steps)}."
                )

            self.num_inference_steps = num_inference_steps

            if num_inference_steps > original_steps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
                    f" {original_steps} because the final timestep schedule will be a subset of the"
                    f" `original_inference_steps`-sized initial timestep schedule."
                )

            # LCM Inference Steps Schedule
            lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
            # Select (approximately) evenly spaced indices from lcm_origin_timesteps.
            inference_indices = np.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            timesteps = lcm_origin_timesteps[inference_indices]

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

        self._step_index = None

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    def retrieve_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                    Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                    timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                    must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None:
            self.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = self.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = self.timesteps
        return timesteps, num_inference_steps   
    
    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    def get_scalings_for_boundary_condition_discrete(self, timestep):
        self.sigma_data = 0.5  # Default: 0.5
        scaled_timestep = timestep * self.timestep_scaling

        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out
    
    def VC_sampling_edit(self,
               S,
               batch_size,
               shape,
               x_0,
               source_cond,
               target_cond,
               verbose=True,
               x_T=None,
               timesteps=None,
               emb_source=None,
               emb_target=None,
               **kwargs
               ):

        device = self.wrapper.device
        self.make_schedule(verbose=verbose)
        self.num_inference_steps = S
        attention_control = self.controller is not None

        if len(shape)==3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, T = shape
            size = (batch_size, C, T) 
    
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.num_inference_steps, device, timesteps, original_inference_steps=self.original_inference_steps
        )
        b = size[0]

        if x_T is None:
            x_target_t = torch.randn(size, device=device)
            x_source_t = x_target_t.clone()
            x_mutual_t = x_target_t.clone() if attention_control else None

        else:
            x_target_t = x_T
            x_source_t = x_target_t.clone()
            x_mutual_t = x_target_t.clone() if attention_control else None

        w = torch.tensor(self.guidance_scale - 1).repeat(b)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=256).to(
            device=device, dtype=x_target_t.dtype
        )
        
        # Latent Consistency MultiStep Sampling Loop:
        num_warmup_steps = len(timesteps) - num_inference_steps
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                x_target_t = x_target_t.to(source_cond.dtype)
                x_source_t = x_source_t.to(source_cond.dtype)
                x_mutual_t = x_source_t.to(source_cond.dtype)
                
                # compute the previous noisy sample x_t -> x_t-1
                if self.wrapper.model_name=="AudioLCM":
                    x_source_t, x_target_t, x_mutual_t, x_target_0_pred = self.edit_step_lcm(x_0, t, x_target_t, x_source_t, x_mutual_t, 
                                                                                             source_cond, target_cond, b, w_embedding, 
                                                                                             return_dict=False)
                elif attention_control:
                    x_source_t, x_target_t, x_mutual_t, x_target_0_pred = self.edit_step_ldm_ac(x_0, t, x_target_t, x_source_t, 
                                                                                                x_mutual_t, source_cond, target_cond, b, 
                                                                                                emb_source, emb_target, return_dict=False)
                else:
                    x_source_t, x_target_t, x_target_0_pred = self.edit_step_ldm(x_0, t, x_target_t, x_source_t, 
                                                                                source_cond, target_cond, b, 
                                                                                emb_source, emb_target, return_dict=False)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()

        return x_source_t, x_target_t, x_target_0_pred
    

    def edit_step_lcm(
        self,
        x_0: torch.FloatTensor,
        timestep: int,
        x_target_t: torch.FloatTensor,
        x_source_t: torch.FloatTensor,
        x_mutual_t: torch.FloatTensor,
        c_source: torch.FloatTensor,
        c_target: torch.FloatTensor,
        b: int,
        w_embedding: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        
        attention_control = self.controller is not None
        device = self.wrapper.device
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 2. compute alphas, betas and "consistent" noise
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        epsilon_cons = (x_source_t-alpha_prod_t.sqrt()*x_0)/beta_prod_t.sqrt()

        ts = torch.full((b,), timestep, device=device, dtype=torch.long)
        
        x_concat = torch.cat([x_source_t,x_target_t])
        c_concat = torch.cat([c_source,c_target])
        ts_concat = torch.full((2*b,), timestep, device=device, dtype=torch.long)
        w_cond_combined = torch.cat([w_embedding, w_embedding])

        if attention_control: # custom attention control
            # 1st pass (source + tgt)
            self.controller.set_src_tgt_pass()
            epsilon_concat = self.wrapper.unet_forward(x_concat, ts_concat, c_concat, w_cond_combined)
            epsilon_pred_source, epsilon_pred_target = epsilon_concat.chunk(2)
            self.controller.reinit_layers()

            # 2nd pass (mutual)
            self.controller.set_mutual_pass()
            epsilon_pred_mutual = self.wrapper.unet_forward(x_mutual_t, ts, c_source, w_embedding)
            self.controller.reinit_layers()

            # 3rd pass (tgt refine)
            self.controller.set_target_only()
            epsilon_pred_target = self.wrapper.unet_forward(x_target_t, ts, c_target, w_embedding)
        else:
            epsilon_concat = self.wrapper.unet_forward(x_concat, ts_concat, c_concat, w_cond_combined)
            epsilon_pred_source, epsilon_pred_target = epsilon_concat.chunk(2)

        if self.phi is None:
            v = (epsilon_pred_target - epsilon_pred_source) + epsilon_cons
            v = v - v.mean()
            epsilon_edit = v / torch.sqrt(v**2).mean()          
        else:
            phi1 = self.phi / np.sqrt(2)
            phi2 = torch.sqrt(1-self.phi**2)
            v = phi1 * (epsilon_pred_target - epsilon_pred_source) + phi2 * epsilon_cons
            v = v - v.mean()
            epsilon_edit = v / v.std()

        # 3. Get scalings for boundary conditions

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the edited epsilon
        x_target_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_edit) / alpha_prod_t.sqrt()
        x_source_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_pred_source) / alpha_prod_t.sqrt()
        x_mutual_0_pred = (x_mutual_t - beta_prod_t.sqrt() * epsilon_pred_mutual) / alpha_prod_t.sqrt() if attention_control else None

        if self.local_blend is not None:
            attention_store = torch.stack([torch.stack(self.controller.M_attention_src, dim=0).mean(dim=0), torch.stack(self.controller.M_attention_tgt, dim=0).mean(dim=0)], dim=0)
            x_mutual_0_pred, x_target_0_pred = self.local_blend(x_0, x_target_0_pred, x_mutual_0_pred, attention_store, alpha_prod_t)

        if self.controller is not None:
            self.controller.reinit_layers()
            self.controller.increment_step()
            self.controller.del_stored_qkv()

        # 5. Denoise model output using boundary conditions
        denoised_target = c_out * x_target_0_pred + c_skip * x_target_t
        denoised_source = c_out * x_0 + c_skip * x_source_t
        denoised_mutual = c_out * x_mutual_0_pred + c_skip * x_mutual_t if attention_control else None
        
        if self.store_latents:
            denoised_source_pred = c_out * x_source_0_pred + x_source_t
            self.latents.append((timestep, denoised_source_pred, denoised_source,denoised_target))
            self.epsilons.append((timestep,epsilon_pred_source, epsilon_pred_target, epsilon_cons, epsilon_edit))


        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(epsilon_edit.shape, device=epsilon_edit.device)
            x_target_prev = alpha_prod_t_prev.sqrt() * denoised_target + beta_prod_t_prev.sqrt() * noise
            x_source_prev = alpha_prod_t_prev.sqrt() * denoised_source + beta_prod_t_prev.sqrt() * noise
            x_mutual_prev = alpha_prod_t_prev.sqrt() * denoised_mutual + beta_prod_t_prev.sqrt() * noise if attention_control else None

        else:
            x_target_prev = denoised_target
            x_source_prev = denoised_source
            x_mutual_prev = denoised_mutual

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred)

        return x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred
    

    def edit_step_ldm_ac(
        self,
        x_0: torch.FloatTensor,
        timestep: int,
        x_target_t: torch.FloatTensor,
        x_source_t: torch.FloatTensor,
        x_mutual_t: torch.FloatTensor,
        c_source: torch.FloatTensor,
        c_target: torch.FloatTensor,
        b: int,
        emb_source: PromptEmbeddings,
        emb_target: PromptEmbeddings,
        return_dict: bool = True,
    ):
                
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        if self.step_index is None:
            self._init_step_index(timestep)
        # 0. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 1. compute alphas, betas and consistent noise eps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        epsilon_cons = (x_source_t-alpha_prod_t.sqrt()*x_0)/beta_prod_t.sqrt()
        
        # 2. prepare latents and conditioning
        nb_branch = 3
        latent = torch.cat([x_source_t, x_target_t, x_mutual_t])       
        ts = torch.full((nb_branch*b,), timestep, device=self.wrapper.device, dtype=torch.long)
        
        # CFG
        do_classifier_free_guidance = self.guidance_scale > 1.0

        x_concat = torch.cat([latent]*2) if do_classifier_free_guidance else latent
        ts_concat = torch.cat([ts]*2) if do_classifier_free_guidance else ts
        
        if emb_source is None :  # AudioLDM cond
            c_cond = torch.cat([c_source, c_target, c_source], dim=0)
            c_null, _ ,_ = self.wrapper.encode_text_improved([""])
            c_uncond = torch.cat([c_null]*nb_branch)

            class_labels_concat = torch.cat([c_cond, c_uncond]) if do_classifier_free_guidance else c_cond
            encoder_hidden_states_concat = None
            encoder_attention_mask_concat = None
        else: # AudioLDM2 cond
            encoder_hidden_states_cond = torch.cat([emb_source.embedding_hidden_states, emb_target.embedding_hidden_states, 
                                                    emb_source.embedding_hidden_states])
            class_labels_cond = torch.cat([emb_source.embedding_class_labels, emb_target.embedding_class_labels, 
                                        emb_source.embedding_class_labels])
            encoder_attention_mask_cond = torch.cat([emb_source.boolean_prompt_mask, emb_target.boolean_prompt_mask,
                                                    emb_source.boolean_prompt_mask])      
            emb_null = self.wrapper.encode_text([""])
            encoder_hidden_states_uncond = torch.cat([emb_null.embedding_hidden_states]*nb_branch)
            class_labels_uncond = torch.cat([emb_null.embedding_class_labels]*nb_branch)
            encoder_attention_mask_uncond = torch.cat([emb_null.boolean_prompt_mask]*nb_branch)
            
            encoder_hidden_states_concat = torch.cat([encoder_hidden_states_cond,encoder_hidden_states_uncond]) if do_classifier_free_guidance else encoder_hidden_states_cond
            class_labels_concat = torch.cat([class_labels_cond, class_labels_uncond]) if do_classifier_free_guidance else class_labels_cond
            encoder_attention_mask_concat = torch.cat([encoder_attention_mask_cond, encoder_attention_mask_uncond]) if do_classifier_free_guidance else encoder_attention_mask_cond

        
        # get noise prediction
        epsilon_concat = self.wrapper.unet_forward(x_concat, ts_concat, encoder_hidden_states=encoder_hidden_states_concat, 
                                                   class_labels=class_labels_concat, encoder_attention_mask=encoder_attention_mask_concat)

        # perform guidance
        if do_classifier_free_guidance:
            eps_pred_cond, eps_pred_uncond = epsilon_concat.chunk(2)
            epsilon_pred_source_c, epsilon_pred_target_c, epsilon_pred_mutual_c = eps_pred_cond.chunk(nb_branch)
            epsilon_pred_source_uc, epsilon_pred_target_uc, epsilon_pred_mutual_uc = eps_pred_uncond.chunk(nb_branch)
            del eps_pred_cond, eps_pred_uncond
            epsilon_pred_source = epsilon_pred_source_uc + self.src_guidance_scale * (epsilon_pred_source_c - epsilon_pred_source_uc)
            epsilon_pred_target = epsilon_pred_target_uc + self.guidance_scale * (epsilon_pred_target_c - epsilon_pred_target_uc)
            epsilon_pred_mutual = epsilon_pred_mutual_uc + self.src_guidance_scale * (epsilon_pred_mutual_c - epsilon_pred_mutual_uc)
            del epsilon_pred_source_uc, epsilon_pred_target_uc, epsilon_pred_mutual_uc, epsilon_pred_source_c, epsilon_pred_target_c, epsilon_pred_mutual_c
        else:
            epsilon_pred_source, epsilon_pred_target, epsilon_pred_mutual = epsilon_concat.chunk(nb_branch)

        del epsilon_concat

        epsilon_mutual_rect = (epsilon_pred_mutual - epsilon_pred_source) + epsilon_cons

        # compute edit noise
        if self.phi is None:
            v = (epsilon_pred_target - epsilon_pred_source) + epsilon_cons
            v = v - v.mean()
            epsilon_edit = v / torch.sqrt(v**2).mean()          
        else:
            phi1 = self.phi / np.sqrt(2)
            phi2 = torch.sqrt(1-self.phi**2)
            v = phi1 * (epsilon_pred_target - epsilon_pred_source) + phi2 * epsilon_cons
            v = v - v.mean()
            epsilon_edit = v / v.std()


        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the edited epsilon
        x_target_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_edit) / alpha_prod_t.sqrt()
        with torch.no_grad():
            x_source_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_pred_source) / alpha_prod_t.sqrt()
            x_mutual_0_pred = (x_mutual_t - beta_prod_t.sqrt() * epsilon_mutual_rect) / alpha_prod_t.sqrt()
            
        del epsilon_edit, epsilon_pred_source, epsilon_pred_target, epsilon_pred_mutual
        
       
        # 5. Denoise model output using boundary conditions
        denoised_target = c_out * x_target_0_pred + c_skip * x_target_t
        denoised_source = c_out * x_0 + c_skip * x_source_t
        denoised_mutual = c_out * x_mutual_0_pred + c_skip * x_mutual_t
        

        if self.store_latents:
            denoised_source_pred = c_out * x_source_0_pred + x_source_t
            self.latents.append((timestep, denoised_source_pred, denoised_source,denoised_target))

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(denoised_target.shape, device=denoised_target.device)           
            x_target_prev = alpha_prod_t_prev.sqrt() * denoised_target + beta_prod_t_prev.sqrt() * noise
            x_source_prev = alpha_prod_t_prev.sqrt() * denoised_source + beta_prod_t_prev.sqrt() * noise
            x_mutual_prev = alpha_prod_t_prev.sqrt() * denoised_mutual + beta_prod_t_prev.sqrt() * noise

        else:
            x_target_prev = denoised_target
            x_source_prev = denoised_source
            x_mutual_prev = denoised_mutual

        # upon completion increase step index by one
        self._step_index += 1
        torch.cuda.empty_cache()

        if not return_dict:
            return (x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred)

        return x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred
    

    def edit_step_ldm(
        self,
        x_0: torch.FloatTensor,
        timestep: int,
        x_target_t: torch.FloatTensor,
        x_source_t: torch.FloatTensor,
        c_source: torch.FloatTensor,
        c_target: torch.FloatTensor,
        b: int,
        emb_source: PromptEmbeddings,
        emb_target: PromptEmbeddings,
        return_dict: bool = True,
    ):
                
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        if self.step_index is None:
            self._init_step_index(timestep)
        # 0. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 1. compute alphas, betas and consistent noise eps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        epsilon_cons = (x_source_t-alpha_prod_t.sqrt()*x_0)/beta_prod_t.sqrt()
        
        # 2. prepare latents and conditioning
        nb_branch = 2
        latent = torch.cat([x_source_t, x_target_t])       
        ts = torch.full((nb_branch*b,), timestep, device=self.wrapper.device, dtype=torch.long)
        
        # CFG
        do_classifier_free_guidance = self.guidance_scale > 1.0

        x_concat = torch.cat([latent]*2) if do_classifier_free_guidance else latent
        ts_concat = torch.cat([ts]*2) if do_classifier_free_guidance else ts
        
        if emb_source is None :  # AudioLDM cond
            c_cond = torch.cat([c_source, c_target], dim=0)
            c_null, _ ,_ = self.wrapper.encode_text_improved([""])
            c_uncond = torch.cat([c_null]*nb_branch)
            
            class_labels_concat = torch.cat([c_cond, c_uncond]) if do_classifier_free_guidance else c_cond
            encoder_hidden_states_concat = None
            encoder_attention_mask_concat = None
        else: # AudioLDM2 cond
            encoder_hidden_states_cond = torch.cat([emb_source.embedding_hidden_states, emb_target.embedding_hidden_states])
            class_labels_cond = torch.cat([emb_source.embedding_class_labels, emb_target.embedding_class_labels])
            encoder_attention_mask_cond = torch.cat([emb_source.boolean_prompt_mask, emb_target.boolean_prompt_mask])      
            emb_null = self.wrapper.encode_text([""])
            encoder_hidden_states_uncond = torch.cat([emb_null.embedding_hidden_states]*nb_branch)
            class_labels_uncond = torch.cat([emb_null.embedding_class_labels]*nb_branch)
            encoder_attention_mask_uncond = torch.cat([emb_null.boolean_prompt_mask]*nb_branch)
            
            encoder_hidden_states_concat = torch.cat([encoder_hidden_states_cond,encoder_hidden_states_uncond]) if do_classifier_free_guidance else encoder_hidden_states_cond
            class_labels_concat = torch.cat([class_labels_cond, class_labels_uncond]) if do_classifier_free_guidance else class_labels_cond
            encoder_attention_mask_concat = torch.cat([encoder_attention_mask_cond, encoder_attention_mask_uncond]) if do_classifier_free_guidance else encoder_attention_mask_cond

        
        # get noise prediction
        epsilon_concat = self.wrapper.unet_forward(x_concat, ts_concat, encoder_hidden_states=encoder_hidden_states_concat, 
                                                   class_labels=class_labels_concat, encoder_attention_mask=encoder_attention_mask_concat)


        # perform guidance
        if do_classifier_free_guidance:
            eps_pred_cond, eps_pred_uncond = epsilon_concat.chunk(2)
            epsilon_pred_source_c, epsilon_pred_target_c = eps_pred_cond.chunk(nb_branch)
            epsilon_pred_source_uc, epsilon_pred_target_uc = eps_pred_uncond.chunk(nb_branch)
            del eps_pred_cond, eps_pred_uncond
            epsilon_pred_source = epsilon_pred_source_uc + self.src_guidance_scale * (epsilon_pred_source_c - epsilon_pred_source_uc)
            epsilon_pred_target = epsilon_pred_target_uc + self.guidance_scale * (epsilon_pred_target_c - epsilon_pred_target_uc)
            del epsilon_pred_source_uc, epsilon_pred_target_uc, epsilon_pred_source_c, epsilon_pred_target_c
        else:
            epsilon_pred_source, epsilon_pred_target = epsilon_concat.chunk(nb_branch)

        del epsilon_concat

        # compute edit noise
        if self.phi is None:
            v = (epsilon_pred_target - epsilon_pred_source) + epsilon_cons
            v = v - v.mean()
            epsilon_edit = v / torch.sqrt(v**2).mean()          
        else:
            phi1 = self.phi / np.sqrt(2)
            phi2 = torch.sqrt(1-self.phi**2)
            v = phi1 * (epsilon_pred_target - epsilon_pred_source) + phi2 * epsilon_cons
            v = v - v.mean()
            epsilon_edit = v / v.std()


        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the edited epsilon
        x_target_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_edit) / alpha_prod_t.sqrt()
        with torch.no_grad():
            x_source_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_pred_source) / alpha_prod_t.sqrt()
            
        del epsilon_edit, epsilon_pred_source, epsilon_pred_target
        
       
        # 5. Denoise model output using boundary conditions
        denoised_target = c_out * x_target_0_pred + c_skip * x_target_t
        denoised_source = c_out * x_0 + c_skip * x_source_t
        

        if self.store_latents:
            denoised_source_pred = c_out * x_source_0_pred + x_source_t
            self.latents.append((timestep, denoised_source_pred, denoised_source,denoised_target))

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(denoised_target.shape, device=denoised_target.device)           
            x_target_prev = alpha_prod_t_prev.sqrt() * denoised_target + beta_prod_t_prev.sqrt() * noise
            x_source_prev = alpha_prod_t_prev.sqrt() * denoised_source + beta_prod_t_prev.sqrt() * noise

        else:
            x_target_prev = denoised_target
            x_source_prev = denoised_source

        # upon completion increase step index by one
        self._step_index += 1
        torch.cuda.empty_cache()

        if not return_dict:
            return (x_source_prev, x_target_prev, x_target_0_pred)

        return x_source_prev, x_target_prev, x_target_0_pred