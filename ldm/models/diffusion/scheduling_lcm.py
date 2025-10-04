"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import List, Optional, Tuple, Union
from einops import rearrange

class LCMSampler(object):
    def __init__(self, model, controller=None, store_latents=False, eta=1, guiding_audio=None, local_blend=None, unet=None,**kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.original_inference_steps = 100
        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, self.ddpm_num_timesteps)[::-1].copy().astype(np.int64))
        self.custom_timesteps = False
        self.timestep_scaling = 10.0
        self.prediction_type = 'epsilon'

        # MATT
        self.controller = controller
        self.store_latents = store_latents
        self.latents = []
        self.epsilons = []
        self.eta = eta
        self.guiding_audio = guiding_audio
        self.local_blend = local_blend
        self.unet = unet


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda") and torch.cuda.is_available():
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_discretize="uniform", verbose=True):
       
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

       
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

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               verbose=True,
               x_T=None,
               guidance_scale=5.,
               original_inference_steps=50,
               timesteps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(verbose=verbose)
        self.num_inference_steps = S
        # sampling
        if len(shape)==3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, T = shape
            size = (batch_size, C, T) 

        samples, intermediates = self.lcm_sampling(conditioning, size,
                                                    x_T=x_T,
                                                    guidance_scale=guidance_scale,
                                                    original_inference_steps=original_inference_steps,
                                                    timesteps=timesteps
                                                    )
        return samples, intermediates
    
    @torch.no_grad()
    def sample_inversion(self,
               S,
               batch_size,
               shape,
               x_0,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               verbose=True,
               x_T=None,
               guidance_scale=5.,
               original_inference_steps=50,
               timesteps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(verbose=verbose)
        self.num_inference_steps = S
        # sampling
        if len(shape)==3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, T = shape
            size = (batch_size, C, T) 

        samples, intermediates = self.lcm_sampling_inversion(x_0, conditioning, size,
                                                    x_T=x_T,
                                                    guidance_scale=guidance_scale,
                                                    original_inference_steps=original_inference_steps,
                                                    timesteps=timesteps
                                                    )
        return samples, intermediates
    
    #@torch.no_grad()
    def sample_edit(self,
               S,
               batch_size,
               shape,
               x_0,
               base_conditioning = None,
               conditioning=None,
               verbose=True,
               x_T=None,
               guidance_scale=5.,
               original_inference_steps=50,
               timesteps=None,
               **kwargs
               ):

        self.make_schedule(verbose=verbose)
        self.num_inference_steps = S
        # sampling
        if len(shape)==3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, T = shape
            size = (batch_size, C, T) 

        x_source_t, x_target_t, x_mutual_t, x_target_0_pred = self.lcm_sampling_edit(x_0, base_conditioning, conditioning, size,
                                                    x_T=x_T,
                                                    guidance_scale=guidance_scale,
                                                    original_inference_steps=original_inference_steps,
                                                    timesteps=timesteps
                                                    )
        return x_source_t, x_target_t, x_mutual_t, x_target_0_pred

    @torch.no_grad()
    def lcm_sampling(self, cond, shape,
                      x_T=None,
                      guidance_scale=1.,original_inference_steps=100,timesteps=None):
        device = self.model.betas.device
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
        )
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        
        w = torch.tensor(guidance_scale - 1).repeat(b)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=256).to(
            device=device, dtype=img.dtype
        )
        
        # import ipdb
        # ipdb.set_trace()
        # 8. LCM MultiStep Sampling Loop:
        num_warmup_steps = len(timesteps) - num_inference_steps
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                img = img.to(cond.dtype)
                ts = torch.full((b,), t, device=device, dtype=torch.long)
                # model prediction (v-prediction, eps, x)
                model_pred = self.model.apply_model(img, ts, cond,self.model.unet, w_cond=w_embedding)

                # compute the previous noisy sample x_t -> x_t-1
                img, denoised = self.step(model_pred, t, img, return_dict=False)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()
        return denoised, img
    
    @torch.no_grad()
    def lcm_sampling_inversion(self, x_0, cond, shape,
                      x_T=None,
                      guidance_scale=1.,original_inference_steps=100,timesteps=None):
        device = self.model.betas.device
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
        )
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        
        w = torch.tensor(guidance_scale - 1).repeat(b)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=256).to(
            device=device, dtype=img.dtype
        )
        
        # 8. LCM MultiStep Sampling Loop:
        num_warmup_steps = len(timesteps) - num_inference_steps
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                img = img.to(cond.dtype)
                ts = torch.full((b,), t, device=device, dtype=torch.long)
                # model_pred = self.model.apply_model(img, ts, cond,self.model.unet, w_cond=w_embedding)
                
                # Matt Modified 
                # compute the previous noisy sample x_t -> x_t-1
                img, denoised = self.inversion_step(x_0, t, img, return_dict=False)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()
        return denoised, img
    
    #@torch.no_grad()
    def lcm_sampling_edit(self, x_0, source_cond, target_cond, shape,
                      x_T=None,
                      guidance_scale=1.,original_inference_steps=100,timesteps=None):
        device = self.model.betas.device
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
        )
        b = shape[0]
        if x_T is None:
            x_target_t = torch.randn(shape, device=device)
            x_source_t = x_target_t.clone()
            x_mutual_t = x_target_t.clone()

        else:
            x_target_t = x_T
            x_source_t = x_target_t.clone()
            x_mutual_t = x_target_t.clone()

        w = torch.tensor(guidance_scale - 1).repeat(b)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=256).to(
            device=device, dtype=x_target_t.dtype
        )
        
        # 8. LCM MultiStep Sampling Loop:
        num_warmup_steps = len(timesteps) - num_inference_steps
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                x_target_t = x_target_t.to(target_cond.dtype)
                x_source_t = x_source_t.to(source_cond.dtype)
                x_mutual_t = x_source_t.to(source_cond.dtype)
                
                # compute the previous noisy sample x_t -> x_t-1
                if self.controller is None or not(self.controller.is_matt):
                    x_source_t, x_target_t, x_mutual_t, x_target_0_pred = self.edit_step(x_0, t, x_target_t,x_source_t, x_mutual_t, source_cond,target_cond, b, w_embedding, return_dict=False)
                else: 
                     x_source_t, x_target_t, x_mutual_t, x_target_0_pred = self.edit_step_w_control(x_0, t, x_target_t,x_source_t, x_mutual_t, source_cond,target_cond, b, w_embedding, return_dict=False)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()
        return x_source_t, x_target_t, x_mutual_t, x_target_0_pred

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

    @torch.no_grad()
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
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

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )


        # 5. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(model_output.shape, device=model_output.device)
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, denoised)

        return prev_sample, denoised
    
    @torch.no_grad()
    def inversion_step(
        self,
        x_0: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        '''
        sample : x_t
        '''
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

        # 2. compute alphas, betas and constrained noise epsilon
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        model_output = (sample-alpha_prod_t.sqrt()*x_0)/beta_prod_t.sqrt()

        # 3. Get scalings for boundary conditions

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )


        # 5. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(model_output.shape, device=model_output.device)
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, denoised)

        return prev_sample, denoised            # img = prev_sample
    
    @torch.no_grad()
    def edit_step(
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
                
        device = self.model.betas.device
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
        
        # 2. compute alphas, betas and constrained noise epsilon
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        epsilon_constrained = (x_source_t-alpha_prod_t.sqrt()*x_0)/beta_prod_t.sqrt()

        x_source_t = x_source_t.to(c_source.dtype)
        # ts = torch.full((b,), timestep, device=device, dtype=torch.long)
        # epsilon_pred_source = self.model.apply_model(x_source_t, ts, c_source, self.model.unet, w_cond=w_embedding)

        x_target_t = x_target_t.to(c_target.dtype)
        x_mutual_t = x_mutual_t.to(c_target.dtype)
        # ts2 = torch.full((b,), timestep, device=device, dtype=torch.long)
        # epsilon_pred_target = self.model.apply_model(x_target_t, ts2, c_target, self.model.unet, w_cond=w_embedding)

        # x_target_t = slerp(0.9, x_source_t, x_target_t)  

        # Instead of doing 3 forward passes on the model, just one forward pass with all source, target and mutual branches concatenated
        x_concat = torch.cat([x_source_t,x_target_t,x_mutual_t], dim=0)
        c_concat = torch.cat([c_source, c_target, c_source], dim=0)
        ts_concat = torch.full((3*b,), timestep, device=device, dtype=torch.long)
        w_cond_combined = torch.cat([w_embedding, w_embedding, w_embedding], dim=0)
        if self.unet is None:
            epsilon_concat = self.model.apply_model(x_concat, ts_concat, c_concat, self.model.unet, w_cond=w_cond_combined)
        else:
            epsilon_concat = self.unet(x_concat, timestep, encoder_hidden_states=None, class_labels=c_concat).sample

        epsilon_pred_source, epsilon_pred_target, epsilon_pred_mutual = epsilon_concat.chunk(3)

        if self.guiding_audio is None:
            epsilon_edit = (self.eta * epsilon_pred_target - self.eta * epsilon_pred_source) + epsilon_constrained 
        else:
            # using an audio base to guide the model
            out = self.guiding_audio
            epsilon_edit = (self.eta * epsilon_pred_target - self.eta * epsilon_pred_source) + epsilon_constrained 
            # epsilon_edit = 


        # 3. Get scalings for boundary conditions

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the edited epsilon
        x_target_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_edit) / alpha_prod_t.sqrt()
        x_mutual_0_pred = (x_mutual_t - beta_prod_t.sqrt() * epsilon_pred_mutual) / alpha_prod_t.sqrt()
        x_target_0_real_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_pred_target) / alpha_prod_t.sqrt()


        # 5. Denoise model output using boundary conditions
        denoised_target = c_out * x_target_0_pred + c_skip * x_target_t
        denoised_source = c_out * x_0 + c_skip * x_source_t
        denoised_mutual = c_out * x_mutual_0_pred + c_skip * x_mutual_t

        denoised_target_real = c_out * x_target_0_real_pred + c_skip * x_target_t

        if self.store_latents:
            self.latents.append((timestep, denoised_mutual, denoised_target_real, denoised_source,denoised_target))
            self.epsilons.append((timestep,epsilon_pred_mutual, epsilon_pred_target, epsilon_constrained, epsilon_edit))

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(epsilon_edit.shape, device=epsilon_edit.device)
            x_target_prev = alpha_prod_t_prev.sqrt() * denoised_target + beta_prod_t_prev.sqrt() * noise
            x_source_prev = alpha_prod_t_prev.sqrt() * denoised_source + beta_prod_t_prev.sqrt() * noise
            x_mutual_prev = alpha_prod_t_prev.sqrt() * denoised_mutual + beta_prod_t_prev.sqrt() * noise
        else:
            x_target_prev = denoised_target
            x_source_prev = denoised_source
            x_mutual_prev = denoised_mutual

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred)

        return x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred
    
    #@torch.no_grad()
    def edit_step_w_control(
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
                
        device = self.model.betas.device
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

        # 2. compute alphas, betas and constrained noise epsilon
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        epsilon_constrained = (x_source_t-alpha_prod_t.sqrt()*x_0)/beta_prod_t.sqrt()

        x_source_t = x_source_t.to(c_source.dtype)
        x_target_t = x_target_t.to(c_target.dtype)
        x_mutual_t = x_mutual_t.to(c_target.dtype)

        ts = torch.full((b,), timestep, device=device, dtype=torch.long)
        
        x_concat = torch.cat([x_source_t,x_target_t], dim=0)    # changed [x_src,x_tgt] to [x_tgt,x_tgt] for a closer eps prediction
        c_concat = torch.cat([c_source,c_target], dim=0)
        ts_concat = torch.full((2*b,), timestep, device=device, dtype=torch.long)
        w_cond_combined = torch.cat([w_embedding, w_embedding], dim=0)

        # 1st pass (source + tgt)
        self.controller.set_src_tgt_pass()
        epsilon_concat = self.model.apply_model(x_concat, ts_concat, c_concat, self.model.unet, w_cond=w_cond_combined)
        epsilon_pred_source, epsilon_pred_target = epsilon_concat.chunk(2)
        self.controller.reinit_layers()

        # 1st pass bis (source prompt with target latents)
        # self.controller.set_no_att_pass()
        # epsilon_pred_source = self.model.apply_model(x_target_t, ts, c_source, self.model.unet, w_cond=w_embedding)
        # self.controller.reinit_layers()

        # 2nd pass (mutual)
        self.controller.set_mutual_pass()
        epsilon_pred_mutual = self.model.apply_model(x_mutual_t, ts, c_source, self.model.unet, w_cond=w_embedding)
        self.controller.reinit_layers()

        # 3rd pass (tgt refine)
        # self.model.unet.diffusion_model.train()
        # for p in self.model.unet.diffusion_model.parameters():
        #     p.requires_grad = True

        # x_target_t = x_target_t.clone().detach().requires_grad_()
        # c_target.requires_grad = True
        # w_embedding.requires_grad = True

        self.controller.set_target_only()

        epsilon_pred_target_2 = self.model.apply_model(x_target_t, ts, c_target, self.model.unet, w_cond=w_embedding)

        
        epsilon_edit = self.eta*(epsilon_pred_target_2 - epsilon_pred_source) + epsilon_constrained    # shape 1, 20 (channels), t


        # if self.guiding_audio is not None and self._step_index<6:
        #     epsilon_guide = (x_source_t-alpha_prod_t.sqrt()*self.guiding_audio)/beta_prod_t.sqrt()
        #     mean_tensor = torch.mean(epsilon_guide, 2).unsqueeze(-1)
        #     epsilon_guide = torch.cat([mean_tensor for i in range(epsilon_edit.shape[2])], 2)
        #     epsilon_edit = 0.9*epsilon_edit + 0.1*epsilon_guide
            

        # # LOCAL BLEND 
        # M_src = torch.stack(self.controller.M_attention_src, dim=0).mean(dim=0) # shape b t c
        # M_tgt = torch.stack(self.controller.M_attention_tgt, dim=0).mean(dim=0)
        # M_src, M_tgt = M_src.mean(dim=0), M_tgt.mean(dim=0) # shape t c
        # # here should apply smthg like M_src*w_blended_src --> to get a shape  t ?
        # w_src = self.controller.w_blended_src.to(torch.bool)
        # w_tgt = self.controller.w_blended_tgt.to(torch.bool)
        # M_src = M_src[:,w_src].mean(dim=1)
        # M_tgt = M_tgt[:,w_tgt].mean(dim=1)#  shape t
        # mask_src = (M_src>=0.01).float()
        # mask_tgt = (M_tgt>=0.01).float() 
        # mask_src = mask_src.view(1, 1, -1)
        # mask_tgt = mask_tgt.view(1, 1, -1)
        # mask = mask_tgt-mask_src

        

        # 3. Get scalings for boundary conditions

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the edited epsilon
        x_target_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_edit) / alpha_prod_t.sqrt()
        x_source_0_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_pred_source) / alpha_prod_t.sqrt()
        x_mutual_0_pred = (x_mutual_t - beta_prod_t.sqrt() * epsilon_pred_mutual) / alpha_prod_t.sqrt()
        x_target_0_real_pred = (x_target_t - beta_prod_t.sqrt() * epsilon_pred_target) / alpha_prod_t.sqrt()

        if self.local_blend is not None:
            attention_store = torch.stack([torch.stack(self.controller.M_attention_src, dim=0).mean(dim=0), torch.stack(self.controller.M_attention_tgt, dim=0).mean(dim=0)], dim=0)
            x_mutual_0_pred, x_target_0_pred = self.local_blend(x_0, x_target_0_pred, x_mutual_0_pred, attention_store, alpha_prod_t)

        # DELTA DENOISING
        # if self._step_index>=3 and self._step_index<=7:
        #     grad = epsilon_pred_target_2 - epsilon_pred_source
        #     grad_fixed = grad.clone()
        #     x_0_opt = x_target_0_pred.clone().detach()
        #     x_0_opt.requires_grad = True
        #     optimizer = torch.optim.SGD(params=[x_0_opt], lr=1e-2)
            

        #     for l in range(1):
        #         optimizer.zero_grad()
        #         loss = x_0_opt * grad_fixed
        #         loss = loss.sum() / (x_0_opt.shape[2] * x_0_opt.shape[1])
        #         loss.backward()
        #         optimizer.step()

        #     x_target_0_pred = x_0_opt.clone().detach()


        self.controller.reinit_layers()
        self.controller.increment_step()
        self.controller.del_stored_qkv()

        # 5. Denoise model output using boundary conditions
        denoised_target = c_out * x_target_0_pred + c_skip * x_target_t
        denoised_source = c_out * x_0 + c_skip * x_source_t
        denoised_mutual = c_out * x_mutual_0_pred + c_skip * x_mutual_t
        denoised_source_pred = c_out * x_source_0_pred + x_source_t

        denoised_target_real = c_out * x_target_0_real_pred + c_skip * x_target_t

        if self.store_latents:
            self.latents.append((timestep, denoised_source_pred, denoised_target_real, denoised_source,denoised_target))
            self.epsilons.append((timestep,epsilon_pred_source, epsilon_pred_target_2, epsilon_constrained, epsilon_edit))


        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(epsilon_edit.shape, device=epsilon_edit.device)
            x_target_prev = alpha_prod_t_prev.sqrt() * denoised_target + beta_prod_t_prev.sqrt() * noise
            x_source_prev = alpha_prod_t_prev.sqrt() * denoised_source + beta_prod_t_prev.sqrt() * noise
            x_mutual_prev = alpha_prod_t_prev.sqrt() * denoised_mutual + beta_prod_t_prev.sqrt() * noise

        else:
            x_target_prev = denoised_target
            x_source_prev = denoised_source
            x_mutual_prev = denoised_mutual



        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred)

        return x_source_prev, x_target_prev, x_mutual_prev, x_target_0_pred