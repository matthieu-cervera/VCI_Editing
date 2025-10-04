import calendar
import matplotlib.pyplot as plt
import os
import time
import torch
import torchaudio
import warnings
import wandb
from torch import inference_mode
from tqdm import tqdm

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from utils import load_audio, get_spec, get_text_embeddings
from pc_drift import forward_directional



def edit_zeta(ldm_stable, eta, skip, cfg_scale_src, cfg_scale_tar, wandb_run, model_id, device, args):
        x0, sr, duration = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device,
                                    stft=('stable-audio' not in model_id), model_sr=ldm_stable.get_sr())
        torch.cuda.empty_cache()
        with inference_mode():
            w0 = ldm_stable.vae_encode(x0)

            # find Zs and wts - forward process
            if args.mode == "ddim":
                if len(cfg_scale_src) > 1:
                    raise ValueError("DDIM only supports one cfg_scale_src value")
                wT = ddim_inversion(ldm_stable, w0, args.source_prompt, cfg_scale_src[0],
                                    num_inference_steps=args.num_diffusion_steps, skip=skip[0])
            else:
                wt, zs, wts, extra_info = inversion_forward_process(
                    ldm_stable, w0, etas=eta,
                    prompts=args.source_prompt, cfg_scales=cfg_scale_src,
                    prog_bar=True,
                    num_inference_steps=args.num_diffusion_steps,
                    cutoff_points=args.cutoff_points,
                    numerical_fix=args.numerical_fix,
                    duration=duration)

            save_path = os.path.join(f'./{args.results_path}/')
            
            os.makedirs(save_path, exist_ok=True)

            if args.mode == "ours":
                # reverse process (via Zs and wT)
                w0, _ = inversion_reverse_process(ldm_stable,
                                                xT=wts if not args.test_rand_gen else torch.randn_like(wts),
                                                tstart=args.tstart,
                                                fix_alpha=args.fix_alpha,
                                                etas=eta, prompts=args.target_prompt,
                                                neg_prompts=args.target_neg_prompt,
                                                cfg_scales=cfg_scale_tar, prog_bar=True,
                                                zs=zs[:int(args.num_diffusion_steps - min(skip))]
                                                if not args.test_rand_gen else torch.randn_like(
                                                    zs[:int(args.num_diffusion_steps - min(skip))]),
                                                #   zs=zs[skip:],
                                                cutoff_points=args.cutoff_points,
                                                duration=duration,
                                                extra_info=extra_info)
            else:  # ddim
                if skip != 0:
                    warnings.warn("Plain DDIM Inversion should be run with t_start == num_diffusion_steps. "
                                "You are now running partial DDIM inversion.", RuntimeWarning)
                if len(cfg_scale_tar) > 1:
                    raise ValueError("DDIM only supports one cfg_scale_tar value")
                if len(args.source_prompt) > 1:
                    raise ValueError("DDIM only supports one args.source_prompt value")
                if len(args.target_prompt) > 1:
                    raise ValueError("DDIM only supports one args.target_prompt value")
                w0 = text2image_ldm_stable(ldm_stable, args.target_prompt,
                                        args.num_diffusion_steps, cfg_scale_tar[0],
                                        wT,
                                        skip=skip)

        # vae decode image
        with inference_mode():
            x0_dec = ldm_stable.vae_decode(w0)
            if 'stable-audio' not in model_id:
                if x0_dec.dim() < 4:
                    x0_dec = x0_dec[None, :, :, :]

                with torch.no_grad():
                    audio = ldm_stable.decode_to_mel(x0_dec)
                    orig_audio = ldm_stable.decode_to_mel(x0)
            else:
                audio = x0_dec.detach().clone().cpu().squeeze(0)
                orig_audio = x0.detach().clone().cpu()
                x0_dec = get_spec(x0_dec, ldm_stable.get_fn_STFT())
                x0 = get_spec(x0.unsqueeze(0), ldm_stable.get_fn_STFT())

                if x0_dec.dim() < 4:
                    x0_dec = x0_dec[None, :, :, :]
                    x0 = x0[None, :, :, :]

        image_name_png = os.path.basename(args.init_aud).split('.')[0]

        save_full_path_spec = os.path.join(save_path, image_name_png + ".png")
        save_full_path_wave = os.path.join(save_path, image_name_png + "_edited.wav")
        save_full_path_origwave = os.path.join(save_path, "orig.wav")
        if x0_dec.shape[2] > x0_dec.shape[3]:
            x0_dec = x0_dec[0, 0].T.cpu().detach().numpy()
            x0 = x0[0, 0].T.cpu().detach().numpy()
        else:
            x0_dec = x0_dec[0, 0].cpu().detach().numpy()
            x0 = x0[0, 0].cpu().detach().numpy()

        #plt.imsave(save_full_path_spec, x0_dec)
        torchaudio.save(save_full_path_wave, audio, sample_rate=sr)



def edit_sdedit(ldm_stable, skip, device, args):
    with torch.no_grad():
        x0, sr, duration = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device, stft=True)
    torch.cuda.empty_cache()

    with inference_mode(), torch.no_grad():
        w0 = ldm_stable.vae_encode(x0)

        text_embeddings_class_labels, text_emb, uncond_emb = get_text_embeddings(
            args.target_prompt, args.target_neg_prompt, ldm_stable)

    timesteps = ldm_stable.model.scheduler.timesteps
    latents = []
    for _ in range(len(timesteps) + 1):
        shape = (1, ldm_stable.model.unet.config.in_channels, w0.shape[2],
                 ldm_stable.model.vocoder.config.model_in_dim // ldm_stable.model.vae_scale_factor)
        lat = torch.randn(shape, device=device, dtype=w0.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        lat = lat * ldm_stable.model.scheduler.init_noise_sigma
        latents.append(lat)

    timesteps = timesteps[skip:]
    latents = latents[skip + 1:]

    noise = torch.randn_like(w0, device=device)
    xt = ldm_stable.model.scheduler.add_noise(w0, noise, timesteps[:1].unsqueeze(0))

    del noise, w0

    for it, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        xt, _ = forward_directional(
            ldm_stable, xt, t, latents[it], uncond_emb, text_emb, args.cfg_tar[0],
            eta=args.eta)

    del latents, uncond_emb, text_emb
    torch.cuda.empty_cache()

    with inference_mode():
        x0_dec = ldm_stable.vae_decode(xt)
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)

    save_path = os.path.join(f'./{args.results_path}/')
    os.makedirs(save_path, exist_ok=True)

    image_name_png = os.path.basename(args.init_aud).split('.')[0]

    save_full_path_spec = os.path.join(save_path, image_name_png + ".png")
    save_full_path_wave = os.path.join(save_path, image_name_png + "_edited.wav")

    #plt.imsave(save_full_path_spec, x0_dec[0, 0].T.cpu().detach().numpy())
    torchaudio.save(save_full_path_wave, audio, sample_rate=16000)