'''
heavily inspired by https://github.com/sled-group/InfEdit 

functions to build custom controller for attention manipulations and register the controller in the model
We didn't test enough the attention control mechanism for audio editing, there can be implementation errors
'''
from __future__ import annotations
import torch
import abc
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch.nn.functional as nnf
import seq_aligner
from transformers import T5Tokenizer, T5EncoderModel
from transformers import RobertaTokenizer, ClapModel


LOW_RESOURCE = False
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_NUM_WORDS = 154

'''
AudioLCM doesn't build on transformers pipelines
'''
class AttentionControlLCM:
    def register_attention_control(model, controller):
        def register_recr(net_, count, place_in_unet):
            for idx, m in enumerate(net_.modules()):
                if m.__class__.__name__ == "CrossAttention":
                    count+=1
                    m.controller = controller# AttnProcessor(place_in_unet)
                    m.place_in_unet = place_in_unet
            return count

        
        depth = len(model.unet.diffusion_model.blocks)
        cross_att_count = 0
        for idx, block in enumerate(model.unet.diffusion_model.blocks):
            if idx < depth // 3:
                place_in_unet = "down"
            elif idx < 2 * depth // 3:
                place_in_unet = "mid"
            else:
                place_in_unet = "up"
            cross_att_count += register_recr(block, 0, place_in_unet=place_in_unet)

        print(f"nbs of cross attention blocks : {cross_att_count}")
        controller.num_att_layers = cross_att_count


    class AttentionManipulator:
        def __init__(self, x_0, tau_s, tau_c, total_steps, alignment_fn, T_shape, w_blended_src, w_blended_tgt):
            super(AttentionControlLCM.AttentionManipulator, self).__init__()
            self.tau_s = tau_s * total_steps
            self.tau_c = tau_c * total_steps
            self.fake_total_steps = total_steps*8
            self.B = x_0.shape[0]
            self.T = T_shape
            self.alignment_fn = alignment_fn  # dict: tgt_idx -> src_idx
            self.cur_step = 0
            self.cur_layer = -1
            self.is_mutual_pass = False         # is it a forward pass on the mutual branch
            self.is_target_only = False         # is it the last forward pass on the tgt branch
            self.cond_len = 0
            self.attention_pass = True

            self.w_blended_src = w_blended_src
            self.w_blended_tgt = w_blended_tgt

            self.src_qkv = []
            self.tgt_qkv = []
            self.mutual_attention = []
            self.M_attention_src = []
            self.M_attention_tgt = []
            self.is_matt = True
        
        # set which input is in forward pass
        def set_src_tgt_pass(self):
            self.is_mutual_pass = False
            self.is_target_only = False
            self.attention_pass = True

        def set_mutual_pass(self):
            self.is_mutual_pass = True
            self.is_target_only = False
            self.attention_pass = True

        def set_target_only(self):
            self.is_target_only = True
            self.is_mutual_pass = False
            self.attention_pass = True

        def set_no_att_pass(self):
            self.attention_pass = False


        def increment_step(self):
            self.cur_step += 1

        def reinit_layers(self):
            self.cur_layer = -1
        
        def del_stored_qkv(self):
            self.src_qkv = []
            self.tgt_qkv = []
            self.mutual_attention = []
            self.M_attention_src = []
            self.M_attention_tgt = []

        def new_refine_crossattention_light(self, m_tgt):
            refined = m_tgt.clone()
            w_src = self.w_blended_src  # shape cond_len
            w_tgt = self.w_blended_tgt

            if self.cur_step<=self.tau_c:
                mutual_attention_avg = torch.stack(self.mutual_attention, dim=0).mean(dim=0)            # shape (heads, T, cond_len-1)
                j_idx_align = torch.tensor([j for j in range(mutual_attention_avg.shape[2]) if 
                                            j < len(self.alignment_fn) and w_src[j]>0 and self.alignment_fn[j]>=0], dtype=torch.long)
                src_idx = torch.tensor([self.alignment_fn[j] for j in j_idx_align.tolist()], dtype=torch.long)
                j_idx_enhance = torch.tensor([j for j in range(mutual_attention_avg.shape[2]) if 
                                            j < len(self.alignment_fn) and w_tgt[j]>0 and self.alignment_fn[j]>=0], dtype=torch.long)
                            
                refined[:,:mutual_attention_avg.shape[1],j_idx_align+1]=mutual_attention_avg[:,:, src_idx]
                refined[:,:,j_idx_enhance+1] *= 4
            return refined 


        def new_forward(self, q, k, v, scale):
            self.cur_layer += 1

            seq_len = q.shape[1]
            cond_len = seq_len - self.T
            self.cond_len = cond_len
            q_time, k_time, v_time = q[:,0:1,:], k[:,0:1:,], v[:,0:1,:]

            if self.is_mutual_pass: # mutual only forward pass --> self att control
                if (self.cur_step<=self.tau_s) and self.cur_layer>=0:
                    q_src_self, k_src_self, v_src_self = self.src_qkv[self.cur_layer][1]        # 0: fake cross 1: self but  "real cross" is q_self, k_cross, v_cross
                    q_src_cross, k_src_cross, v_src_cross = self.src_qkv[self.cur_layer][0]
                    sim = torch.einsum('b i d, b j d -> b i j', q_src_self, k_src_cross) * scale
                    self.mutual_attention.append(sim.softmax(dim=-1))
                    return torch.cat([q_time,q_src_cross,q_src_self], dim=1), torch.cat([k_time, k_src_cross,k_src_self],dim=1), torch.cat([v_time, v_src_cross,v_src_self],dim=1)
                
                else:
                    q_tgt_self, _,_ = self.tgt_qkv[self.cur_layer][1]
                    q_tgt_cross,_,_ = self.tgt_qkv[self.cur_layer][0]
                    q_src_self, k_src_self, v_src_self = self.src_qkv[self.cur_layer][1]
                    q_src_cross, k_src_cross, v_src_cross = self.src_qkv[self.cur_layer][0]
                    sim = torch.einsum('b i d, b j d -> b i j', q_tgt_self, k_src_cross) * scale
                    self.mutual_attention.append(sim.softmax(dim=-1))
                    
                    return torch.cat([q_time, q_src_cross, q_tgt_self], dim=1), torch.cat([k_time, k_src_cross,k_src_self],dim=1), torch.cat([v_time, v_src_cross,v_src_self],dim=1)
            
            else:         # src & tgt pass --> meaning we must not change q,k,v but store them for each layer
                b = q.shape[0]//2
                q_src_cross, q_src_self, q_tgt_cross, q_tgt_self = q[:b,1:cond_len,:], q[:b,cond_len:,:],q[b:,1:cond_len,:], q[b:,cond_len:,:]
                k_src_cross, k_src_self, k_tgt_cross, k_tgt_self = k[:b,1:cond_len,:], k[:b,cond_len:,:],k[b:,1:cond_len,:], k[b:,cond_len:,:]
                v_src_cross, v_src_self, v_tgt_cross, v_tgt_self = v[:b,1:cond_len,:],v[:b,cond_len:,:],v[b:,1:cond_len,:], v[b:,cond_len:,:]

                self.src_qkv.append([(q_src_cross, k_src_cross, v_src_cross),(q_src_self, k_src_self, v_src_self)])
                self.tgt_qkv.append([(q_tgt_cross, k_tgt_cross, v_tgt_cross),(q_tgt_self,k_tgt_self,v_tgt_self)])

                sim_src = torch.einsum('b i d, b j d -> b i j', q_src_self, k_src_cross) * scale
                sim_tgt = torch.einsum('b i d, b j d -> b i j', q_tgt_self, k_tgt_cross) * scale
                self.M_attention_src.append(sim_src.softmax(dim=-1))
                self.M_attention_tgt.append(sim_tgt.softmax(dim=-1))
                return q,k,v


        def forward(self, q, k, v, scale):
            self.cur_layer += 1

            if self.is_mutual_pass: # mutual only forward pass --> self att control
                if (self.cur_step*8+self.cur_layer)/self.fake_total_steps>=self.tau_s/8:
                    q_src, k_src, v_src = self.src_qkv[self.cur_layer]
                    sim = torch.einsum('b i d, b j d -> b i j', q_src, k_src) * scale
                    self.mutual_attention.append(sim.softmax(dim=-1))
                    return q_src, k_src, v_src
                else:
                    q_tgt, _,_ = self.tgt_qkv[self.cur_layer]
                    _, k_src, v_src = self.src_qkv[self.cur_layer]
                    sim = torch.einsum('b i d, b j d -> b i j', q_tgt, k_src) * scale
                    self.mutual_attention.append(sim.softmax(dim=-1))
                    
                    return q_tgt, k_src, v_src
            
            else:         # src & tgt pass --> meaning we must not change q,k,v but store them for each layer
                b = q.shape[0]//2
                q_src, q_tgt = q[:b], q[b:]
                k_src, k_tgt = k[:b], k[b:]
                v_src, v_tgt = v[:b], v[b:]
                self.src_qkv.append((q_src, k_src, v_src))
                self.tgt_qkv.append((q_tgt, k_tgt, v_tgt))
                return q,k,v  

    class LocalBlend:
        
        def get_mask(self,x_t,maps,word_idx, thresh):
            maps = maps * word_idx.reshape(1, 1, 1, -1)
            maps = maps[:, :, :, 1:self.len - 1].mean(0, keepdim=True)
            maps = maps.max(-1)[0]
            maps = nnf.interpolate(maps, size=(x_t.shape[-1],), mode='linear', align_corners=False)
            maps = maps / maps.max(-1, keepdim=True)[0]
            mask = maps > thresh
            return mask
        
        def get_map(self,x_t,maps,word_idx, thresh):
            maps = maps * word_idx.reshape(1, 1, 1, -1)
            maps = maps[:, :, :, :].mean(0, keepdim=True)
            maps = maps.max(-1)[0]
            maps = nnf.interpolate(maps, size=(x_t.shape[-1],), mode='linear', align_corners=False)
            maps = maps / maps.max(-1, keepdim=True)[0]
            return maps

        def __call__(self, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
            maps = attention_store
            C,T = x_t.shape[1], x_t.shape[2]
            maps = maps.unsqueeze(2) 
            maps_s = maps[0,:]
            maps_m = maps[1,:]
            thresh_e = temperature / alpha_prod ** (0.5)
            if thresh_e < self.thresh_e:
                thresh_e = self.thresh_e
            thresh_m = self.thresh_m
            mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e)
            
            if self.compute_second_mask:
                mask_m = self.get_mask(x_t, maps_s, (self.alpha_m-self.alpha_me), thresh_m)
            else:
                mask_m = ~mask_e

            if self.alpha_e.sum() == 0:
                x_t_out = x_t
            else:
                x_t_out = torch.where(mask_e, x_t, x_m)
                x_t_out = torch.where(mask_m, x_s, x_t_out)
            return x_m, x_t_out

        def __init__(self,thresh_e=0.3, thresh_m=0.3, compute_second_mask=False, save_inter = False):
            self.thresh_e = thresh_e
            self.thresh_m = thresh_m
            self.compute_second_mask = compute_second_mask
            self.save_inter = save_inter
            self.MAX_NUM_WORDS = 154

            
        def set_map(self, ms, alpha, alpha_e, alpha_m,len):
            self.m = ms.to(device)
            self.alpha = alpha.to(device)
            self.alpha_e = alpha_e.to(device)
            self.alpha_m = alpha_m.to(device)
            alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
            self.alpha_me = alpha_me.to(torch.float).to(device)
            self.len = len

class AttentionControlLDM:

    class LocalBlend:
        def get_mask(self,x_t,maps,word_idx, thresh, i):
            maps = maps * word_idx.reshape(1,1,1,1,-1)
            maps = (maps[:,:,:,:,1:self.len-1]).mean(0,keepdim=True)
            maps = (maps).max(-1)[0]
            maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
            maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
            mask = maps > thresh
            return mask        

        def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            h,w = x_t.shape[2],x_t.shape[3]
            h , w = ((h+1)//2+1)//2, ((w+1)//2+1)//2
            maps = [item.reshape(2, -1, 1, h // int((h*w/item.shape[-2])**0.5),  w // int((h*w/item.shape[-2])**0.5), MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            maps_s = maps[0,:]
            maps_m = maps[1,:]
            thresh_e = temperature / alpha_prod ** (0.5)
            if thresh_e < self.thresh_e:
                thresh_e = self.thresh_e
            thresh_m = self.thresh_m
            mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
            mask_m = self.get_mask(x_t, maps_s, (self.alpha_m-self.alpha_me), thresh_m, i)
            mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)

            if self.alpha_e.sum() == 0:
                x_t_out = x_t
            else:
                x_t_out = torch.where(mask_e, x_t, x_m)
                x_t_out = torch.where(mask_m, x_s, x_t_out)
            if use_xm:
                x_t_out = torch.where(mask_me, x_m, x_t_out)
            
            return x_m, x_t_out

        def __init__(self,thresh_e=0.3, thresh_m=0.3, save_inter = False):
            self.thresh_e = thresh_e
            self.thresh_m = thresh_m
            self.save_inter = save_inter
            
        def set_map(self, ms, alpha, alpha_e, alpha_m,len):
            self.m = ms
            self.alpha = alpha
            self.alpha_e = alpha_e
            self.alpha_m = alpha_m
            alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
            self.alpha_me = alpha_me.to(torch.float)
            self.len = len


    class AttentionControl(abc.ABC):

        def step_callback(self, x_t):
            return x_t

        def between_steps(self):
            return

        @property
        def num_uncond_att_layers(self):
            return self.num_att_layers if LOW_RESOURCE else 0

        @abc.abstractmethod
        def forward(self, attn, is_cross: bool, place_in_unet: str):
            raise NotImplementedError

        def __call__(self, attn, is_cross: bool, place_in_unet: str):
            if self.cur_att_layer >= self.num_uncond_att_layers:
                if LOW_RESOURCE:
                    attn = self.forward(attn, is_cross, place_in_unet)
                else:
                    h = attn.shape[0]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return attn

        def reset(self):
            self.cur_step = 0
            self.cur_att_layer = 0

        def __init__(self):
            self.cur_step = 0
            self.num_att_layers = -1
            self.cur_att_layer = 0


    class EmptyControl(AttentionControl):

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            return attn
        def self_attn_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
            print("empty control")
            b = q.shape[0] // num_heads
            out = torch.einsum("h i j, h j d -> h i d", attn, v)
            return out


    class AttentionStore(AttentionControl):

        @staticmethod
        def get_empty_store():
            return {"down_cross": [], "mid_cross": [], "up_cross": [],
                    "down_self": [],  "mid_self": [],  "up_self": []}

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn)
            return attn

        def between_steps(self):
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

        def get_average_attention(self):
            average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
            return average_attention

        def reset(self):
            super(AttentionControlLDM.AttentionStore, self).reset()
            self.step_store = self.get_empty_store()
            self.attention_store = {}

        def __init__(self):
            super(AttentionControlLDM.AttentionStore, self).__init__()
            self.step_store = self.get_empty_store()
            self.attention_store = {}


    class AttentionControlEdit(AttentionStore, abc.ABC):

        def step_callback(self,i, t, x_s, x_t, x_m, alpha_prod):
            if (self.local_blend is not None) and (i>0):
                use_xm = (self.cur_step+self.start_steps+1 == self.num_steps)
                x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
            return x_m, x_t

        def replace_self_attention(self, attn_base, att_replace):
            if att_replace.shape[2] <= 16 ** 2:
                return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            else:
                return att_replace

        @abc.abstractmethod
        def replace_cross_attention(self, attn_base, att_replace):
            raise NotImplementedError
        
        def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
            b = q.shape[0] // num_heads

            sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
            attn = sim.softmax(-1)
            out = torch.einsum("h i j, h j d -> h i d", attn, v)
            return out
        
        def self_attn_forward(self, q, k, v, num_heads):
            if q.shape[0]//num_heads == 3:
                if (self.self_replace_steps <= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                    q=torch.cat([q[:num_heads*2],q[num_heads:num_heads*2]])
                    k=torch.cat([k[:num_heads*2],k[:num_heads]])
                    v=torch.cat([v[:num_heads*2],v[:num_heads]])
                else:
                    q=torch.cat([q[:num_heads],q[:num_heads],q[:num_heads]])
                    k=torch.cat([k[:num_heads],k[:num_heads],k[:num_heads]])
                    v=torch.cat([v[:num_heads*2],v[:num_heads]])
                return q,k,v
            else:
                qu, qc = q.chunk(2)
                ku, kc = k.chunk(2)
                vu, vc = v.chunk(2)
                if (self.self_replace_steps <= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                    qu=torch.cat([qu[:num_heads*2],qu[num_heads:num_heads*2]])
                    qc=torch.cat([qc[:num_heads*2],qc[num_heads:num_heads*2]])
                    ku=torch.cat([ku[:num_heads*2],ku[:num_heads]])
                    kc=torch.cat([kc[:num_heads*2],kc[:num_heads]])
                    vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                    vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])
                else:
                    qu=torch.cat([qu[:num_heads],qu[:num_heads],qu[:num_heads]])
                    qc=torch.cat([qc[:num_heads],qc[:num_heads],qc[:num_heads]])
                    ku=torch.cat([ku[:num_heads],ku[:num_heads],ku[:num_heads]])
                    kc=torch.cat([kc[:num_heads],kc[:num_heads],kc[:num_heads]])
                    vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                    vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])

                return torch.cat([qu, qc], dim=0) ,torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            if is_cross :
                h = attn.shape[0] // self.batch_size
                attn = attn.reshape(self.batch_size,h,  *attn.shape[1:])
                attn_base, attn_replace, attn_masa = attn[0], attn[1], attn[2]
                attn_replace_new = self.replace_cross_attention(attn_masa, attn_replace) 
                attn_base_store = self.replace_cross_attention(attn_base, attn_replace)
                if (self.cross_replace_steps >= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                    attn[1] = attn_replace_new
                attn_store=torch.cat([attn_base_store,attn_replace_new])
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
                attn_store = attn_store.reshape(2 *h, *attn_store.shape[2:])
                super(AttentionControlLDM.AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
            return attn

        def __init__(self, prompts, num_steps: int,start_steps: int,
                    cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                    self_replace_steps: Union[float, Tuple[float, float]],
                    local_blend: Optional[AttentionControlLDM.LocalBlend]):
            super(AttentionControlLDM.AttentionControlEdit, self).__init__()
            self.batch_size = len(prompts)+1
            self.self_replace_steps = self_replace_steps
            self.cross_replace_steps = cross_replace_steps
            self.num_steps=num_steps
            self.start_steps=start_steps
            self.local_blend = local_blend


    class AttentionReplace(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, att_replace):
            return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    local_blend: Optional[AttentionControlLDM.LocalBlend] = None):
            super(AttentionControlLDM.AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            self.mapper = seq_aligner.get_replacement_mapper(prompts, self.tokenizer).to(device).to(torch_dtype)


    class AttentionRefine(AttentionControlEdit):

        def replace_cross_attention(self, attn_masa, att_replace):
            print(self.mapper, attn_masa.shape, self.alphas.shape)
            attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
            attn_replace = attn_masa_replace * self.alphas + \
                    att_replace * (1 - self.alphas)
            return attn_replace

        def __init__(self, prompts, prompt_specifiers, num_steps: int,start_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    tokenizer: T5Tokenizer, encoder: T5EncoderModel, local_blend: Optional[AttentionControlLDM.LocalBlend] = None, 
                    encoder_type: Optional[str] = "FLAN", max_len: Optional[int] = 77):
            
            super(AttentionControlLDM.AttentionRefine, self).__init__(prompts, num_steps,start_steps, cross_replace_steps, self_replace_steps, local_blend)
            self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers, tokenizer, encoder, 
                                                                                        device, encoder_type=encoder_type, max_len=max_len)
            self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(torch_dtype)
            self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
            self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
            self.tokenizer = tokenizer
            self.encoder = encoder
            ms = ms.to(device)
            alpha_e = alpha_e.to(device)
            alpha_m = alpha_m.to(device)
            t_len = len(tokenizer(prompts[1])["input_ids"])
            self.local_blend.set_map(ms,alphas,alpha_e,alpha_m,t_len)

        def set_src_tgt_pass(self):
            pass

        def set_mutual_pass(self):
            pass

        def set_target_only(self):
            pass

        def set_no_att_pass(self):
            pass

        def increment_step(self):
            self.cur_step += 1

        def reinit_layers(self):
            self.cur_layer = -1
        
        def del_stored_qkv(self):
            pass


    def register_attention_control(model, controller):
        model._original_attention_processors = {}

        class AttnProcessor():
            def __init__(self,place_in_unet):
                self.place_in_unet = place_in_unet

            def __call__(self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                scale=1.0,):
                # The `Attention` class can call different attention processors / attention functions
        
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                h = attn.heads
                is_cross = encoder_hidden_states is not None
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                q = attn.to_q(hidden_states)
                k = attn.to_k(encoder_hidden_states)
                v = attn.to_v(encoder_hidden_states)
                q = attn.head_to_batch_dim(q)
                k = attn.head_to_batch_dim(k)
                v = attn.head_to_batch_dim(v)


                if not is_cross:
                    q,k,v = controller.self_attn_forward(q, k, v, attn.heads)


                attention_probs = attn.get_attention_scores(q, k, attention_mask)
                if is_cross:
                    attention_probs  = controller(attention_probs , is_cross, self.place_in_unet)
                

                hidden_states = torch.bmm(attention_probs, v)
                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj   
                hidden_states = attn.to_out[0](hidden_states)       #, scale=scale
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states
            
        def register_recr(net_, count, place_in_unet):
            for idx, m in enumerate(net_.modules()):
                # print(m.__class__.__name__)
                if m.__class__.__name__ == "Attention":
                    count+=1
                    model._original_attention_processors[id(m)] = m.processor
                    m.processor = AttnProcessor( place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
        controller.num_att_layers = cross_att_count


    def unregister_attention_control(model):
        if not hasattr(model, '_original_attention_processors'):
            print("No attention controllers to unregister.")
            return

        for net in model.unet.modules():
            if net.__class__.__name__ == "Attention":
                if id(net) in model._original_attention_processors:
                    net.processor = model._original_attention_processors[id(net)]

        del model._original_attention_processors

class AttentionControlLDM2:
    class LocalBlend:
    
        def get_mask(self,x_t,maps,word_idx, thresh, i):
            maps = maps * word_idx.reshape(1,1,1,1,-1)
            maps = (maps[:,:,:,:,1:self.len-1]).mean(0,keepdim=True)
            maps = (maps).max(-1)[0]
            maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
            maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
            mask = maps > thresh
            return mask        

        def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            h,w = x_t.shape[2],x_t.shape[3]
            h , w = ((h+1)//2+1)//2, ((w+1)//2+1)//2
            maps = [item.reshape(2, -1, 1, h // int((h*w/item.shape[-2])**0.5),  w // int((h*w/item.shape[-2])**0.5), MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            maps_s = maps[0,:]
            maps_m = maps[1,:]
            thresh_e = temperature / alpha_prod ** (0.5)
            if thresh_e < self.thresh_e:
                thresh_e = self.thresh_e
            thresh_m = self.thresh_m
            mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
            mask_m = self.get_mask(x_t, maps_s, (self.alpha_m-self.alpha_me), thresh_m, i)
            mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)

            if self.alpha_e.sum() == 0:
                x_t_out = x_t
            else:
                x_t_out = torch.where(mask_e, x_t, x_m)
                x_t_out = torch.where(mask_m, x_s, x_t_out)
            if use_xm:
                x_t_out = torch.where(mask_me, x_m, x_t_out)
            
            return x_m, x_t_out

        def __init__(self,thresh_e=0.3, thresh_m=0.3, save_inter = False):
            self.thresh_e = thresh_e
            self.thresh_m = thresh_m
            self.save_inter = save_inter
            
        def set_map(self, ms, alpha, alpha_e, alpha_m,len):
            self.m = ms
            self.alpha = alpha
            self.alpha_e = alpha_e
            self.alpha_m = alpha_m
            alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
            self.alpha_me = alpha_me.to(torch.float)
            self.len = len


    class AttentionControl(abc.ABC):

        def step_callback(self, x_t):
            return x_t

        def between_steps(self):
            return

        @property
        def num_uncond_att_layers(self):
            return self.num_att_layers if LOW_RESOURCE else 0

        @abc.abstractmethod
        def forward(self, attn, is_cross: bool, place_in_unet: str):
            raise NotImplementedError

        def __call__(self, attn, is_cross: bool, place_in_unet: str):
            if self.cur_att_layer >= self.num_uncond_att_layers:
                if LOW_RESOURCE:
                    attn = self.forward(attn, is_cross, place_in_unet)
                else:
                    h = attn.shape[0]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return attn

        def reset(self):
            self.cur_step = 0
            self.cur_att_layer = 0

        def __init__(self):
            self.cur_step = 0
            self.num_att_layers = -1
            self.cur_att_layer = 0


    class EmptyControl(AttentionControl):

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            return attn
        def self_attn_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
            print("empty control")
            b = q.shape[0] // num_heads
            out = torch.einsum("h i j, h j d -> h i d", attn, v)
            return out


    class AttentionStore(AttentionControl):

        @staticmethod
        def get_empty_store():
            return {"down_cross": [], "mid_cross": [], "up_cross": [],
                    "down_self": [],  "mid_self": [],  "up_self": []}

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn)
            return attn

        def between_steps(self):
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

        def get_average_attention(self):
            average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
            return average_attention

        def reset(self):
            super(AttentionControlLDM2.AttentionStore, self).reset()
            self.step_store = self.get_empty_store()
            self.attention_store = {}

        def __init__(self):
            super(AttentionControlLDM2.AttentionStore, self).__init__()
            self.step_store = self.get_empty_store()
            self.attention_store = {}


    class AttentionControlEdit(AttentionStore, abc.ABC):

        def step_callback(self,i, t, x_s, x_t, x_m, alpha_prod):
            if (self.local_blend is not None) and (i>0):
                use_xm = (self.cur_step+self.start_steps+1 == self.num_steps)
                x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
            return x_m, x_t

        def replace_self_attention(self, attn_base, att_replace):
            if att_replace.shape[2] <= 16 ** 2:
                return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            else:
                return att_replace

        @abc.abstractmethod
        def replace_cross_attention(self, attn_base, att_replace):
            raise NotImplementedError
        
        def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
            b = q.shape[0] // num_heads

            sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
            attn = sim.softmax(-1)
            out = torch.einsum("h i j, h j d -> h i d", attn, v)
            return out
        
        def self_attn_forward(self, q, k, v, num_heads):
            if q.shape[0]//num_heads == 3:
                if (self.self_replace_steps <= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                    q=torch.cat([q[:num_heads*2],q[num_heads:num_heads*2]])
                    k=torch.cat([k[:num_heads*2],k[:num_heads]])
                    v=torch.cat([v[:num_heads*2],v[:num_heads]])
                else:
                    q=torch.cat([q[:num_heads],q[:num_heads],q[:num_heads]])
                    k=torch.cat([k[:num_heads],k[:num_heads],k[:num_heads]])
                    v=torch.cat([v[:num_heads*2],v[:num_heads]])
                return q,k,v
            else:
                qu, qc = q.chunk(2)
                ku, kc = k.chunk(2)
                vu, vc = v.chunk(2)
                if (self.self_replace_steps <= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                    qu=torch.cat([qu[:num_heads*2],qu[num_heads:num_heads*2]])
                    qc=torch.cat([qc[:num_heads*2],qc[num_heads:num_heads*2]])
                    ku=torch.cat([ku[:num_heads*2],ku[:num_heads]])
                    kc=torch.cat([kc[:num_heads*2],kc[:num_heads]])
                    vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                    vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])
                else:
                    qu=torch.cat([qu[:num_heads],qu[:num_heads],qu[:num_heads]])
                    qc=torch.cat([qc[:num_heads],qc[:num_heads],qc[:num_heads]])
                    ku=torch.cat([ku[:num_heads],ku[:num_heads],ku[:num_heads]])
                    kc=torch.cat([kc[:num_heads],kc[:num_heads],kc[:num_heads]])
                    vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                    vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])

                return torch.cat([qu, qc], dim=0) ,torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

        def forward(self, attn, is_cross: bool, place_in_unet: str):
            if is_cross and attn.shape[2] !=8:
                h = attn.shape[0] // self.batch_size
                attn = attn.reshape(self.batch_size,h,  *attn.shape[1:])
                attn_base, attn_replace, attn_masa = attn[0], attn[1], attn[2]
                attn_replace_new = self.replace_cross_attention(attn_masa, attn_replace) 
                attn_base_store = self.replace_cross_attention(attn_base, attn_replace)
                if (self.cross_replace_steps >= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                    attn[1] = attn_replace_new

                attn_store=torch.cat([attn_base_store,attn_replace_new])
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
                attn_store = attn_store.reshape(2 *h, *attn_store.shape[2:])
                super(AttentionControlLDM2.AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
            return attn

        def __init__(self, prompts, num_steps: int,start_steps: int,
                    cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                    self_replace_steps: Union[float, Tuple[float, float]],
                    local_blend: Optional[AttentionControlLDM2.LocalBlend]):
            super(AttentionControlLDM2.AttentionControlEdit, self).__init__()
            self.batch_size = len(prompts)+1
            self.self_replace_steps = self_replace_steps
            self.cross_replace_steps = cross_replace_steps
            self.num_steps=num_steps
            self.start_steps=start_steps
            self.local_blend = local_blend


    class AttentionReplace(AttentionControlEdit):

        def replace_cross_attention(self, attn_base, att_replace):
            return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

        def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    local_blend: Optional[AttentionControlLDM2.LocalBlend] = None):
            super(AttentionControlLDM2.AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
            self.mapper = seq_aligner.get_replacement_mapper(prompts, self.tokenizer).to(device).to(torch_dtype)


    class AttentionRefine(AttentionControlEdit):

        def replace_cross_attention(self, attn_masa, att_replace):
            
            if attn_masa.shape[2] == 8 : # CLAP cond
                mapper = self.mapper_clap
                alphas = self.alphas_clap
                return att_replace
            else : # t5 cond
                mapper = self.mapper
                alphas = self.alphas
                # print(mapper, attn_masa.shape, att_replace.shape, alphas)

                attn_masa_replace = attn_masa[:, :, mapper].squeeze()
                attn_replace = attn_masa_replace * alphas + \
                        att_replace * (1 - alphas)
                return attn_replace

        def __init__(self, prompts, prompt_specifiers, num_steps: int,start_steps: int, cross_replace_steps: float, self_replace_steps: float,
                    tokenizer: T5Tokenizer, encoder: T5EncoderModel, clap_tokenizer: RobertaTokenizer, clap_encoder:ClapModel, 
                    local_blend: Optional[AttentionControlLDM2.LocalBlend] = None):
            
            super(AttentionControlLDM2.AttentionRefine, self).__init__(prompts, num_steps,start_steps, cross_replace_steps, self_replace_steps, local_blend)

            self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers, tokenizer, encoder, 
                                                                                        device, encoder_type="FLAN", max_len=128)
            self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(torch_dtype)

            # self.mapper_clap, alphas_clap, ms_clap, alpha_e_clap, alpha_m_clap = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers, clap_tokenizer, 
            #                                                                                                        clap_encoder, device, encoder_type="CLAP_AudioLDM2", max_len=128)
            # self.mapper_clap, alphas_clap, ms_clap = self.mapper_clap.to(device), alphas_clap.to(device).to(torch_dtype), ms_clap.to(device).to(torch_dtype)


            self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
            # self.alphas_clap = alphas_clap.reshape(alphas_clap.shape[0], 1, 1, alphas_clap.shape[1])

            self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
            self.tokenizer = tokenizer
            self.encoder = encoder
            self.clap_tokenizer = clap_tokenizer
            self.clap_encoder = clap_encoder
            ms = ms.to(device)
            alpha_e = alpha_e.to(device)
            alpha_m = alpha_m.to(device)
            t_len = len(tokenizer(prompts[1])["input_ids"])
            self.local_blend.set_map(ms,alphas,alpha_e,alpha_m,t_len)

        def set_src_tgt_pass(self):
            pass

        def set_mutual_pass(self):
            pass

        def set_target_only(self):
            pass

        def set_no_att_pass(self):
            pass

        def increment_step(self):
            self.cur_step += 1

        def reinit_layers(self):
            self.cur_layer = -1
        
        def del_stored_qkv(self):
            pass


    def register_attention_control(model, controller):
        model._original_attention_processors = {}

        class AttnProcessor():
            def __init__(self,place_in_unet):
                self.place_in_unet = place_in_unet

            def __call__(self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                scale=1.0,):
                # The `Attention` class can call different attention processors / attention functions
        
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                h = attn.heads
                is_cross = encoder_hidden_states is not None

                is_clap = is_cross and (encoder_hidden_states.shape[1] == 8)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                q = attn.to_q(hidden_states)
                k = attn.to_k(encoder_hidden_states)
                v = attn.to_v(encoder_hidden_states)
                q = attn.head_to_batch_dim(q)
                k = attn.head_to_batch_dim(k)
                v = attn.head_to_batch_dim(v)


                if not is_cross and not is_clap:
                    q,k,v = controller.self_attn_forward(q, k, v, attn.heads)


                attention_probs = attn.get_attention_scores(q, k, attention_mask)
                if is_cross and not is_clap:
                    attention_probs  = controller(attention_probs , is_cross, self.place_in_unet)
                

                hidden_states = torch.bmm(attention_probs, v)
                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj   
                hidden_states = attn.to_out[0](hidden_states)       #, scale=scale
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states
            
        def register_recr(net_, count, place_in_unet):
            for idx, m in enumerate(net_.modules()):
                # print(m.__class__.__name__)
                if m.__class__.__name__ == "Attention":
                    count+=1
                    model._original_attention_processors[id(m)] = m.processor
                    m.processor = AttnProcessor( place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
        controller.num_att_layers = cross_att_count


    def unregister_attention_control(model):
        if not hasattr(model, '_original_attention_processors'):
            print("No attention controllers to unregister.")
            return

        for net in model.unet.modules():
            if net.__class__.__name__ == "Attention":
                if id(net) in model._original_attention_processors:
                    net.processor = model._original_attention_processors[id(net)]

        del model._original_attention_processors