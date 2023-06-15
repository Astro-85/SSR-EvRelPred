"""
Model for EvRel
"""
import sys
import random
import copy
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from vidsitu_code.mdl_sf_base import get_head_dim
from transformers import RobertaForSequenceClassification, RobertaModel

from HERO.model.model import HeroModel
from HERO.model.model import VideoModelConfig
from HERO.model.layers import MLPLayer
from HERO.model.modeling_utils import mask_logits


class Hero_EvRel(HeroModel):
    def __init__(self, cfg, comm, vfeat_dim, max_frm_seq_len):
        super().__init__(cfg, vfeat_dim=4352, max_frm_seq_len=max_frm_seq_len)
        self.cfg = cfg
        self.comm = comm
        self.build_model()
       
    def build_model(self):
        hsz = self.cfg.c_config.hidden_size
        self.vidsitu_pool = nn.Linear(
            in_features=hsz,
            out_features=1,
            bias=False)
        self.vidsitu_cls_head = MLPLayer(hsz, 5) # 5 evrel classes including null class
        
    def pad_tensors(tensors, lens=None, pad=0, max_len=0):
        """B x [T, ...]"""
        if lens is None:
            lens = [t.size(0) for t in tensors]
        if max_len == 0:
            max_len = max(lens)
        bs = len(tensors)
        hid = tensors[0].size(-1)
        dtype = tensors[0].dtype
        output = torch.zeros(bs, max_len, hid, dtype=dtype)
        if pad:
            output.data.fill_(pad)
        for i, (t, l) in enumerate(zip(tensors, lens)):
            output.data[i, :l, ...] = t.data
        return output

    def get_gather_index(txt_lens, num_frames, batch_size, max_len, out_size):
        assert len(txt_lens) == len(num_frames) == batch_size
        gather_index = torch.arange(0, out_size, dtype=torch.long
                                    ).unsqueeze(0).expand(batch_size, -1).clone()

        for i, (tl, nframe) in enumerate(zip(txt_lens, num_frames)):
            gather_index.data[i, nframe:tl+nframe] = torch.arange(
                max_len, max_len+tl, dtype=torch.long).data
        return gather_index

    def get_hero_inputs(self, inp):         
        # Annotations are in a tensor of shape 
        # [batch size, number of events (5), number of annotations (1 or 3), tokenized string length]
        # When validating effective batch size will be b_num*3
        frm_ann_toks = inp["evrel_seq_out_ones"]
        B, num_ev, num_seq_eg, seq_len = frm_ann_toks.shape
        hero_frm_feats = inp["hero_frm_feats"]
        labels = inp["evrel_labs"]
        b_device = "cuda:" + str(frm_ann_toks.get_device())
        
        def format_input_ids(input_id):
            # Replace start token with 2 and remove padding
            input_id[0] = 2
            
            if (input_id==1).nonzero().numel() > 0:
                return(input_id[:int((input_id==1).nonzero()[0].squeeze())])
            else:
                return input_id
            
        def pad_tensors(tensors, lens=None, pad=0, max_len=0):
            """B x [T, ...]"""
            if lens is None:
                lens = [t.size(0) for t in tensors]
            if max_len == 0:
                max_len = max(lens)
            bs = len(tensors)
            hid = tensors[0].size(-1)
            dtype = tensors[0].dtype
            output = torch.zeros(bs, max_len, hid, dtype=dtype)
            if pad:
                output.data.fill_(pad)
            for i, (t, l) in enumerate(zip(tensors, lens)):
                output.data[i, :l, ...] = t.data
            return output

        def get_gather_index(txt_lens, num_frames, batch_size, max_len, out_size):
            assert len(txt_lens) == len(num_frames) == batch_size
            gather_index = torch.arange(0, out_size, dtype=torch.long
                                        ).unsqueeze(0).expand(batch_size, -1).clone()

            for i, (tl, nframe) in enumerate(zip(txt_lens, num_frames)):
                gather_index.data[i, nframe:tl+nframe] = torch.arange(
                    max_len, max_len+tl, dtype=torch.long).data
            return gather_index

        def video_collate(inputs, b_device):
            (frame_level_input_ids,
             frame_level_v_feats,
             frame_level_attn_masks,
             clip_level_v_feats,
             clip_level_attn_masks, num_subs,
             sub_idx2frame_idx) = map(list, unzip(inputs))

            # all_f_sub_input_ids: list[tensor(sep, w0, w1)]
            # whose list size = total number of subs
            all_f_sub_input_ids, all_f_v_feats, all_f_attn_masks = [], [], []
            for i in range(len(num_subs)):
                all_f_sub_input_ids += frame_level_input_ids[i]
                all_f_v_feats += frame_level_v_feats[i]
                all_f_attn_masks += frame_level_attn_masks[i]

            txt_lens = [i.size(0) for i in all_f_sub_input_ids]  # len. of each sub
            # hardcoded padding value
            all_f_sub_input_ids = pad_sequence(
                all_f_sub_input_ids, batch_first=True, padding_value=1)

            all_f_sub_pos_ids = torch.arange(0, all_f_sub_input_ids.size(1),
                                             dtype=torch.long).unsqueeze(0)
            all_f_sub_pos_ids.data[all_f_sub_pos_ids > 511] = 511  # FIXME quick hack
            all_f_attn_masks = pad_sequence(
                all_f_attn_masks, batch_first=True, padding_value=0)

            v_lens = [i.size(0) for i in all_f_v_feats]
            all_f_v_feats = pad_tensors(all_f_v_feats, v_lens, 0)
            all_f_v_pos_ids = torch.arange(0, all_f_v_feats.size(1), dtype=torch.long
                                           ).unsqueeze(0)

            # all_f_sub_input_attn_masks (total_subs, max_sl) for subtitles only
            all_f_sub_input_attn_masks = [torch.tensor([1] * tl) for tl in txt_lens]
            all_f_sub_input_attn_masks = pad_sequence(
                all_f_sub_input_attn_masks, batch_first=True, padding_value=0)

            bs, max_vl, _ = all_f_v_feats.size()
            out_size = all_f_attn_masks.size(1)
            frame_level_gather_index = get_gather_index(
                txt_lens, v_lens, bs, max_vl, out_size)

            num_frames = [i.size(0) for i in clip_level_v_feats]
            clip_level_v_feats = pad_tensors(
                clip_level_v_feats, num_frames, pad=0)
            clip_level_pos_ids = torch.arange(
                0, clip_level_v_feats.size(1), dtype=torch.long
            ).unsqueeze(0).expand(clip_level_v_feats.size(0), -1).clone()

            clip_level_attn_masks = pad_sequence(
                    clip_level_attn_masks, batch_first=True, padding_value=0)
            
            test_tensor = torch.tensor([1, 2, 3, 4, 5])
            # Store required tensors on cuda
            cuda_x = torch.device(b_device)
            all_f_sub_pos_ids = all_f_sub_pos_ids.to(cuda_x)
            all_f_v_feats = all_f_v_feats.to(cuda_x)
            all_f_v_pos_ids = all_f_v_pos_ids.to(cuda_x)
            all_f_attn_masks = all_f_attn_masks.to(cuda_x)
            frame_level_gather_index = frame_level_gather_index.to(cuda_x)
            all_f_sub_input_attn_masks = all_f_sub_input_attn_masks.to(cuda_x)
            clip_level_v_feats = clip_level_v_feats.to(cuda_x)
            clip_level_pos_ids = clip_level_pos_ids.to(cuda_x)
            clip_level_attn_masks = clip_level_attn_masks.to(cuda_x)
            
            batch = {'f_sub_input_ids': all_f_sub_input_ids,  # (total_sub, max_sl)
                     'f_sub_pos_ids': all_f_sub_pos_ids,      # (total_sub, max_sl)
                     'f_v_feats': all_f_v_feats,              # (total_sub, max_vl, k)
                     'f_v_pos_ids': all_f_v_pos_ids,          # (total_sub, max_vl)
                     'f_attn_masks': all_f_attn_masks,        # (total_sub, max_vl+max_sl)
                     'f_gather_index': frame_level_gather_index,  # (total_sub, max_vl+max_sl)
                     'f_sub_input_attn_masks': all_f_sub_input_attn_masks, # (total_sub, max_sl)
                     'c_v_feats': clip_level_v_feats,         # (bz, max_len, k)
                     'c_pos_ids': clip_level_pos_ids,         # (bz, max_len) [matched, unmatched]
                     'c_attn_masks': clip_level_attn_masks,   # (bz, max_len)
                     'num_subs': num_subs,                    # [num_sub]
                     'sub_idx2frame_idx': sub_idx2frame_idx}  # [ [(sub_ix, [frame_ix]) ] ]
            return batch
        
        def txt_input_collate(input_ids, attn_masks, b_device):
            # hard_coded padding value, TODO: check correctness
            pad_values = 1 if len(input_ids[0].size()) == 1 else 0
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_values)
            pos_ids = torch.arange(
                0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
            pos_ids.data[pos_ids > 511] = 511  # FIXME quick hack
            attn_masks = pad_sequence(
                attn_masks, batch_first=True, padding_value=0)
            cuda_x = torch.device(b_device)
            return input_ids.to(cuda_x), pos_ids.to(cuda_x), attn_masks.to(cuda_x)
        
        # Prepare video and subtitle inputs
        sampling_rate = int(list(hero_frm_feats.size())[1]/5)
        video_q_inputs = []
        for b_num in range(B):
            for seq_eg_num in range(num_seq_eg):
                frame_level_input_ids = []
                frame_level_v_feats = []
                frame_level_attn_masks = []
                vsm_frame_level_input_ids = []
                vsm_frame_level_v_feats = []
                vsm_frame_level_attn_masks = []
                vsm_sub_queries_and_targets = []
                
                clip_level_v_feats = hero_frm_feats[b_num, :, :]
                clip_level_attn_masks = torch.tensor([1]*len(clip_level_v_feats)) # Should be [1, 1, 1, 1, 1]
                vsm_clip_level_v_feats = hero_frm_feats[b_num, :, :]
                vsm_clip_level_attn_masks = torch.tensor([1]*len(clip_level_v_feats)) # Should be [1, 1, 1, 1, 1]
                num_subs = 5
                sub2frames = [(0, [0, 1]), (1, [2, 3]), (2, [4, 5]), (3, [6, 7]), (4, [8, 9])]
                
                matched_sub_idx = [sub_idx for sub_idx, matched_frames in sub2frames
                                   if matched_frames]
                n_samples = min(len(matched_sub_idx), 2)
                query_sub_ids = set(random.sample(matched_sub_idx, n_samples))
                
                for ev_num in range(num_ev):
                    #print("\nfrm_ann_toks:\n", frm_ann_toks[b_num, ev_num, seq_eg_num, :])
                    input_ids = format_input_ids(frm_ann_toks[b_num, ev_num, seq_eg_num, :])
                    frame_level_input_ids.append(input_ids) 
                    frame_level_v_feats.append(
                        hero_frm_feats[b_num, 
                                       ev_num*sampling_rate:ev_num*sampling_rate+sampling_rate, :]
                        #[None, :]
                    )                    
                    frame_level_attn_masks.append(torch.tensor([1] * (len(input_ids)+2)))
                
                
                
                '''
                sub_ctx_len = 0
                max_txt_len = 100 #60
                for sub_idx, matched_frames in sub2frames:
                    # text input
                    if sub_ctx_len >= 0:
                        curr_sub_ctx_input_ids = []
                        for tmp_sub_idx in range(sub_idx-sub_ctx_len,
                                                 sub_idx+1):
                            if tmp_sub_idx >= 0 and tmp_sub_idx < num_subs\
                                    and tmp_sub_idx not in query_sub_ids:b  
                                in_ids = frame_level_input_ids[tmp_sub_idx]
                                if max_txt_len != -1:
                                    in_ids = in_ids[:max_txt_len]
                                curr_sub_ctx_input_ids.extend(copy.deepcopy(in_ids))
                    curr_sub_ctx_input_ids = [2] + curr_sub_ctx_input_ids
                   
                    n_frame = len(matched_frames)
                    attn_masks_fill_0_pos = None
                    if n_frame:
                        matched_v_feats = torch
                    '''    
                
                frame_level_input_ids_zeros = [torch.zeros(x.size(), dtype=x.dtype, device=x.device) for x in frame_level_input_ids]
                #frame_level_v_feats_zeros = [torch.zeros(x.size(), dtype=x.dtype) for x in frame_level_v_feats]
                #clip_level_v_feats_zeros = torch.zeros(clip_level_v_feats.size(), dtype=clip_level_v_feats.dtype)
                
                
                #print("clip_level_v_feats: ", clip_level_v_feats)
                #print("clip_level_v_feats_zeros: ", clip_level_v_feats_zeros)
                #print(frame_level_input_ids[0])
                #print(frame_level_input_ids_zeros[0])
                   
                out = (frame_level_input_ids_zeros,   # num_subs list[tensor(sep,w0,w1,...)]
                       frame_level_v_feats,     # num_subs list[tensor(#sub_frames, d)]
                       frame_level_attn_masks,  # num_subs list[L_sub + #sub_frames
                       #vsm_frame_level_input_ids,
                       #vsm_frame_level_v_feats,
                       #vsm_frame_level_attn_masks,
                       #vsm_sub_queries_and_targets,
                       clip_level_v_feats,      # tensor(num_frames, d)
                       clip_level_attn_masks,   # #frames list[1]
                       num_subs, sub2frames)    # num_subs, [(sub_ix, [frame_ix]) ]
                
                # Repeat each element 4 times since we have 4 querries for 4 evrel preds per clip
                video_q_inputs.append([out] * 4)       
        all_video_qa_inputs = []
        for i in range(len(video_q_inputs)):
            all_video_qa_inputs.extend(video_q_inputs[i])     
        
        hero_inputs = video_collate(all_video_qa_inputs, b_device)
        
        # Prepare querry inputs
        ev1_ev3 = torch.tensor([0, 134, 12, 246, 2])
        ev2_ev3 = torch.tensor([0, 176, 12, 246, 2])
        ev3_ev4 = torch.tensor([0, 246, 12, 306, 2])
        ev3_ev5 = torch.tensor([0, 246, 12, 245, 2])
        ev_seq_tok_list = [ev1_ev3, ev2_ev3, ev3_ev4, ev3_ev5]
        q_attn_mask = torch.tensor([1, 1, 1, 1, 1])
        
        all_q_input_ids = []
        all_q_attn_masks = []
        for b_num in range(B):
            for seq_eg_num in range(num_seq_eg):
                for q_num in range(4):
                    all_q_input_ids.append(ev_seq_tok_list[q_num])
                    all_q_attn_masks.append(q_attn_mask)
        
        input_ids, pos_ids, attn_masks = txt_input_collate(all_q_input_ids, all_q_attn_masks, b_device)
        
        # Prepare target inputs
        targets = labels.transpose(dim0=1, dim1=2).reshape((np.prod(list(labels.size())), 1))  
        
        hero_inputs["targets"] = targets
        hero_inputs["q_input_ids"] = input_ids
        hero_inputs["q_pos_ids"] = pos_ids
        hero_inputs["q_attn_masks"] = attn_masks
        
        # Prepare VSM inputs
        

        
        
        #hero_inputs["vsm_q_input_ids"]
        #hero_inputs["vsm_q_input_ids"]
        #hero_inputs["vsm_q_attn_masks"]
        #hero_inputs["vsm_targets"]

        
        return hero_inputs

    def get_modularized_video(self, frame_embeddings, frame_mask):
        """
        Args:
            frame_embeddings: (Nv, L, D)
            frame_mask: (Nv, L)
        """
        vidsitu_attn_scores = self.vidsitu_pool(
                frame_embeddings)  # (Nv, L, 1)

        vidsitu_attn_scores = F.softmax(
            mask_logits(vidsitu_attn_scores,
                        frame_mask.unsqueeze(-1)), dim=1)

        vidsitu_pooled_video = torch.einsum(
            "vlm,vld->vmd", vidsitu_attn_scores,
            frame_embeddings)  # (Nv, 1, D)
        return vidsitu_pooled_video.squeeze(1)
        
    def forward(self, inp):
        B, num_ev, num_seq_eg, seq_len = inp["rob_full_evrel_seq_out_ones"].shape
        batch = defaultdict(lambda: None, self.get_hero_inputs(inp))
        c_attn_masks = batch["c_attn_masks"]
        # (num_video * 5, num_frames, hid_size)
        frame_embeddings = self.v_encoder.forward_repr(
            batch, encode_clip=False)
        frame_embeddings = self.v_encoder.c_encoder.embeddings(
            frame_embeddings,
            position_ids=None)
        q_embeddings = self.v_encoder.f_encoder._compute_txt_embeddings(
            batch["q_input_ids"], batch["q_pos_ids"], txt_type_ids=None)
        frame_q_embeddings = torch.cat(
            (frame_embeddings, q_embeddings), dim=1)
        frame_q_attn_mask = torch.cat(
            (c_attn_masks, batch["q_attn_masks"]), dim=1)
        fused_video_q = self.v_encoder.c_encoder.forward_encoder(
            frame_q_embeddings, frame_q_attn_mask)
        num_frames = c_attn_masks.shape[1]
        video_embeddings = fused_video_q[:, :num_frames, :]

        video_masks = c_attn_masks.to(dtype=video_embeddings.dtype)
        vidsitu_pooled_video = self.get_modularized_video(
            video_embeddings, video_masks)
        logits = self.vidsitu_cls_head(vidsitu_pooled_video)
        
        targets = batch['targets']
        scores = torch.sigmoid(logits).squeeze(-1)
        targets = targets.squeeze(-1).to(dtype=torch.long)
        loss = F.cross_entropy(scores, targets, reduction='mean')
        
        logits_list = logits.tolist()
        mdl_out_list = []
        b_device = "cuda:" + str(logits.get_device())
        for b_idx in range(B):
            same_b_idx_logits = []
            for evrel_idx in range(4):
                same_evrel_idx_logits = []
                for seq_eg_idx in range(num_seq_eg):
                    start = b_idx * num_seq_eg * 4 + evrel_idx
                    interval = seq_eg_idx * 4
                    same_evrel_idx_logits.append(logits_list[start + interval])
                same_b_idx_logits.append(same_evrel_idx_logits)
            mdl_out_list.append(same_b_idx_logits)

        mdl_out = torch.tensor(mdl_out_list, dtype=torch.float32, device=b_device)  
        
        out_dct = {}
        out_dct["loss"] = loss
        out_dct["mdl_out"] = mdl_out
        return out_dct

    
class Simple_EvRel_Roberta(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.full_cfg = cfg
        self.cfg = cfg.mdl
        self.comm = comm
        self.build_model()

    def build_model(self):
        self.rob_mdl = RobertaForSequenceClassification.from_pretrained(
            self.full_cfg.mdl.rob_mdl_name, num_labels=5
        )
        return

    def forward(self, inp):
        src_toks1 = inp["rob_full_evrel_seq_out"]
        src_attn1 = inp["rob_full_evrel_seq_out_lens"]

        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)

        out = self.rob_mdl(
            input_ids=src_toks,
            attention_mask=src_attn_mask,
            return_dict=True,
            # token_type_ids=src_tok_typ_ids,
        )
        # B*num_ev x num_seq_eg*seq_len x vocab_size
        logits = out["logits"]
        labels = inp["evrel_labs"]

        relation_freq_list = [0, 16421/16421, 23357/16421, 32707/16421, 24579/16421] 
        labels_flattened = labels.clone().detach().flatten().tolist()
        freq_sum = 0.0
        for i in labels_flattened:
            freq_sum += relation_freq_list[i]
        norm_value = freq_sum / len(labels_flattened)
       
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )/norm_value
        out["loss"] = loss
        out["mdl_out"] = logits.view(B, num_ev, num_seq_eg, -1)
        return out


class SFPret_SimpleEvRel(nn.Module):    
    def __init__(self, cfg, comm):
        super().__init__()
        self.full_cfg = cfg
        self.cfg = cfg.mdl
        self.comm = comm
        self.build_model()

    def build_model(self):
        self.rob_mdl = RobertaModel.from_pretrained(
            self.full_cfg.mdl.rob_mdl_name, add_pooling_layer=True
        )
        head_dim = get_head_dim(self.full_cfg) #Slowfeast feat dim

        self.vid_feat_encoder = nn.Sequential(
            *[nn.Linear(head_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
        )

        self.vis_lang_encoder = nn.Sequential(
            *[nn.Linear(1792, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
        )
        self.vis_lang_classf = nn.Sequential(
            *[nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 5)]
        )
        return

    def get_src(self, inp):
        return inp["rob_full_evrel_seq_out_ones"], inp["rob_full_evrel_seq_out_ones_lens"]

    def forward(self, inp):       
        src_toks1, src_attn1 = self.get_src(inp)
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)

        lang_out = self.rob_mdl(
            input_ids=src_toks, attention_mask=src_attn_mask, return_dict=True
        )
        pooler_out = lang_out.pooler_output

        pooler_out_5 = pooler_out.view(B, 5, num_seq_eg, pooler_out.size(-1))

        frm_feats = inp["frm_feats"] #Slowfast feats
        
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        vis_out = self.vid_feat_encoder(frm_feats)
        vis_out = (
            vis_out.view(B, 5, 1, -1)
            .contiguous()
            .expand(B, 5, num_seq_eg, -1)
            .contiguous()
        )
        # vis_lang_out = self.vis_lang_encoder(
        #     torch.cat([vis_out.new_zeros(vis_out.shape), pooler_out_5], dim=-1)
        # )
        vis_lang_out = self.vis_lang_encoder(torch.cat([vis_out, pooler_out_5], dim=-1))
        vis_lang_out1 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([0, 1, 2, 2]).long()
        ).contiguous()
        vis_lang_out2 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([2, 2, 3, 4]).long()
        ).contiguous()

        vis_lang_out3 = torch.cat([vis_lang_out1, vis_lang_out2], dim=-1)
        vis_lan_out_full_seq = torch.cat
        logits = self.vis_lang_classf(vis_lang_out3).contiguous()

        # B*num_ev x num_seq_eg*seq_len x vocab_size
        labels = inp["evrel_labs"]
        relation_freq_list = [0, 16421/16421, 23357/16421, 32707/16421, 24579/16421] 
        labels_flattened = labels.clone().detach().flatten().tolist()
        freq_sum = 0.0
        for i in labels_flattened:
            freq_sum += relation_freq_list[i]
        norm_value = freq_sum / len(labels_flattened)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )/norm_value
        out_dct = {}
        out_dct["loss"] = loss
        out_dct["mdl_out"] = logits.view(B, num_ev - 1, num_seq_eg, -1)
        
        return out_dct


class SFPret_OnlyVb_SimpleEvRel(SFPret_SimpleEvRel):
    def get_src(self, inp):
        return inp["rob_full_evrel_vbonly_out_ones"], inp["evrel_vbonly_out_ones_lens"]


class SFPret_OnlyVid_SimpleEvRel(SFPret_SimpleEvRel):
    def forward(self, inp):
        src_toks1, src_attn1 = self.get_src(inp)
        # src_tok_typ_ids1 = inp["evrel_seq_tok_ids"]
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)
        # src_tok_typ_ids = src_tok_typ_ids1.view(B * num_ev * num_seq_eg, seq_len)
        lang_out = self.rob_mdl(
            input_ids=src_toks, attention_mask=src_attn_mask, return_dict=True
        )
        pooler_out = lang_out.pooler_output

        pooler_out_5 = pooler_out.view(B, 5, num_seq_eg, pooler_out.size(-1))

        frm_feats = inp["frm_feats"]
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        vis_out = self.vid_feat_encoder(frm_feats)
        vis_out = (
            vis_out.view(B, 5, 1, -1)
            .contiguous()
            .expand(B, 5, num_seq_eg, -1)
            .contiguous()
        )

        vis_lang_out = self.vis_lang_encoder(
            torch.cat([vis_out, pooler_out_5.new_zeros(pooler_out_5.shape)], dim=-1)
        )
        vis_lang_out1 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([0, 1, 2, 2]).long()
        ).contiguous()
        vis_lang_out2 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([2, 2, 3, 4]).long()
        ).contiguous()

        vis_lang_out3 = torch.cat([vis_lang_out1, vis_lang_out2], dim=-1)
        logits = self.vis_lang_classf(vis_lang_out3).contiguous()

        # B*num_ev x num_seq_eg*seq_len x vocab_size

        labels = inp["evrel_labs"]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )
        out_dct = {}
        out_dct["loss"] = loss
        out_dct["mdl_out"] = logits.view(B, num_ev - 1, num_seq_eg, -1)
        return out_dct


class Simple_TxEncEvRel(SFPret_SimpleEvRel):
    def forward(self, inp):
        src_toks1, src_attn1 = self.get_src(inp)
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)

        lang_out = self.rob_mdl(
            input_ids=src_toks, attention_mask=src_attn_mask, return_dict=True
        )
        pooler_out = lang_out.pooler_output

        pooler_out_5 = pooler_out.view(B, 5, num_seq_eg, pooler_out.size(-1))

        frm_feats = inp["frm_feats"]
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        vis_out = self.vid_feat_encoder(frm_feats)
        vis_out = (
            vis_out.view(B, 5, 1, -1)
            .contiguous()
            .expand(B, 5, num_seq_eg, -1)
            .contiguous()
        )

        vis_lang_out = self.vis_lang_encoder(
            torch.cat([vis_out.new_zeros(vis_out.shape), pooler_out_5], dim=-1)
        )
        vis_lang_out1 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([0, 1, 2, 2]).long()
        ).contiguous()
        vis_lang_out2 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([2, 2, 3, 4]).long()
        ).contiguous()

        vis_lang_out3 = torch.cat([vis_lang_out1, vis_lang_out2], dim=-1)
        logits = self.vis_lang_classf(vis_lang_out3).contiguous()

        # B*num_ev x num_seq_eg*seq_len x vocab_size

        labels = inp["evrel_labs"]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )
        out_dct = {}
        out_dct["loss"] = loss
        out_dct["mdl_out"] = logits.view(B, num_ev - 1, num_seq_eg, -1)
        return out_dct
