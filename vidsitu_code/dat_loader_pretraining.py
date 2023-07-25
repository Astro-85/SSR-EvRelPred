from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode as CN
from typing import List, Dict
from munch import Munch
from PIL import Image
import numpy as np
from collections import Counter
from utils.video_utils import get_sequence, pack_pathway_output, tensor_normalize
from utils.dat_utils import (
    DataWrap,
    get_dataloader,
    simple_collate_dct_list,
    coalesce_dicts,
    arg_mapper,
    pad_words_new,
    pad_tokens,
    read_file_with_assertion,
)
from transformers import GPT2TokenizerFast, RobertaTokenizerFast


def st_ag(ag):
    return f"<{ag}>"


def end_ag(ag):
    return f"</{ag}>"


def enclose_ag(agname, ag_str):
    return f"{st_ag(agname)} {ag_str} {end_ag(agname)}"


def enclose_ag_st(agname, ag_str):
    return f"{st_ag(agname)} {ag_str}"


class VsituDS(Dataset):
    
    def __init__(self, cfg: CN, comm: Dict, split_type: str):
        self.master_index = 0
        self.full_cfg = cfg
        self.cfg = cfg.ds.vsitu
        self.sf_cfg = cfg.sf_mdl
        self.task_type = self.full_cfg.task_type

        self.comm = Munch(comm)
        self.split_type = split_type
        if len(comm) == 0:
            self.set_comm_args()

        assert self.full_cfg.ds.val_set_type == "lb"
        self.full_val = True
        self.read_files(self.split_type)

        if self.task_type == "vb":
            self.itemgetter = getattr(self, "vb_only_item_getter")
        elif self.task_type == "vb_arg":
            self.itemgetter = getattr(self, "vb_args_item_getter")
            self.is_evrel = False
            self.comm.dct_id = "gpt2_hf_tok"
        elif self.task_type == "evrel":
            self.itemgetter = getattr(self, "vb_args_item_getter")
            self.comm.dct_id = "rob_hf_tok"
            self.is_evrel = True
        else:
            raise NotImplementedError

    def set_comm_args(self):
        frm_seq_len = self.sf_cfg.DATA.NUM_FRAMES * self.sf_cfg.DATA.SAMPLING_RATE
        fps = self.sf_cfg.DATA.TARGET_FPS
        cent_frm_per_ev = {f"Ev{ix+1}": int((ix + 1 / 2) * fps * 2) for ix in range(5)}

        self.comm.num_frms = self.sf_cfg.DATA.NUM_FRAMES
        self.comm.sampling_rate = self.sf_cfg.DATA.SAMPLING_RATE
        self.comm.frm_seq_len = frm_seq_len
        self.comm.fps = fps
        self.comm.cent_frm_per_ev = cent_frm_per_ev
        self.comm.max_frms = 300

        self.comm.vb_id_vocab = read_file_with_assertion(
            self.cfg.vocab_files.verb_id_vocab, reader="pickle"
        )
        self.comm.rob_hf_tok = RobertaTokenizerFast.from_pretrained(
            self.full_cfg.mdl.rob_mdl_name
        )
        self.comm.gpt2_hf_tok = read_file_with_assertion(
            self.cfg.vocab_files.new_gpt2_vb_arg_vocab, reader="pickle"
        )

        def ptoken_id(self):
            return self.pad_token_id

        def unktoken_id(self):
            return self.unk_token_id

        def eostoken_id(self):
            return self.eos_token_id

        GPT2TokenizerFast.pad = ptoken_id
        GPT2TokenizerFast.unk = unktoken_id
        GPT2TokenizerFast.eos = eostoken_id

        self.comm.ev_sep_token = "<EV_SEP>"
        assert self.cfg.num_ev == 5
        self.comm.num_ev = self.cfg.num_ev

        ag_dct = self.cfg.arg_names
        ag_dct_main = {}
        ag_dct_start = {}
        ag_dct_end = {}

        for agk, agv in ag_dct.items():
            ag_dct_main[agk] = agv
            ag_dct_start[agk] = st_ag(agv)
            ag_dct_end[agk] = end_ag(agv)
        ag_dct_all = {
            "ag_dct_main": ag_dct_main,
            "ag_dct_start": ag_dct_start,
            "ag_dct_end": ag_dct_end,
        }
        self.comm.ag_name_dct = CN(ag_dct_all)

        self.comm.evrel_dct = {
            "Null": 0,
            "Causes": 1,
            "Reaction To": 2,
            "Enables": 3,
            "NoRel": 4,
        }
        self.comm.evrel_dct_opp = {v: k for k, v in self.comm.evrel_dct.items()}

        if self.sf_cfg.MODEL.ARCH in self.sf_cfg.MODEL.MULTI_PATHWAY_ARCH:
            self.comm.path_type = "multi"
        elif self.sf_cfg.MODEL.ARCH in self.sf_cfg.MODEL.SINGLE_PATHWAY_ARCH:
            self.comm.path_type = "single"
        else:
            raise NotImplementedError

    def read_files(self, split_type: str):
        # Video frames
        #self.vsitu_frm_dir = Path(self.cfg.video_frms_tdir)
        split_files_cfg = self.full_cfg.ds.viscom.split_files_lb
        vsitu_ann_files_cfg = self.full_cfg.ds.viscom.viscom_ann_files_lb
        vinfo_files_cfg = self.cfg.vinfo_files_lb
        
        #print("\nvsitu_ann_files_cfg: \n", vsitu_ann_files_cfg)

        self.vseg_lst = read_file_with_assertion(split_files_cfg[split_type])
        vseg_ann_lst = read_file_with_assertion(vsitu_ann_files_cfg[split_type])
        
        #print(len(vseg_ann_lst))
        
        vsitu_ann_dct = {}

        for vseg_ann in vseg_ann_lst:
            #print("\nvseg_ann: \n", vseg_ann)
            vseg = vseg_ann["img_fn"]
            if vseg not in vsitu_ann_dct:
                vsitu_ann_dct[vseg] = []
            vsitu_ann_dct[vseg].append(vseg_ann)
        self.vsitu_ann_dct = vsitu_ann_dct

    def __len__(self) -> int:
        if self.full_cfg.debug_mode:
            return 30
        return len(self.vseg_lst)

    def __getitem__(self, index: int) -> Dict:
        return self.itemgetter(index)
    
    # Function given one 10 second video (vid_seg_ann_lst)
    def get_vb_arg_data(self, vid_seg_ann_lst: List, is_evrel: bool = False):
        agset = ["Arg0", "Arg1", "Arg2"]
        only_vb_lst_all_ev = []
        seq_lst_all_ev = []
        seq_lst_all_ev_lens = []
        evrel_lst_all_ev = []

        word_voc = self.comm.gpt2_hf_tok
        addn_word_voc = word_voc.get_added_vocab()

        vb_id_lst = []
        seq_id_lst = []

        evrel_seq_lst_all_ev = []
        
        out_dct = {}
        before_after_lst = []

        for vsix, vid_seg_ann in enumerate(vid_seg_ann_lst):
            if vsix > 0:
                break

        self.master_index += 1
        
        central_ev = vid_seg_ann['event_vb_args'][0].replace('_', '.')
        for arg in vid_seg_ann['event_vb_args'][1].keys():
                central_ev += ' <' + arg + '> ' + vid_seg_ann['event_vb_args'][1][arg]
        central_ev += ' <AScn> ' + vid_seg_ann['place']
        

        for befores in vid_seg_ann['before_vb_args']:
            before_pair = [befores, "before"]
            before_after_lst.append(before_pair)

        for afters in vid_seg_ann["after_vb_args"]:
            after_pair = [afters, "after"]
            before_after_lst.append(after_pair)
        
        before_after_lst.append([vid_seg_ann['intent_vb_args'], "intent"])

        #print("\nbefore_after_lst: \n", before_after_lst)

        # Randomize order of before and after events
        random.shuffle(before_after_lst)
        before_after_lst = before_after_lst[0:4]

        # Generate ground truth list that is not encoded
        gt_lst = []
        gt_dct = {"before": 1, "after": 2, "intent": 3}
        for pair in before_after_lst:
            gt_lst.append(gt_dct[pair[1]])
        
        def arrange_vb_args(vb_args_list):
            #print(vb_args_list)
            event_str = vb_args_list[0].replace('_', '.')
            for arg in vb_args_list[1].keys():
                event_str += ' <' + arg + '> ' + vb_args_list[1][arg]
            return(event_str)
        
        
        s1 = arrange_vb_args(before_after_lst[0][0])
        s2 = arrange_vb_args(before_after_lst[1][0])
        s3 = central_ev
        s4 = arrange_vb_args(before_after_lst[2][0])
        s5 = arrange_vb_args(before_after_lst[3][0])  

        evrel_wvoc = self.comm.rob_hf_tok
       
        full_out_evrel_seq_by_ev = []
        full_out_evrel_seq_by_ev_lens = []
        
        for evix in [0, 1, 3, 4]:
            full_out_evrel_seq_lst = []
            full_out_evrel_seq_lens = []
            
            if evix == 0:
                s1 = "*" + s1
                s3 = "**" + s3
                new_seq = evrel_wvoc(
                    s1 + evrel_wvoc.sep_token + 
                    s2 + evrel_wvoc.sep_token + 
                    s3 + evrel_wvoc.sep_token + 
                    s4 + evrel_wvoc.sep_token + 
                    s5
                )["input_ids"]
            elif evix == 1:
                s2 = "*" + s2
                s3 = "**" + s3
                new_seq = evrel_wvoc(
                    s1 + evrel_wvoc.sep_token + 
                    s2 + evrel_wvoc.sep_token + 
                    s3 + evrel_wvoc.sep_token + 
                    s4 + evrel_wvoc.sep_token + 
                    s5
                )["input_ids"]
            elif evix == 3:
                s3 = "*" + s3
                s4 = "**" + s4
                new_seq = evrel_wvoc(
                    s1 + evrel_wvoc.sep_token + 
                    s2 + evrel_wvoc.sep_token + 
                    s3 + evrel_wvoc.sep_token + 
                    s4 + evrel_wvoc.sep_token + 
                    s5
                )["input_ids"]
            else:
                s3 = "*" + s3
                s5 = "**" + s5
                new_seq = evrel_wvoc(
                    s1 + evrel_wvoc.sep_token + 
                    s2 + evrel_wvoc.sep_token + 
                    s3 + evrel_wvoc.sep_token + 
                    s4 + evrel_wvoc.sep_token + 
                    s5
                )["input_ids"]


            new_seq_pad, new_seq_msk = pad_tokens(
                new_seq,
                pad_index=evrel_wvoc.pad_token_id,
                pad_side="right",
                append_eos=False,
                eos_index=evrel_wvoc.eos_token_id,
                max_len=240,
            )
            
            full_out_evrel_seq_lst.append(new_seq_pad.tolist())
            full_out_evrel_seq_lens.append(new_seq_msk)
            
            full_out_evrel_seq_by_ev.append(full_out_evrel_seq_lst)
            full_out_evrel_seq_by_ev_lens.append(full_out_evrel_seq_lens)
            
        evrel_labs = []
        for gt in gt_lst: evrel_labs.append([gt])
            
        out_dct["evrel_seq_out"] = torch.tensor(full_out_evrel_seq_by_ev).long()
        out_dct["evrel_seq_out_lens"] = torch.tensor(full_out_evrel_seq_by_ev_lens).long()
        out_dct["evrel_labs"] = torch.tensor(evrel_labs).long()
        out_dct["gt_lst"] = torch.tensor(gt_lst).long()
            
        return out_dct

    def vb_args_item_getter(self, idx: int):
        #print("\nidx: \n", idx)
        vid_seg_name = self.vseg_lst[idx]
        if self.split_type == "train":
            #print("\nTRAINING\n")
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            #print("\nvid_seg_ann_: \n", vid_seg_ann_)
            vid_seg_ann = vid_seg_ann_[0]
            #print("\nvid_seg_ann: \n", vid_seg_ann)
            seq_out_dct = self.get_vb_arg_data([vid_seg_ann], is_evrel=self.is_evrel)
            #print("\nseq_out_dct: \n", seq_out_dct)
        elif "valid" in self.split_type:
            #print("\nVALIDATING\n")
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            #assert len(vid_seg_ann_) >= 3
            vid_seg_ann_ = vid_seg_ann_[:1]
            seq_out_dct = self.get_vb_arg_data(vid_seg_ann_, is_evrel=self.is_evrel)
        elif "test" in self.split_type:
            assert self.task_type == "evrel"
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            assert len(vid_seg_ann_) >= 3
            vid_seg_ann_ = vid_seg_ann_[:3]
            seq_out_dct = self.get_vb_arg_data(vid_seg_ann_, is_evrel=self.is_evrel)
        else:
            raise NotImplementedError
        seq_out_dct["vseg_idx"] = torch.tensor(idx)

        # put this into extended_cfg
        if self.full_cfg.mdl.mdl_name not in set(
            [
                "txed_only",
                "tx_only",
                "gpt2_only",
                "new_gpt2_only",
                "tx_ev_only",
                "new_gpt2_ev_only",
                "rob_evrel",
                "bart_evrel",
            ]
        ):
            frm_feats_out_dct = self.get_frm_feats_all(idx)
            return coalesce_dicts([frm_feats_out_dct, seq_out_dct])
        else:
            return seq_out_dct


class BatchCollator:
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm

    def __call__(self, batch):
        out_dict = simple_collate_dct_list(batch)
        return out_dict


def get_data(cfg):
    DS = VsituDS
    BC = BatchCollator

    train_ds = DS(cfg, {}, split_type="train")
    valid_ds = DS(cfg, train_ds.comm, split_type="valid")
    assert cfg.ds.val_set_type == "lb"
    if cfg.only_test:
        if cfg.task_type == "vb":
            test_ds = DS(cfg, train_ds.comm, split_type="test_verb")
        elif cfg.task_type == "vb_arg":
            test_ds = DS(cfg, train_ds.comm, split_type="test_srl")
        elif cfg.task_type == "evrel":
            test_ds = DS(cfg, train_ds.comm, split_type="test_evrel")
        else:
            raise NotImplementedError
    else:
        test_ds = None
    batch_collator = BC(cfg, train_ds.comm)
    train_dl = get_dataloader(cfg, train_ds, is_train=True, collate_fn=batch_collator)
    valid_dl = get_dataloader(cfg, valid_ds, is_train=False, collate_fn=batch_collator)

    if cfg.only_test:
        test_dl = get_dataloader(
            cfg, test_ds, is_train=False, collate_fn=batch_collator
        )
    else:
        test_dl = None
    data = DataWrap(
        path=cfg.misc.tmp_path, train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl
    )
    return data
