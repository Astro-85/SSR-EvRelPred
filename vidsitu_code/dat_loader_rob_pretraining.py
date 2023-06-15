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
        self.vsitu_frm_dir = Path(self.cfg.video_frms_tdir)
        split_files_cfg = self.cfg.split_files_lb
        vsitu_ann_files_cfg = self.cfg.vsitu_ann_files_lb
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
        
        '''
        if "valid" in split_type or "test" in split_type:
            vseg_info_lst = read_file_with_assertion(vinfo_files_cfg[split_type])
            vsitu_vinfo_dct = {}
            for vseg_info in vseg_info_lst:
                vseg = vseg_info["vid_seg_int"]
                assert vseg not in vsitu_vinfo_dct
                assert len(vseg_info["vbid_lst"]["Ev1"]) >= 9
                vid_seg_ann_lst = [
                    {
                        f"Ev{eix}": {"VerbID": vseg_info["vbid_lst"][f"Ev{eix}"][ix]}
                        for eix in range(1, 6)
                    }
                    for ix in range(len(vseg_info["vbid_lst"]["Ev1"]))
                ]
                vseg_info["vb_id_lst_new"] = vid_seg_ann_lst
                vsitu_vinfo_dct[vseg] = vseg_info
            self.vsitu_vinfo_dct = vsitu_vinfo_dct
        '''

    def __len__(self) -> int:
        if self.full_cfg.debug_mode:
            return 30
        return len(self.vseg_lst)

    def __getitem__(self, index: int) -> Dict:
        return self.itemgetter(index)
    
    """
    def read_img(self, img_fpath):
        # Output should be H x W x C
        
        img = Image.open(img_fpath).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img)

        return img_np

    def get_vb_data(self, vid_seg_ann_lst: List):
        voc_to_use = self.comm.vb_id_vocab
        label_lst_all_ev = []
        label_lst_mc = []
        for ev in range(1, 6):
            label_lst_one_ev = []
            for vseg_aix, vid_seg_ann in enumerate(vid_seg_ann_lst):
                if vseg_aix == 10:
                    break
                vb_id = vid_seg_ann[f"Ev{ev}"]["VerbID"]

                if vb_id in voc_to_use.indices:
                    label = voc_to_use.indices[vb_id]
                else:
                    label = voc_to_use.unk_index
                label_lst_one_ev.append(label)
            label_lst_all_ev.append(label_lst_one_ev)
            mc = Counter(label_lst_one_ev).most_common(1)
            label_lst_mc.append(mc[0][0])

        label_tensor_large = torch.full((5, 10), voc_to_use.pad_index, dtype=torch.long)
        label_tensor_large[:, : len(vid_seg_ann_lst)] = torch.tensor(label_lst_all_ev)
        label_tensor10 = label_tensor_large
        label_tensor = torch.tensor(label_lst_mc)

        return {"label_tensor10": label_tensor10, "label_tensor": label_tensor}
    """
    
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

        #print("\n-----------------------------------------------------------\n")
        #print("\nvid_seg_ann_lst: \n", vid_seg_ann_lst)
        
        # BEGIN CODE FOR VISCOM PRETRAINING
        
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
        
        #print('s1: ', s1)
        #print('s2: ', s2)
        #print('s3: ', s3)
        #print('s4: ', s4)
        #print('s5: ', s5)
        #print('gt_list: ', gt_lst)
        #print('')


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
            
        out_dct["rob_full_evrel_seq_out"] = torch.tensor(full_out_evrel_seq_by_ev).long()
        out_dct["rob_full_evrel_seq_out_lens"] = torch.tensor(full_out_evrel_seq_by_ev_lens).long()
        out_dct["rob_full_evrel_labs"] = torch.tensor(evrel_labs).long()
        out_dct["gt_lst"] = torch.tensor(gt_lst).long()
            
        return out_dct
        
        # END CODE FOR VISCOM PRETRAINING
        
        '''
        for ev in range(1, 6):
            only_vb_lst = []
            seq_lst = []
            seq_lst_lens = []
            evrel_lst = []
            evrel_seq_lst = []
            
            # Loops only once since vid_seg_ann is the only item in vid_seg_ann_lst
            i = 0
            for vsix, vid_seg_ann in enumerate(vid_seg_ann_lst):
                i += 1
                print(i)
                print("\nvsix: \n", vsix)
                print("\nvid_seg_ann: \n", vid_seg_ann)
                ann1 = vid_seg_ann[f"Ev{ev}"]    # ann1 of event number specified by ev
                print("\nann1: \n", ann1)
                vb_id = ann1["VerbID"]
                print("\nvb_id: \n", vb_id)
                arg_lst = list(ann1["Arg_List"].keys())
                print("\narg_lst: \n", arg_lst)
                arg_lst_sorted = sorted(arg_lst, key=lambda x: int(ann1["Arg_List"][x]))
                arg_str_dct = ann1["Args"]    # Turns arg_list into dictionary
                print("\narg_str_dct: \n", arg_str_dct)

                seq = ""
                
                
                # Encodes verb ID
                if vb_id in addn_word_voc:
                    prefix_lst = [addn_word_voc[vb_id]]
                else:
                    prefix_lst = word_voc.encode(vb_id)
                    
                    
                #print("prefix_lst: ", prefix_lst)
                
                # Concatenates all arguments into a single string seq
                for ag in arg_lst_sorted:
                    arg_str = arg_str_dct[ag]
                    #print("arg_str: ", arg_str)
                    ag_n = arg_mapper(ag)
                    #print("ag_n: ", ag_n)
                    
                    # If evrel, then don't add extra arguments like manner, location to seq
                    if not (is_evrel and self.cfg.evrel_trimmed):
                        seq += " " + enclose_ag_st(ag_n, arg_str)
                    else:
                        if self.cfg.evrel_trimmed and ag_n in agset:
                            seq += " " + enclose_ag_st(ag_n, arg_str)
                            
                    #print("seq: ", seq)
                
                # Stores event relation in evr unless event has no relation (Ev3)
                if "EvRel" in ann1:
                    evr = ann1["EvRel"]
                else:
                    evr = "Null"
                
                #print("\nevr:\n", evr)
                # Evrel dict- Null:0 Causes:1 Reaction To:2 Enables:3 NoRel:4
                evrel_curr = self.comm.evrel_dct[evr]
                #print("evrel_curr: ", evrel_curr)
                evrel_lst.append(evrel_curr)
                #print("evrel_lst: ")
                #print(' '.join(map(str, evrel_lst)))
                evrel_seq_lst.append((vb_id, seq))
                #print("evrel_seq_lst: ")
                #print(' '.join(map(str, evrel_seq_lst)))
                
                
                # This will also happen every time since loop only iterates once
                if vsix == 0:
                    vb_id_lst.append(prefix_lst[0])
                    seq_id_lst.append(seq)
                    
                    
                # RobertaTokenizerFast is identical to BartTokenizerFast
                
                # Encode and pad seq
                seq_padded, seq_len = pad_words_new(
                    seq,
                    max_len=60,
                    wvoc=word_voc,
                    append_eos=True,
                    use_hf=True,
                    pad_side="right",
                    prefix_lst=prefix_lst,
                )
                
                # Encode and pad verb ID
                only_vb_padded, _ = pad_words_new(
                    vb_id,
                    max_len=5,
                    wvoc=word_voc,
                    append_eos=False,
                    use_hf=True,
                    pad_side="right",
                )
                seq_padded = seq_padded.tolist()
                seq_lst.append(seq_padded)
                seq_lst_lens.append(seq_len)
                only_vb_padded = only_vb_padded.tolist()
                only_vb_lst.append(only_vb_padded)

                
                #print("\nAndrew outputted: ")
                #print("seq_padded: ", seq_padded)
                #print("seq_lst: ", seq_lst)
                #print("only_vb_padded: ", only_vb_padded)
                #print("only_vb_lst: ", only_vb_lst)
                #print("evrel_lst: ", evrel_lst)
                
            
            #print("evrel_lst: ", evrel_lst)
            #print("evrel_seq_lst: ", evrel_seq_lst)
            
            # Contains encoded verbs, seqs, and relations for all events in a 10s segment
            seq_lst_all_ev.append(seq_lst)
            only_vb_lst_all_ev.append(only_vb_lst)
            seq_lst_all_ev_lens.append(seq_lst_lens)
            evrel_lst_all_ev.append(evrel_lst)    # Contains list of only event relations
            evrel_seq_lst_all_ev.append(evrel_seq_lst)    # Contains list of only verb ID's and seq
            

            #print("vb_id_lst: ", vb_id_lst)
            #print("seq_id_lst: ", seq_id_lst)
            
            
        assert len(vb_id_lst) == len(seq_id_lst)
        assert len(vb_id_lst) == 5
        seq_lst_all_ev_comb = []
        space_sep = word_voc(" ")["input_ids"]
        seq_lst_all_ev_comb = []
        vb_lst_all_ev_comb = []
        for vbi in vb_id_lst:
            vb_lst_all_ev_comb += [vbi, space_sep[0]]

        seq_lst_all_ev_comb = vb_lst_all_ev_comb[:]
        for ev_ix, ev in enumerate(range(1, 6)):
            seq_lst_all_ev_comb += word_voc(seq_id_lst[ev_ix])["input_ids"]

        max_full_seq_len = 60 * 5
        seq_out_ev_comb_tok, seq_out_ev_comb_tok_len = pad_tokens(
            seq_lst_all_ev_comb,
            pad_index=word_voc.pad_token_id,
            pad_side="right",
            append_eos=True,
            eos_index=word_voc.eos_token_id,
            max_len=max_full_seq_len,
        )

        out_dct = {
            "seq_out_by_ev": torch.tensor(seq_lst_all_ev).long(),
            "evrel_out_by_ev": torch.tensor(evrel_lst_all_ev).long(),
            "seq_out_lens_by_ev": torch.tensor(seq_lst_all_ev_lens).long(),
            "seq_out_ev_comb_tok": torch.tensor([seq_out_ev_comb_tok.tolist()]).long(),
            "seq_out_ev_comb_tok_len": torch.tensor([seq_out_ev_comb_tok_len]).long(),
            "vb_out_by_ev": torch.tensor(only_vb_lst_all_ev).long(),
            "vb_out_ev_comb_tok": torch.tensor([vb_lst_all_ev_comb]).long(),
        }

        def get_new_s(s):
            return s[0] + s[1]

        if is_evrel:
            out_evrel_seq_by_ev = []
            out_evrel_seq_by_ev_lens = []
            out_evrel_labs_by_ev = []

            out_evrel_tok_ids_by_ev = []
            evrel_wvoc = self.comm.rob_hf_tok
            
# BEGIN ADDED CODE FOR BART
            
            full_out_evrel_seq_by_ev = []
            full_out_evrel_seq_by_ev_lens = []                      
            
            
            full_out_evrel_seq_lst = []
            full_out_evrel_seq_lens = []
            full_out_evrel_labs_by_ev = []
            
            for vix in range(len(vid_seg_ann_lst)):
                s1 = evrel_seq_lst_all_ev[0][vix]
                s2 = evrel_seq_lst_all_ev[1][vix]
                s3 = evrel_seq_lst_all_ev[2][vix]
                s4 = evrel_seq_lst_all_ev[3][vix]
                s5 = evrel_seq_lst_all_ev[4][vix]
                
                #print("\ns1\n", s1)
               

                s1_new = get_new_s(s1)
                s2_new = get_new_s(s2)
                s3_new = get_new_s(s3)
                s4_new = get_new_s(s4)
                s5_new = get_new_s(s5)
                
                #print("\ns1_new\n", s1_new)
                
                new_seq = evrel_wvoc(
                    s1_new + evrel_wvoc.sep_token + 
                    s2_new + evrel_wvoc.sep_token + 
                    s3_new + evrel_wvoc.sep_token + 
                    s4_new + evrel_wvoc.sep_token + 
                    s5_new
                )["input_ids"]
                
                #print("\nnew_seq\n", new_seq)

                
                
                new_seq_pad, new_seq_msk = pad_tokens(
                    new_seq,
                    pad_index=evrel_wvoc.pad_token_id,
                    pad_side="right",
                    append_eos=False,
                    eos_index=evrel_wvoc.eos_token_id,
                    max_len=240,
                )
                
                #print("\nnew seq pad\n", new_seq_pad)


                full_out_evrel_seq_lst = new_seq_pad.tolist()
                full_out_evrel_seq_lens = new_seq_msk 
                    
                #print("full_new_seq_pad: \n", new_seq_pad)
                #print("full_out_evrel_seq_lst: \n", full_out_evrel_seq_lst)
                
                full_out_evrel_labs_lst = []
                
                
                evrel_dict = {0:0, 1:"causes", 2:"reaction", 3:"enables", 4:"unrelated"}
                evrel_seq_string = ""
                for evix in [0, 1, 3, 4]:
                    evrel_seq_string += evrel_dict[evrel_lst_all_ev[evix][vix]]
                    if evix != 4:
                        evrel_seq_string += " "
                
                #print("\nevrel_seq_string\n", evrel_seq_string)
                
                new_evrel_seq = evrel_wvoc(evrel_seq_string)["input_ids"]
                
                #print("\nnew_evrel_seq\n", new_evrel_seq)
                
                #print("\npad_token_id\n", evrel_wvoc.pad_token_id)
                #print("\neos_token_id\n", evrel_wvoc.eos_token_id)
                
                new_evrel_seq_pad, new_evrel_seq_msk = pad_tokens(
                    new_evrel_seq,
                    pad_index=1,
                    pad_side="right",
                    append_eos=False,
                    eos_index=evrel_wvoc.eos_token_id,
                    max_len=240,
                )
                
                
                new_evrel_seq_pad_list = []
                new_evrel_seq_pad_list = new_evrel_seq_pad.tolist()
                #print("\nnew_evrel_seq_pad\n", new_evrel_seq_pad)
                    
                    
                        
                    #full_out_evrel_labs_lst.append(evrel_enc)
                    #full_out_evrel_labs_by_ev.append(evrel_enc)
                
                #print("\nfull_out_evrel_labs_lst:\n", full_out_evrel_labs_lst)
                #print("\nfull_out_evrel_labs_by_ev:\n", full_out_evrel_labs_by_ev)
            
                #full_out_evrel_seq_by_ev.append(full_out_evrel_seq_lst)
                #full_out_evrel_seq_by_ev_lens.append(full_out_evrel_seq_lens)
                #full_out_evrel_labs_by_ev.append(new_evrel_seq_pad_list)
                    
                
            out_dct["bart_full_evrel_seq_out"] = torch.tensor(full_out_evrel_seq_lst).long()
            out_dct["bart_full_evrel_seq_out_lens"] = torch.tensor(full_out_evrel_seq_lens).long()
            out_dct["bart_full_evrel_labs"] = torch.tensor(new_evrel_seq_pad_list).long()
            
            #print("\nout_dct[full_evrel_seq_out]: \n", out_dct["full_evrel_seq_out"])
            #print("\nout_dct[bart_full_evrel_labs]\n", out_dct["bart_full_evrel_labs"])
            
            #print("out_dct[full_evrel_seq_out_lens]: \n", out_dct["full_evrel_seq_out_lens"])
            
            
# END ADDED CODE FOR BART




# BEGIN ADDEED CODE FOR FULL SEQUENCE ROBERTA

            full_out_evrel_seq_by_ev = []
            full_out_evrel_seq_by_ev_lens = []                      
            
            for evix in [0, 1, 3, 4]:
                full_out_evrel_seq_lst = []
                full_out_evrel_seq_lens = []  
                
                for vix in range(len(vid_seg_ann_lst)):
                    s1 = evrel_seq_lst_all_ev[0][vix]
                    s2 = evrel_seq_lst_all_ev[1][vix]
                    s3 = evrel_seq_lst_all_ev[2][vix]
                    s4 = evrel_seq_lst_all_ev[3][vix]
                    s5 = evrel_seq_lst_all_ev[4][vix]
                    evcurr_seq = evrel_seq_lst_all_ev[evix][vix]
                    
                    s0_new = get_new_s(evcurr_seq)

                    s1_new = get_new_s(s1)
                    s2_new = get_new_s(s2)
                    s3_new = get_new_s(s3)
                    s4_new = get_new_s(s4)
                    s5_new = get_new_s(s5)

                    
                    if evix == 0:
                        s1_new = "*" + s1_new
                        s3_new = "**" + s3_new
                        new_seq = evrel_wvoc(
                            s1_new + evrel_wvoc.sep_token + 
                            s2_new + evrel_wvoc.sep_token + 
                            s3_new + evrel_wvoc.sep_token + 
                            s4_new + evrel_wvoc.sep_token + 
                            s5_new
                        )["input_ids"]
                    elif evix == 1:
                        s2_new = "*" + s2_new
                        s3_new = "**" + s3_new
                        new_seq = evrel_wvoc(
                            s1_new + evrel_wvoc.sep_token + 
                            s2_new + evrel_wvoc.sep_token + 
                            s3_new + evrel_wvoc.sep_token + 
                            s4_new + evrel_wvoc.sep_token + 
                            s5_new
                        )["input_ids"]
                    elif evix == 3:
                        s3_new = "*" + s3_new
                        s4_new = "**" + s4_new
                        new_seq = evrel_wvoc(
                            s1_new + evrel_wvoc.sep_token + 
                            s2_new + evrel_wvoc.sep_token + 
                            s3_new + evrel_wvoc.sep_token + 
                            s4_new + evrel_wvoc.sep_token + 
                            s5_new
                        )["input_ids"]
                    else:
                        s3_new = "*" + s3_new
                        s5_new = "**" + s5_new
                        new_seq = evrel_wvoc(
                            s1_new + evrel_wvoc.sep_token + 
                            s2_new + evrel_wvoc.sep_token + 
                            s3_new + evrel_wvoc.sep_token + 
                            s4_new + evrel_wvoc.sep_token + 
                            evrel_wvoc.unk_token + s5_new
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
                    
                    #print("full_new_seq_pad: \n", new_seq_pad)
                    #print("full_out_evrel_seq_lst: \n", full_out_evrel_seq_lst)

                full_out_evrel_seq_by_ev.append(full_out_evrel_seq_lst)
                full_out_evrel_seq_by_ev_lens.append(full_out_evrel_seq_lens)
                    
                
            out_dct["rob_full_evrel_seq_out"] = torch.tensor(full_out_evrel_seq_by_ev).long()
            out_dct["rob_full_evrel_seq_out_lens"] = torch.tensor(full_out_evrel_seq_by_ev_lens).long()
            
            #print("out_dct[full_evrel_seq_out]: \n", out_dct["full_evrel_seq_out"])
            #print("out_dct[full_evrel_seq_out_lens]: \n", out_dct["full_evrel_seq_out_lens"])



# END ADDED CODE FOR FULL SEQUENCE ROBERTA

            
            for evix in [0, 1, 3, 4]:
                out_evrel_seq_lst = []
                out_evrel_seq_lens = []
                out_evrel_tok_ids_lst = []
                out_evrel_labs_lst = []
                for vix in range(len(vid_seg_ann_lst)):
                    ev3_seq = evrel_seq_lst_all_ev[2][vix]
                    evcurr_seq = evrel_seq_lst_all_ev[evix][vix]

                    if evix < 2:
                        s1 = evcurr_seq
                        s2 = ev3_seq
                    else:
                        s1 = ev3_seq
                        s2 = evcurr_seq
                    s1_new = get_new_s(s1)
                    s2_new = get_new_s(s2)

                    new_seq_noevrel = evrel_wvoc(
                        s1_new + evrel_wvoc.sep_token + s2_new
                    )["input_ids"]

                    new_seq = new_seq_noevrel

                    new_seq_pad, new_seq_msk = pad_tokens(
                        new_seq,
                        pad_index=evrel_wvoc.pad_token_id,
                        pad_side="right",
                        append_eos=False,
                        eos_index=evrel_wvoc.eos_token_id,
                        max_len=120,
                    )

                    evrel_out = evrel_lst_all_ev[evix][vix]
                    
                    out_evrel_labs_lst.append(evrel_out)
                    out_evrel_seq_lst.append(new_seq_pad.tolist())
                    out_evrel_seq_lens.append(new_seq_msk)
                    
                    #print("new_seq_pad: \n", new_seq_pad)
                    #print("out_evrel_seq_lst: \n", out_evrel_seq_lst)
                

                out_evrel_seq_by_ev.append(out_evrel_seq_lst)
                out_evrel_seq_by_ev_lens.append(out_evrel_seq_lens)
                out_evrel_tok_ids_by_ev.append(out_evrel_tok_ids_lst)
                out_evrel_labs_by_ev.append(out_evrel_labs_lst)

            #print("out_evrel_labs_by_ev: ", out_evrel_labs_by_ev)
            out_dct["evrel_seq_out"] = torch.tensor(out_evrel_seq_by_ev).long()
            out_dct["evrel_seq_out_lens"] = torch.tensor(
                out_evrel_seq_by_ev_lens
            ).long()
            
            #print("out_dct[evrel_seq_out]: \n", out_dct["evrel_seq_out"])
            #print("out_dct[evrel_seq_out_lens]: \n",  out_dct["evrel_seq_out_lens"])

            out_dct["evrel_labs"] = torch.tensor(out_evrel_labs_by_ev).long()
            #print("gt labels:\n", out_dct["evrel_labs"])

            out_evrel_seq_one_by_ev = []
            out_evrel_seq_onelens_by_ev = []
            out_evrel_vb_one_by_ev = []
            out_evrel_vb_onelens_by_ev = []

            for evix in [0, 1, 2, 3, 4]:
                out_evrel_seq_one_lst = []
                out_evrel_seq_onelens_lst = []

                out_evrel_vbonly_one_lst = []
                out_evrel_vbonly_onelens_lst = []
                for vix in range(len(vid_seg_ann_lst)):
                    s1 = evrel_seq_lst_all_ev[evix][vix]
                    s1_new = get_new_s(s1)

                    new_seq_noevrel = evrel_wvoc(s1_new)["input_ids"]
                    new_seq_pad, new_seq_msk = pad_tokens(
                        new_seq_noevrel,
                        pad_index=evrel_wvoc.pad_token_id,
                        pad_side="right",
                        append_eos=False,
                        eos_index=evrel_wvoc.eos_token_id,
                        max_len=60,
                    )

                    out_evrel_seq_one_lst.append(new_seq_pad.tolist())
                    out_evrel_seq_onelens_lst.append(new_seq_msk)
                    vb_only_rob = evrel_wvoc(s1[0])["input_ids"]
                    vb_only_rob_pad, vb_only_rob_msk = pad_tokens(
                        vb_only_rob,
                        pad_index=evrel_wvoc.pad_token_id,
                        pad_side="right",
                        append_eos=False,
                        eos_index=evrel_wvoc.eos_token_id,
                        max_len=5,
                    )
                    out_evrel_vbonly_one_lst.append(vb_only_rob_pad.tolist())
                    out_evrel_vbonly_onelens_lst.append(vb_only_rob_msk)

                out_evrel_seq_one_by_ev.append(out_evrel_seq_one_lst)
                out_evrel_seq_onelens_by_ev.append(out_evrel_seq_onelens_lst)
                out_evrel_vb_one_by_ev.append(out_evrel_vbonly_one_lst)
                out_evrel_vb_onelens_by_ev.append(out_evrel_vbonly_onelens_lst)

            out_dct["evrel_seq_out_ones"] = torch.tensor(out_evrel_seq_one_by_ev).long()
            out_dct["evrel_seq_out_ones_lens"] = torch.tensor(
                out_evrel_seq_onelens_by_ev
            ).long()
            out_dct["evrel_vbonly_out_ones"] = torch.tensor(
                out_evrel_vb_one_by_ev
            ).long()
            out_dct["evrel_vbonly_out_ones_lens"] = torch.tensor(
                out_evrel_vb_onelens_by_ev
            ).long()



        
        return out_dct
        '''
    
    """
    def get_frms_all(self, idx):
        vid_seg_name = self.vseg_lst[idx]
        frm_pth_lst = [
            self.vsitu_frm_dir / f"{vid_seg_name}/{vid_seg_name}_{ix:06d}.jpg"
            for ix in range(1, 301)
        ]

        frms_by_ev_fast = []
        frms_by_ev_slow = []
        for ev in range(1, 6):
            ev_id = f"Ev{ev}"
            center_ix = self.comm.cent_frm_per_ev[ev_id]
            frms_ixs_for_ev = get_sequence(
                center_idx=center_ix,
                half_len=self.comm.frm_seq_len // 2,
                sample_rate=self.comm.sampling_rate,
                max_num_frames=300,
            )
            frm_pths_for_ev = [frm_pth_lst[ix] for ix in frms_ixs_for_ev]

            frms_for_ev = torch.from_numpy(
                np.stack([self.read_img(f) for f in frm_pths_for_ev])
            )

            frms_for_ev = tensor_normalize(
                frms_for_ev, self.sf_cfg.DATA.MEAN, self.sf_cfg.DATA.STD
            )

            # T x H x W x C => C x T x H x W
            frms_for_ev_t = (frms_for_ev).permute(3, 0, 1, 2)
            frms_for_ev_slow_fast = pack_pathway_output(self.sf_cfg, frms_for_ev_t)
            if len(frms_for_ev_slow_fast) == 1:
                frms_by_ev_fast.append(frms_for_ev_slow_fast[0])
            elif len(frms_for_ev_slow_fast) == 2:
                frms_by_ev_slow.append(frms_for_ev_slow_fast[0])
                frms_by_ev_fast.append(frms_for_ev_slow_fast[1])
            else:
                raise NotImplementedError

        out_dct = {}
        # 5 x C x T x H x W
        frms_all_ev_fast = np.stack(frms_by_ev_fast)
        out_dct["frms_ev_fast_tensor"] = torch.from_numpy(frms_all_ev_fast).float()
        if len(frms_by_ev_slow) > 0:
            frms_all_ev_slow = np.stack(frms_by_ev_slow)
            out_dct["frms_ev_slow_tensor"] = torch.from_numpy(frms_all_ev_slow).float()

        return out_dct

    def get_frm_feats_all(self, idx: int):
        vid_seg_name = self.vseg_lst[idx]
        vid_seg_feat_file = (
            Path(self.cfg.vsit_frm_feats_dir) / f"{vid_seg_name}_feats.npy"
        )
        vid_feats = read_file_with_assertion(vid_seg_feat_file, reader="numpy")
        vid_feats = torch.from_numpy(vid_feats).float()
        assert vid_feats.size(0) == 5
        return {"frm_feats": vid_feats}
    """
    """
    def get_label_out_dct(self, idx: int):
        vid_seg_name = self.vseg_lst[idx]
        if self.split_type == "train":
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            vid_seg_ann = vid_seg_ann_[0]
            label_out_dct = self.get_vb_data([vid_seg_ann])
        elif "valid" in self.split_type:
            vid_seg_ann_ = self.vsitu_vinfo_dct[vid_seg_name]["vb_id_lst_new"]
            assert len(vid_seg_ann_) >= 9
            label_out_dct = self.get_vb_data(vid_seg_ann_)
        else:
            raise NotImplementedError

        return label_out_dct
    """
    """
    def vb_only_item_getter(self, idx: int):
        frms_out_dct = self.get_frms_all(idx)

        frms_out_dct["vseg_idx"] = torch.tensor(idx)
        label_out_dct = self.get_label_out_dct(idx)
        out_dct = coalesce_dicts([frms_out_dct, label_out_dct])
        return out_dct
    """

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
