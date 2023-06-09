"""
Model for EvRel
"""
import sys
import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN1


from vidsitu_code.mdl_sf_base import get_head_dim
from transformers import RobertaForSequenceClassification, RobertaModel


from transformers import BartForConditionalGeneration, BartModel, BartTokenizerFast


class Simple_EvRel_Bart(nn.Module):

    def __init__(self, cfg, comm):
        super().__init__()
        self.full_cfg = cfg
        self.cfg = cfg.mdl
        self.comm = comm
        self.build_model()

    def build_model(self):
        self.bart_mdl = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizerFast.from_pretrained('roberta-base')
        
        og_stdout = sys.stdout
        with open("/home/andrew/codebases/VidSitu/tmp/output_files/evrel_bart_pretraining_decoder_test1.txt", "a") as f:
            sys.stdout = f
            print(self.bart_mdl)
            sys.stdout = og_stdout
        f.close()
 
        return

    def forward(self, inp):
        src_toks1 = inp["bart_full_evrel_seq_out"]
        src_attn1 = inp["bart_full_evrel_seq_out_lens"]

        #print("\ninput\n", src_toks1, "\ninput length\n", src_attn1, "\nlabels\n", inp["full_evrel_labs"])
        #print("\nsrc_toks1\n", src_toks1, "\nsrc_toks1.shape\n", src_toks1.shape)
        #print("\nlabels\n", inp["full_evrel_labs"])


        B = 1
        num_ev = 1
        num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)

        #print("\nbatch length\n", num_seq_eg)

        out = self.bart_mdl(
            input_ids=src_toks,
            attention_mask=src_attn_mask,
            return_dict=True,
            labels=inp["bart_full_evrel_labs"]

        )

        #print("\n\nBart loss\n", out["loss"])
        
        logits = out["logits"]
        #print("\nlogits size\n", logits.size())
        #labels = inp["bart_full_evrel_labs"]
        
        '''
        correction_factor = 0
        target_tokens = [0, 2, 225, 241, 879, 3245, 3368, 4289, 4685, 6058, 9764, 9849, 10845, 16354]
        for i in target_tokens:
            logits[:, :, i] += correction_factor
        logits -= correction_factor
        
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        #print("\nCaluculated loss:\n", loss)
        '''

        batch_mdl_out_list = []
        evrel_dict = {0:0, "before":1 , "after":2}
        for i in range(num_seq_eg):
            #print("\nlogits size\n", out["logits"].size())
            #print("\nsliced logits\n", logits[i][1:10], i)
            #print("\nsliced logits*\n", logits[i, 1:10], i)
            
            '''
            probs = logits[i, 1:10].softmax(dim=0)
            values, predictions = probs.topk(1)  
            #print("\nvalues\n", values)
            #print("\npredictions\n", predictions, i)
            '''
            predictions = torch.zeros(6).long().cuda() 
            for j in range(6):
                predictions[j] = torch.argmax(logits[i, j])
            
           

            #predictions = torch.flatten(predictions)
            #print("\npredictions\n", predictions, i)

            decoded_seq = self.tokenizer.decode(predictions.tolist()).split()
            
            #print("\ndecoded sequence\n", decoded_seq, i)
            
            og_stdout = sys.stdout
            with open("/home/andrew/codebases/VidSitu/tmp/output_files/evrel_bart_pretraining_decoder_test1.txt", "a") as f:
                sys.stdout = f
                print("\ndecoded sequence\n", decoded_seq, i)
                sys.stdout = og_stdout
            f.close()

            mdl_out_list = [[[-1, 0, 0, 0],
                             [-1, 0, 0, 0],
                             [-1, 0, 0, 0]],

                            [[-1, 0, 0, 0],
                             [-1, 0, 0, 0],
                             [-1, 0, 0, 0]],

                            [[-1, 0, 0, 0],
                             [-1, 0, 0, 0],
                             [-1, 0, 0, 0]],

                            [[-1, 0, 0, 0],
                             [-1, 0, 0, 0],
                             [-1, 0, 0, 0]]]

            for j, evrel_pred in enumerate(decoded_seq):
                if j > 3:
                    break

                if "before" in evrel_pred:
                    mdl_out_list[j][0][1] = 1
                    mdl_out_list[j][1][1] = 1
                    mdl_out_list[j][2][1] = 1
                elif "after" in evrel_pred:
                    mdl_out_list[j][0][2] = 1
                    mdl_out_list[j][1][2] = 1
                    mdl_out_list[j][2][2] = 1
                elif "intent" in evrel_pred:
                    mdl_out_list[j][0][3] = 1
                    mdl_out_list[j][1][3] = 1
                    mdl_out_list[j][2][3] = 1
                else:
                    mdl_out_list[j][0][0] = 1
                    mdl_out_list[j][1][0] = 1
                    mdl_out_list[j][2][0] = 1


            batch_mdl_out_list.append(mdl_out_list)

        #out["loss"] = loss
        out["mdl_out"] = torch.tensor(batch_mdl_out_list).float()      
        print("\nout[mdl_out]\n", out["mdl_out"])
        
        #print("\ngt_lst: \n", inp["gt_lst"])
        out["gt_lst"] = inp["gt_lst"]


        #torch.set_printoptions(threshold=10_000)
        #print("\nBart out\n", out)

        return out


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
        
        #print("\nsrc_toks1\n", src_toks1, "\nsrc_toks1.shape\n", src_toks1.shape)
        #print("\nsrc_attn1\n", src_attn1, "\nsrc_attn1.shape\n", src_attn1.shape)


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
        labels = inp["rob_full_evrel_labs"]
        
        #print("\ngt_st: ", inp["gt_lst"])
        #print("\nlabels: \n", labels)
       
        '''
        relation_freq_list = [0, 16421/16421, 23357/16421, 32707/16421, 24579/16421]
        
        labels_flattened = labels.clone().detach().flatten().tolist()
        freq_sum = 0.0
        for i in labels_flattened:
            freq_sum += relation_freq_list[i]
        norm_value = freq_sum / len(labels_flattened)
        '''
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        ) #/ norm_value
        
        
        out["gt_lst"] = inp["gt_lst"]
        out["loss"] = loss
        out["mdl_out"] = logits.view(B, num_ev, num_seq_eg, -1)#[:,:,:,:4]
        #print(out["mdl_out"].shape)
        #print("\nmdl_out: \n", out["mdl_out"])
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
        head_dim = get_head_dim(self.full_cfg)

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
        return inp["evrel_seq_out_ones"], inp["evrel_seq_out_ones_lens"]

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


class SFPret_OnlyVb_SimpleEvRel(SFPret_SimpleEvRel):
    def get_src(self, inp):
        return inp["evrel_vbonly_out_ones"], inp["evrel_vbonly_out_ones_lens"]


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
