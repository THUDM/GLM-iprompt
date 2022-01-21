# -*- encoding: utf-8 -*-
'''
@File    :   base_strategy.py
@Time    :   2021/10/08 22:22:42
@Author  :   Ming Ding
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import torch
import torch.nn.functional as F


class MultigenStrategy:
    def __init__(self, num_beams, length_penalty=1., consider_end=False,
                end_tokens=[], invalid_slices=[], no_repeat_ngram_size=0, min_tgt_length=0,verifier=None,verifier_params=None,num_samples=12,factor=0.1,end_factor=0.1,tmp_factor=1.1,mode='chinese'):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.ngram = no_repeat_ngram_size
        self.min_tgt_length = min_tgt_length
        self.invalid_slices = invalid_slices
        self.consider_end = consider_end
        self.verifier=verifier
        self.verifier_params=verifier_params
        self.start_pos=0
        self.iprompt_start_pos=0
        self.iprompt_end_pos=0
        self.gen_start=0
        self._init_cache()
        self.factor=factor
        self.end_factor=end_factor
        self.num_samples=num_beams
        self.tmp_factor=tmp_factor
        self.mode=mode
        self.max_set=50000
        if self.mode=='english':
            self.max_set=50256
    
    def set_iprompt_factor(self,factor):
        self.factor=factor
    def set_verifier_params(self,params):
        self.verifier_params=params
        
    def set_verifier_param(self,param_idx,idx):
        self.verifier_params[idx]=param_idx
    
    def set_end_tokens(self,end_tokens):
        self.end_tokens=end_tokens
        
    def adjust_start(self,length):
        self.start_pos=length
        
    def set_start(self,start):
        self.iprompt_start_pos=start
    def set_end(self,end):
        self.iprompt_end_pos=end
    def set_gen_pos(self,pos):
        self.gen_pos=pos
    def set_sop_pos(self,pos):
        self.sop_pos=pos
    def set_ini_pos(self,pos):
        self.ini_pos=pos
    def set_model(self,model):
        self.model=model
    def get_pos(self):
        return self.ini_pos,self.iprompt_start_pos,self.iprompt_end_pos
    def _init_cache(self):
        
        self.end_beams = [] # list of LongTensors
        self.end_beams_penalized_scores = [] # list of LongTensors
        self.cached_beam_scores = 0 # [batch_size]
        self.actual_scores=0
        self.cached_beam_ngram_bans = [{} for i in range(self.num_beams)]
        self.is_done = False
        torch.cuda.empty_cache()
        
    
    def _add_end_beams(self, beam):
        
        self.end_beams.append(beam)
       
        if len(self.end_beams)>=self.num_samples:
            self.is_done=True
            
    def forward(self, logits, tokens, mems, use_ip=True,ban_end=False,max_length=375):
        batch_size, vocab_size = logits.shape
        #print(batch_size)
        seq_len = tokens.shape[-1]
        logits = logits.float()
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        if self.min_tgt_length > seq_len:
            for end_token in self.end_tokens:
                logits[..., end_token] = -65504
        if self.ngram > 0 and seq_len > self.ngram:
            for i in range(batch_size):
                ngram_prefix = tokens[i, -(self.ngram-1):].tolist() # TODO ngram=1
                for banned_index in self.cached_beam_ngram_bans[i].get(tuple(ngram_prefix), []):
                    logits[i, banned_index] = -65504
        for i in range(batch_size):
            logits[i,tokens[i]]-=1
        
        
    
        #emphasis on last word
        total_added=0
        times=0
        beam_continue = []
        scores_continue = []
        bans_continue = []
        mems_continue = []
        verify_scores=[]
        for i in range(batch_size):
            to_gen=True
            scores_for_sample=logits[i]*self.tmp_factor
            while to_gen:
                probs = F.softmax(scores_for_sample, dim=0)
                next_tokens = torch.multinomial(next_token_prob,
                    num_samples=5) # [2*nb]
                scores_for_sample[next_tokens]-=1000
            
                
                times+=1
                sampled_scores = next_token_scores[next_tokens]
            
            
            
                for j in range(len(next_tokens)):
                    if to_gen==False:
                        continue
                    if total_added>=self.num_samples:
                        break
                    if ban_end and next_tokens[j]>=self.max_set:
                        continue
                    beam = torch.cat((tokens[i], next_tokens[j:j+1]))
                    #print(beam,vocab_size,logits.shape)
                    if self.verifier is not None:
                        verify_score,beam=self.verifier(beam,self.verifier_params)
                    else:
                        verify_score=0
                        
                        
                    if verify_score<-100:
                        self._add_end_beams(beam)
                    
                    to_gen=False
                    if ((int(next_tokens[i]) in self.end_tokens) or (len(beam)>=max_length)) and not(ban_end):
                        self._add_end_beams(beam)
                    else:
                        beam_continue.append(beam)
                        mems_continue.append(mems[:, next_indices[i]])
                        # update caches
                        if self.ngram > 0:
                            bans = self.cached_beam_ngram_bans[next_indices[i]].copy()
                            ngram_prefix = tuple(tokens[next_indices[i], -(self.ngram-1):].tolist()) # TODO ngram=1
                            bans[ngram_prefix] = bans.get(ngram_prefix, tuple()) + (next_tokens[i],)
                            bans_continue.append(bans)
                    
        #print(total_added,len(beam_continue),self.num_beams)
        
        if len(beam_continue)>0:
            tokens = torch.stack(beam_continue)
            mems = torch.stack(mems_continue, dim=1)
            self.is_done=True
        torch.cuda.empty_cache()
        # TODO is_done
        return tokens, mems

    def finalize(self, tokens, mems):
        ret=self.end_beams
        self._init_cache()
        return ret
