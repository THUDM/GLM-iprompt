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

from SwissArmyTransformer.generation.autoregressive_sampling import evaluate_perplexity


    
class BeamSearchStrategy:
    def __init__(self, num_beams, length_penalty=1., consider_end=False,
                end_tokens=[], invalid_slices=[], no_repeat_ngram_size=0, min_tgt_length=0,verifier=None,verifier_params=None):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.ngram = no_repeat_ngram_size
        self.min_tgt_length = min_tgt_length
        self.invalid_slices = invalid_slices
        self.consider_end = consider_end
        self.verifier=verifier
        self.verifier_params=verifier_params
        self._init_cache()
    
    def set_verifier_params(self,params):
        self.verifier_params=params
        
    def set_verifier_param(self,param_idx,idx):
        self.verifier_params[idx]=param_idx
    
    def set_end_tokens(self,end_tokens):
        self.end_tokens=end_tokens
        
    def _init_cache(self):
        
        self.end_beams = [] # list of LongTensors
        self.end_beams_penalized_scores = [] # list of LongTensors
        self.cached_beam_scores = 0 # [batch_size]
        self.cached_beam_ngram_bans = [{} for i in range(self.num_beams)]
        self.is_done = False
        torch.cuda.empty_cache()
        
    
    def _add_end_beams(self, score, beam):
        score = score / ((1. + len(beam)) / 6) ** self.length_penalty # Magic number for OpenNMT
        #print(beam)
        for i in range(len(self.end_beams), -1, -1):
            if i == 0 or score < self.end_beams_penalized_scores[i-1]:
                break
        self.end_beams.insert(i, beam)
        self.end_beams_penalized_scores.insert(i, score)

        self.end_beams = self.end_beams[:self.num_beams]
        self.end_beams_penalized_scores = self.end_beams_penalized_scores[:self.num_beams]
        if len(self.end_beams)>2:
            self.is_done=True
            
    def forward(self, logits, tokens, mems):
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
        
        next_token_scores = F.log_softmax(logits, dim=-1) # [batch_size, vocab_size]
        prev_scores = self.cached_beam_scores
        if isinstance(self.cached_beam_scores, torch.Tensor):
            prev_scores = prev_scores[:, None].expand_as(next_token_scores)
        scores_for_sample=next_token_scores*0.9+prev_scores*0.3
        
        next_token_scores = next_token_scores + prev_scores
        
        next_token_scores = next_token_scores.view(batch_size * vocab_size)
        
        scores_for_sample=scores_for_sample.view(batch_size*vocab_size)
        #emphasis on last word
        total_added=0
        times=0
        beam_continue = []
        scores_continue = []
        bans_continue = []
        mems_continue = []
        while (total_added<self.num_beams):
            probs = F.softmax(scores_for_sample, dim=0)
            if (times>2 and len(self.end_beams)>0):
                self.is_done=True
                break
            if (times>5 and total_added>0):
                break
            
           
            next_tokens = torch.multinomial(probs,
                    num_samples=20) # [2*nb]
            scores_for_sample[next_tokens]-=1000
            
            if times==1:
                for i in range(batch_size):
                    next_tokens[-i*2-1]=vocab_size*i+43361
                    next_tokens[-i*2-2]=vocab_size*i+43359
            times+=1
            sampled_scores = next_token_scores[next_tokens]
            
            #print(next_tokens,next_token_scores)
            sampled_scores, _indices = torch.sort(sampled_scores, descending=True, dim=0)
            next_tokens = next_tokens[_indices]

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='trunc')
            next_tokens = next_tokens % vocab_size

                # select out end beams or continue beams
            if mems.shape[1] < batch_size:
                mems = mems.expand(-1, batch_size, -1, -1)
            
            for i in range(len(next_tokens)):
                if total_added>=self.num_beams:
                    break
                if next_tokens[i]>=50000:
                    continue
                beam = torch.cat((tokens[next_indices[i]], next_tokens[i:i+1]))
                #print(beam,vocab_size,logits.shape)
                verify_score=self.verifier(beam,self.verifier_params)
                if verify_score<-100:
                    #print(self.verifier_params[0].DecodeIds(beam.cpu().tolist()),verify_score)
                    continue
                    
                total_added+=1
                if int(next_tokens[i]) in self.end_tokens:
                    #print(self.verifier_params[0].DecodeIds(beam.cpu().tolist()),next_tokens[i])
                    self._add_end_beams(next_token_scores[i], beam)
                elif len(beam_continue) < self.num_beams:
                    beam_continue.append(beam)
                    mems_continue.append(mems[:, next_indices[i]])
                        # update caches
                    scores_continue.append(sampled_scores[i]+verify_score)
                    if self.ngram > 0:
                        bans = self.cached_beam_ngram_bans[next_indices[i]].copy()
                        ngram_prefix = tuple(tokens[next_indices[i], -(self.ngram-1):].tolist()) # TODO ngram=1
                        bans[ngram_prefix] = bans.get(ngram_prefix, tuple()) + (next_tokens[i],)
                        bans_continue.append(bans)
                else:
                    break
        #print(total_added,len(beam_continue),self.num_beams)
        
        if len(beam_continue)>0:
            tokens = torch.stack(beam_continue)
            mems = torch.stack(mems_continue, dim=1)
            self.cached_beam_scores = torch.tensor(scores_continue, device=logits.device)
            self.cached_beam_ngram_bans = bans_continue
            
        del mems_continue
        torch.cuda.empty_cache()
        # TODO is_done
        return tokens, mems

    def finalize(self, tokens, mems):
        ret=self.end_beams
        mems=None
        '''
        if self.consider_end:
            for i in range(tokens.shape[0]):
                self._add_end_beams(self.cached_beam_scores[i], tokens[i])
            mems = None
            ret = self.end_beams
        else:
            ret = tokens
        '''
        self._init_cache()
        return ret, mems
