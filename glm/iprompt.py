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

from ppl import ppl
def compute_ip(model,ini_pos,start_pos,end_pos,gen_pos,sop_pos,beams,tokenizer):
    #compute new beams
    new_beams=[]
    targets=[]
    cls=tokenizer.get_command('ENC').Id
    smask=tokenizer.get_command('sMASK').Id
    eos=tokenizer.get_command('eos').Id
    sop=tokenizer.get_command('sop').Id
    
    for i in range(len(beams)):
        bm=beams[i].cpu().tolist()
        
        new_bm=[cls]+bm[ini_pos:start_pos]+[smask]+bm[end_pos:gen_pos]+bm[sop_pos+1:]+bm[gen_pos+1:sop_pos-1]+[eos]
        
        new_beams.append(new_bm)
        new_tag=bm[start_pos:end_pos]
        #dec=tokenizer.DecodeIds(new_bm)
        #dec2=tokenizer.DecodeIds(new_tag)
        #print(dec,dec2)
        targets.append(new_tag)
        
        
            
    #print(input_seq,target_seq)
    return ppl(model,new_beams,targets,1+start_pos-ini_pos,sop=sop)

def compute_p(model,sop_pos,beam,sop=50006):
    new_beams=[]
    targets=[]
    for i  in range(len(beam)):
        bm=beam[i].cpu().tolist()
        new_bm=bm[:sop_pos]
        new_tag=bm[sop_pos+1:]
        new_beams.append(new_bm)
        targets.append(new_tag)
    return ppl(model,new_beams,targets,sop_pos,sop=sop)
class iPromptStrategy:
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
        self.num_samples=num_beams*3
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
        
    
    def _add_end_beams(self, score, verify_score,beam):
        if True:
            sop=50006
            if self.mode=='english':
                sop=50257
            score=compute_p(self.model,self.sop_pos,[beam],sop=sop)
            if self.verifier is not None:
                verify_score=self.verifier(beam,self.verifier_params)+verify_score
            
            
        score = score / (len(beam)-self.sop_pos-1)
        if self.mode=='chinese':
            ip_score=compute_ip(self.model,self.ini_pos,self.iprompt_start_pos,self.iprompt_end_pos,self.gen_pos,self.sop_pos,[beam],self.verifier_params[0])
        if self.mode=='english':
            ip_score=compute_ip(self.model,self.ini_pos,self.iprompt_start_pos,self.iprompt_end_pos,self.gen_pos,self.sop_pos,[beam],self.verifier_params[0])
        #print(self.verifier_params[0].DecodeIds(beam.cpu().tolist()),score,ip_score)
        #print(score,verify_score,ip_score)
        score=score*self.end_factor+ip_score[0]*(1-self.end_factor)+verify_score
        #print(self.verifier_params[0].DecodeIds(beam.cpu().tolist()),score)
        for i in range(len(self.end_beams), -1, -1):
            if i == 0 or score < self.end_beams_penalized_scores[i-1]:
                break
        #print(self.verifier_params[0].DecodeIds(beam.cpu().tolist()),score)
        self.end_beams.insert(i, beam)
        self.end_beams_penalized_scores.insert(i, score)

        #self.end_beams = self.end_beams[:self.num_samples+2]
        #self.end_beams_penalized_scores = self.end_beams_penalized_scores[:self.num_samples+2]
        if len(self.end_beams)>self.num_samples:
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
        
        next_token_scores = F.log_softmax(logits, dim=-1) # [batch_size, vocab_size]
        prev_scores = self.cached_beam_scores
        act_scores=self.actual_scores
        if isinstance(self.cached_beam_scores, torch.Tensor):
            prev_scores = prev_scores[:, None].expand_as(next_token_scores)
            act_scores=act_scores[:,None].expand_as(next_token_scores)
        scores_for_sample=next_token_scores*self.tmp_factor+act_scores*0.75
        
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
        verify_scores=[]
        while (total_added<self.num_beams):
            probs = F.softmax(scores_for_sample, dim=0)
            if (times>2 and len(self.end_beams)>0):
                self.is_done=True
                break
            if (times>5 and total_added>0):
                break
            
           
            next_tokens = torch.multinomial(probs,
                    num_samples=self.num_samples*2) # [2*nb]
            scores_for_sample[next_tokens]-=1000
            
            if not(ban_end) and (times==1):
                
                for i in range(batch_size):
                    if 43361 in self.end_tokens:
                        next_tokens[-i*2-1]=vocab_size*i+43361
                    if 43359 in self.end_tokens:
                        next_tokens[-i*2-2]=vocab_size*i+43359
            times+=1
            sampled_scores = next_token_scores[next_tokens]
            
            #print(next_tokens,next_token_scores)
            

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='trunc')
            next_tokens = next_tokens % vocab_size

                # select out end beams or continue beams
            if mems.shape[1] < batch_size:
                mems = mems.expand(-1, batch_size, -1, -1)
            
            for i in range(len(next_tokens)):
                if self.is_done:
                    continue
                if total_added>=self.num_samples:
                    break
                if ban_end and next_tokens[i]>=self.max_set:
                    continue
                if next_tokens[i]>=self.max_set+10:
                    continue
                beam = torch.cat((tokens[next_indices[i]], next_tokens[i:i+1]))
                #print(beam,vocab_size,logits.shape)
                if self.verifier is not None:
                    verify_score=self.verifier(beam,self.verifier_params)
                else:
                    verify_score=0
                    
                if verify_score<-100:
                    #print(self.verifier_params[0].DecodeIds(beam.cpu().tolist()),verify_score)
                    continue
                    
                total_added+=1
                if ((int(next_tokens[i]) in self.end_tokens) or (len(beam)>=max_length)) and not(ban_end):
                    self._add_end_beams(next_token_scores[i],verify_score, beam)
                elif len(beam_continue) < self.num_samples:
                    beam_continue.append(beam)
                    mems_continue.append(mems[:, next_indices[i]])
                        # update caches
                    scores_continue.append(sampled_scores[i])
                    verify_scores.append(verify_score)
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
            if use_ip:
                iprompt_scores=compute_ip(self.model,self.ini_pos,self.iprompt_start_pos,self.iprompt_end_pos,self.gen_pos,self.sop_pos,beam_continue,self.verifier_params[0])
                self.actual_scores=self.cached_beam_scores*self.factor+iprompt_scores*(1-self.factor)+torch.tensor(verify_scores,device=logits.device)
                self.actual_scores, _indices = torch.sort(self.actual_scores, descending=True, dim=0)
                self.actual_scores=self.actual_scores[:self.num_beams]
                _indices=_indices[:self.num_beams]
                tokens = tokens[_indices]
                self.cached_beam_scores=self.cached_beam_scores[_indices]
                #print(mems.shape)
                mems=mems[:,_indices,:,:]
                #print(mems.shape)
                if self.ngram>0:
                    self.cached_beam_ngram_bans=[]
                    for index in range(len(_indices)):
                        self.cached_beam_ngram_bans .append( bans_continue[_indices[index]])
            else:
                self.actual_scores=self.cached_beam_scores*self.factor+torch.tensor(verify_scores,device=logits.device)
                if self.ngram>0:
                    self.cached_beam_ngram_bans=bans_continue
        else:
            self.is_done=True
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
