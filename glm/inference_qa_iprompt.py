# -*- encoding: utf-8 -*-
'''
@File    :   inference_glm.py
@Time    :   2021/10/22 19:41:58
@Author  :   Ming Ding
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
import os
import sys
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import argparse
import stat
from functools import partial
from poem_verifier import *

from SwissArmyTransformer import mpu, get_args, get_tokenizer, load_checkpoint, initialize_distributed, set_random_seed
from SwissArmyTransformer.generation.autoregressive_sampling import get_masks_and_position_ids_default, update_mems
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from beam_search_strategy import BeamSearchStrategy
from iprompt import iPromptStrategy
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
from pynvml import *


def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids


def main(args):
    args.do_train = False
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)
    # build model 
    model = GLMModel(args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    load_checkpoint(model, args)
    set_random_seed(args.seed)
    model.eval()
    dict,shengdict,allbu,allsb=cilin()
    
    raw_end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    if args.sampling_strategy == 'BaseStrategy':
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k,end_tokens=end_tokens)
    elif args.sampling_strategy == 'BeamSearchStrategy':
        strategy = BeamSearchStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True, end_tokens=raw_end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=0)
    elif args.sampling_strategy == 'iPromptStrategy':
        strategy = iPromptStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True, end_tokens=raw_end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=0)
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')
    strategy.set_model(model)
    def qagen(title,desc=None,mode='qa',author='令狐冲'):
        strategy._init_cache()
        desc_str=''
        if ' ' in title:
            tt=title.split()[0]
            author=title.split()[1]
            title=tt
        if desc is not None:
            desc_str=' 问题描述:'+desc
        
        strategy.set_end_tokens(raw_end_tokens)
        
        strategy.set_ini_pos(1)
        raw_text='问题:'
        len1=len(tokenizer.EncodeAsIds(raw_text).tokenization)+1
        strategy.set_start(len1)
        raw_text=raw_text+title
        len2=len(tokenizer.EncodeAsIds(raw_text).tokenization)+1
        strategy.set_end(len2)
        raw_text=raw_text+desc_str+' 回答用户:'+author+' 回答:'
        pretext=raw_text
     
        generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
        if 'MASK]' not in raw_text:
            raw_text += ' ' + generation_mask
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        seq = [tokenizer.get_command('ENC').Id] + seq
        strategy.set_gen_pos(len(seq)-1)
        strategy.set_sop_pos(len(seq))
        if not raw_text.endswith('MASK]'):
            seq = seq + [tokenizer.get_command('eos').Id]
        print('raw text: {}\n'.format(raw_text))
        if len(seq) > args.max_sequence_length:
            raise ValueError('text too long.')

        # generation
        output_list = [seq]
        # continually detect the first mark position
        while True:
            # TODO find the best one
            # detect
            mask_tokens = ['MASK', 'sMASK', 'gMASK']
            mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
            mask_position = len(seq)-1
            
            
            get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=len(seq))
            ll=len(seq)
            output_list = []
            input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
                
            
            end_tokens=raw_end_tokens
            
            vparam=[tokenizer]
            strategy.set_verifier_params(vparam)
            
            
            pos=strategy.get_pos()
            output,mems = generate_sentence(model, input_seq,
                        args.batch_size,tokenizer,
                        strategy=strategy,
                        pos=strategy.get_pos(),
                        get_masks_and_position_ids=get_func,
                        mems=None,
                        weight=0.65,
                        end_tokens=end_tokens,
                        
                        )
            #print(output)
            decoded_output=tokenizer.DecodeIds(output[0].tolist())
            if '<|end' in decoded_output:
                decoded_output=decoded_output.split('<|end')[0]
            pre_text=decoded_output.split('|>')[0]
            decoded_output=decoded_output[len(pre_text)+2:]
            decoded_output=decoded_output.replace('<n>','\n').replace('%25','%')
            print(decoded_output)
            return decoded_output
    
    def getfirstsp(answer,pointer):
        
        endnote=['\n','。','!','！','?','？','"',"'",'>',')',';','》','“','‘','...']
        nowpt=pointer+9
        
        caster=True
        add_charter=0
        while caster:
            nowpt+=1
            while (nowpt<len(answer)) and(answer[nowpt] in endnote):
                nowpt+=1
                add_charter+=1
                caster=False
            if nowpt>=len(answer):
                caster=False
        nowpt-=1
        if add_charter>1 and answer[nowpt] in ['"',"'",'“','‘']:
            nowpt-=1
            add_charter-=1
            
        if nowpt>=len(answer):
            nowpt=len(answer)-1
       #nowpt means the end of current sentence
        return nowpt
    def add_potential_ends():
        endtk=['\n','。','!','！','?','？','"',"'",'>',')',';','》','“','‘','...']
        end_tokens=raw_end_tokens[:]
        for i in range(50000):
            tk=tokenizer.DecodeIds([i])
            for w in endtk:
                if w in tk:
                    end_tokens.append(i)
                   # print(tk)
        #print(end_tokens)
        strategy.set_end_tokens(end_tokens)
    
        
    def refine_answer(pretext,answer):
        
        pointer=0
        isend=False
        while pointer<len(answer):
            strategy._init_cache()
            nowpt=getfirstsp(answer,pointer)+1
            
            if nowpt>=len(answer):
                isend=True
                
            new_prior=pretext+answer[:pointer]
            if not(isend):
                new_poior='[sMASK]'+answer[nowpt:]
            else:
                new_poior='[gMASK]'
                strategy.set_end_tokens(raw_end_tokens)
                
            
            encode_prior=[tokenizer.get_command('ENC').Id] +tokenizer.EncodeAsIds(new_prior).tokenization
            encode_poior=tokenizer.EncodeAsIds(new_poior).tokenization
            if not(isend):
                encode_poior=encode_poior+[tokenizer.get_command('eos').Id]
            seq=encode_prior+encode_poior
            strategy.set_gen_pos(len(encode_prior))
            strategy.set_sop_pos(len(seq))
            get_func=partial(get_masks_and_position_ids_glm,mask_position=len(encode_prior),context_length=len(seq))
            output_list = []
            input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
            pos=strategy.get_pos()
            
            
            output,mems = generate_sentence(model, input_seq,
                        args.batch_size,tokenizer,
                        strategy=strategy,
                        pos=pos,
                        get_masks_and_position_ids=get_func,
                        mems=None,
                        weight=0.75,
                        excess_beam=[answer[pointer:nowpt]],
                        use_ip_sep=0
                        )
                        
            
            decoded_output=tokenizer.DecodeIds(output[0].tolist()).replace('<n>','\n').replace('%25','%')
            #print(decoded_output)
            new_answer=answer[:pointer]+decoded_output.split('startofpiece|>')[-1]+answer[nowpt:]
            pointer=nowpt
            #print(poem)
        print(new_answer)
        return new_answer
    def process(title,author='令狐冲',desc=None):
        answer=qagen(title)
        desc_str=''
        if desc is not None:
            desc_str=' 问题描述:'+desc
        if ' ' in title:
            tt=title.split()[0]
            author=title.split()[1]
            title=tt
        raw_text='问题:'+title+desc_str+' 回答用户:'+author+' 回答:'
        
        
        prev_answer=answer
        '''
        for i in range(10):
            print("Refinement process ",i+1,":")
            add_potential_ends()
            answer=refine_answer(raw_text,answer)
            if answer==prev_answer:
                return 0
            prev_answer=answer
        '''
        return 0
        
    generate_continually(process, args.input_source)

def generate_sentence(
        model,
        seq,
        batch_size,
        tokenizer,
        strategy=BaseStrategy(),
        max_memory_length=100000,
        pos=None,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        mems=None,
        end_tokens=[],
        verifier_params=None,
        weight=1.25,
        excess_beam=None,
        use_ip_sep=5
        ):
    assert len(seq.shape) == 1

    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1 # [0, context_length-1] are given
    cl=context_length
    assert context_length > 0
    tokens, attention_mask, position_ids = get_masks_and_position_ids(seq)
    if len(end_tokens)>0:
        strategy.set_end_tokens(end_tokens)
    
    tokens = tokens[..., :context_length]
    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    # step-by-step generation
    num_generated=0
    with torch.no_grad():
        while counter < len(seq) - 1:
            
            # Now, we want to generate seq[counter + 1],
            # token[:, index: counter+1] needs forwarding.

            if seq[counter + 1] >= 0: # provided
                tokens = torch.cat(
                    (
                    tokens,
                        seq[counter+1: counter+2].expand(tokens.shape[0], 1)
                    ), dim=1
                )
                counter += 1
                continue

            # forward
            if pos is not None:
                ini_pos=pos[0]
                log_attention_weights_part=torch.zeros(counter+1).cuda()
                st_pos=pos[1]
                end_pos=pos[2]
                
                log_attention_weights_part[1:ini_pos] = weight
                log_attention_weights_part[st_pos:end_pos] = weight
            else:
                log_attention_weights_part = None
            
            #for piece in tokens:
            #    w=piece.cpu().tolist()
            #    decode_tokens = tokenizer.DecodeIds(w)
            #    print(w,decode_tokens)
            num_generated+=1
            use_ip=False
            max_length=375
            if use_ip_sep!=0:
                if num_generated%use_ip_sep==0:
                    use_ip=True
            else:
                max_length=450
                
                
            logits, *mem_kv = model(
                tokens[:, index:],
                position_ids[..., index: counter+1],
                attention_mask[..., index: counter+1, :counter+1], # TODO memlen
                mems=mems,
                log_attention_weights=log_attention_weights_part,
                
            )
            mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
            counter += 1
            index = counter
            expansion_size=len(logits)
            
            # sampling
            ban_end=False
            if num_generated<5:
                ban_end=True
            
            if expansion_size>0:
                logits = logits[:, -1].expand(expansion_size, -1) # [batch size, vocab size]
                tokens = tokens.expand(expansion_size, -1)
                tokens, mems = strategy.forward(logits, tokens, mems,use_ip=use_ip,ban_end=ban_end,max_length=max_length)
            #print(counter,strategy.is_done,strategy.end_tokens)
            if strategy.is_done:
                break
    del mems
    del logits
    torch.cuda.empty_cache()
    if excess_beam is not None:
        for st in excess_beam:
            encodedst=tokenizer.EncodeAsIds(st).tokenization
            #print(st)
            new_beam=torch.cat((seq[:cl],torch.LongTensor(encodedst).cuda()),dim=0)
            #print(new_beam)
            strategy._add_end_beams(0,-0.2,new_beam)
            
    return strategy.finalize(tokens, None)
    
    

    
if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy', help='type name of sampling strategy')
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)
