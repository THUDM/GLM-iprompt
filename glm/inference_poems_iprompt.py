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
        strategy = BeamSearchStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True, end_tokens=raw_end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=0,verifier=poem_verifier)
    elif args.sampling_strategy == 'iPromptSearchStrategy':
        strategy = iPromptStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True, end_tokens=raw_end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=0,verifier=poem_verifier,tmp_factor=1.03)
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')
    strategy.set_model(model)
    def poemgen(title,author='李白',emo=None):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        emo_str=''
        if ' ' in title:
            tt=title.split()[0]
            author=title.split()[1]
            title=tt
        if emo is not None:
            emo_str=' 情感：'+emo
        
        raw_text=title+' '
        len0=len(tokenizer.EncodeAsIds(raw_text).tokenization)+1
        strategy.set_ini_pos(len0)
        raw_text=raw_text+'作者:'+author+' 体裁:'
        len00=len(tokenizer.EncodeAsIds(raw_text).tokenization)+1
        
        raw_text=raw_text+'诗歌'+emo_str+' 题名:'
        len1=len(tokenizer.EncodeAsIds(raw_text).tokenization)+1
        strategy.set_start(len1)
        raw_text=raw_text+title
        len2=len(tokenizer.EncodeAsIds(raw_text).tokenization)+1
        strategy.set_end(len2)
        raw_text=raw_text+' 原文:[gMASK]'
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        seq = [tokenizer.get_command('ENC').Id] + seq
        strategy.set_gen_pos(len(seq)-1)
        strategy.set_sop_pos(len(seq))
    
        print('raw text: {}\n'.format(raw_text))
        if len(seq) > args.max_sequence_length:
            raise ValueError('text too long.')

        # generation
        output_list = [seq]
        # continually detect the first mark position
        while True:
            mask_position = len(seq)-1
            
            
            get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=len(seq))
            ll=len(seq)
            output_list = []
            input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
                
            
            end_tokens=raw_end_tokens+[tokenizer.TokenToId('。'),tokenizer.TokenToId('.')]
            endtk=['.','。']
            vparam=[tokenizer,dict,shengdict,2,2,5,7,endtk,'']
            strategy.set_verifier_params(vparam)
            mems=None
            yayun=''
            rhy=2
            for id in range(8):
                if '。' in endtk:
                    endtk=[',','，']
                    end_tokens=[tokenizer.TokenToId(',')]
                else:
                    end_tokens=[tokenizer.TokenToId('。'),tokenizer.TokenToId('.')]
                    endtk=['.','。']
                strategy.set_end_tokens(end_tokens)
                strategy.set_verifier_param(endtk,7)
                
                if id%2==0:
                    strategy.set_verifier_param('',8)
                else:
                    strategy.set_verifier_param(yayun,8)
                if id>0:
                    strategy.set_verifier_param(1-id%2,4)
                if rhy!=2:
                    if id%2==1:
                        strategy.set_verifier_param(1-rhy,3)
                #print(input_seq)
                output,mems = generate_sentence(model, input_seq,
                        args.batch_size,tokenizer,
                        strategy=strategy,
                        pos=[len0,len1,len2],
                        get_masks_and_position_ids=get_func,
                        mems=mems,
                        end_tokens=end_tokens,
                        use_ip=1,
                        weight=0.78
                        )
                #print(output[0])
                decoded_output=tokenizer.DecodeIds(output[0].tolist())
                if id==7:
                    print(decoded_output.split('>')[-1])
                yayun,rhy,length=verify_rhy(decoded_output,id,shengdict,yayun,rhy)
                if id==0:
                    strategy.set_verifier_param(length,5)
                    strategy.set_verifier_param(length,6)
                strategy._init_cache()
              
                input_seq[:len(output[0])]=output[0]
                mems=None
                
                
            decoded_output=decoded_output.split('>')[-1]
            return yayun,rhy,length,decoded_output
    def splitpoem(id,poem):
        pts=0
        num_tks=0
        while num_tks<id:
            if poem[pts] in [',','，','。','.']:
                num_tks+=1
            pts+=1
        spbf=poem[:pts]
        pp=pts
        while num_tks<=id:
            if poem[pts] in [',','，','。','.']:
                num_tks+=1
            pts+=1
        spaf=poem[pts:]
        sp=poem[pp:pts]
        return spbf,spaf,sp
    
    def refine_poem(yayun,rhy,length,pretext,poem,wt=0):
        
        endtk=['.','。']
        for id in range(8):
            strategy._init_cache()
            bf,af,sp=splitpoem(id,poem)
            if '。' in endtk:
                endtk=[',','，']
                end_tokens=[tokenizer.TokenToId(',')]
            else:
                end_tokens=[tokenizer.TokenToId('。'),tokenizer.TokenToId('.')]
                endtk=['.','。']
            strategy.set_end_tokens(end_tokens)
            strategy.set_verifier_param(endtk,7)
            if id>0:
                strategy.set_verifier_param(1-id%2,4)
            else:
                strategy.set_verifier_param(2,4)
            if id%2==1:
                strategy.set_verifier_param(1-rhy,3)
                rhy=1-rhy
            
            new_prior=pretext+bf
            
            new_poior='[sMASK]'+af
            if id==7:
                new_poior='[gMASK]'
            encode_prior=[tokenizer.get_command('ENC').Id] +tokenizer.EncodeAsIds(new_prior).tokenization
            
            encode_poior=tokenizer.EncodeAsIds(new_poior).tokenization
            if id!=7:
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
                        end_tokens=end_tokens,
                        use_ip=0,
                        weight=0.9,
                        excess_beam=[sp]
                        )
            decoded_output=tokenizer.DecodeIds(output[0].tolist())
            #print(decoded_output)
            poem=bf+decoded_output.split('>')[-1]+af
            #print(poem)
        print(poem)
        return poem
    def process(title,author='李白',emo=None):
        yayun,rhy,length,poem=poemgen(title,author,emo)
        emo_str=''
        if emo is not None:
            emo_str=' 情感：'+emo
        if ' ' in title:
            tt=title.split()[0]
            author=title.split()[1]
            title=tt
        raw_text=title+' 作者:'+author+' 体裁:诗歌'+emo_str+' 题名:'+title+' 原文:'
        
        prev_poem=poem
        for i in range(15):
            print("Refinement process ",i+1,":")
            poem=refine_poem(yayun,rhy,length,raw_text,poem,wt=i)
            if poem==prev_poem:
                return 0
            prev_poem=poem
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
        use_ip=0,
        excess_beam=None
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
    ban_end=False
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
            logits, *mem_kv = model(
                tokens[:, index:],
                position_ids[..., index: counter+1],
                attention_mask[..., index: counter+1, :counter+1], # TODO memlen
                mems=mems,
                log_attention_weights=log_attention_weights_part
            )
            mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
            counter += 1
            index = counter
            expansion_size=len(logits)
            
            # sampling
            if use_ip==1:
                use=True
            else:
                use=False
            if expansion_size>0:
                logits = logits[:, -1].expand(expansion_size, -1) # [batch size, vocab size]
                tokens = tokens.expand(expansion_size, -1)
                tokens, mems = strategy.forward(logits, tokens, mems,use_ip=use,ban_end=ban_end)
                
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
            strategy._add_end_beams(0,0,new_beam)
    return strategy.finalize(tokens, None)
    
    

    
if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy', help='type name of sampling strategy')
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)
