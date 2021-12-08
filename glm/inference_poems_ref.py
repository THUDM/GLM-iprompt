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
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')
    
    def poemgen(title,author='李白',emo=None):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        emo_str=''
        
        if emo is not None:
            emo_str=' 情感：'+emo
        
        raw_text=title+' 作者:'+author+' 体裁:诗歌'+emo_str+' 标题:'+title+' 正文:'
     
        generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
        if 'MASK]' not in raw_text:
            raw_text += ' ' + generation_mask
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        seq = [tokenizer.get_command('ENC').Id] + seq
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
                        log_attention_weights=None,
                        get_masks_and_position_ids=get_func,
                        mems=mems,
                        end_tokens=end_tokens
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
                
                
            
            return yayun,rhy,length,decoded_output
    def splitpoem(id,poem):
        pts=0
        num_tks=0
        while num_tks<id:
            if poem[pts] in [',','，','。','.']:
                num_tks+=1
            pts+=1
        spbf=poem[:pts]
        while num_tks<=id:
            if poem[pts] in [',','，','。','.']:
                num_tks+=1
            pts+=1
        spaf=poem[pts:]
        
        return spbf,spaf
        
    def refine_poem(yayun,rhy,length,pretext,poem):
        
        endtk=['.','。']
        for id in range(8):
            strategy._init_cache()
            bf,af=splitpoem(id,poem)
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
            encode_prior=[tokenizer.get_command('ENC').Id] +tokenizer.EncodeAsIds(new_prior).tokenization
            encode_poior=tokenizer.EncodeAsIds(new_poior).tokenization
            seq=encode_prior+encode_poior
            get_func=partial(get_masks_and_position_ids_glm,mask_position=len(encode_prior),context_length=len(seq))
            output_list = []
            input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
            output,mems = generate_sentence(model, input_seq,
                        args.batch_size,tokenizer,
                        strategy=strategy,
                        log_attention_weights=None,
                        get_masks_and_position_ids=get_func,
                        mems=None,
                        end_tokens=end_tokens
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
        
        raw_text=title+' 作者:'+author+' 体裁:诗歌'+emo_str+' 标题: '+title+' 正文: '
        
        
        for i in range(10):
            print("Refinement process ",i+1,":")
            poem=refine_poem(yayun,rhy,length,raw_text,poem)
        return 0
        
    generate_continually(process, args.input_source)

def generate_sentence(
        model,
        seq,
        batch_size,
        tokenizer,
        strategy=BaseStrategy(),
        max_memory_length=100000,
        log_attention_weights=None,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        mems=None,
        end_tokens=[],
        verifier_params=None
        ):
    assert len(seq.shape) == 1

    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1 # [0, context_length-1] are given
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
            if log_attention_weights is not None:
                log_attention_weights_part = log_attention_weights[..., index: counter+1, :counter+1] # TODO memlen
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
            if expansion_size>0:
                logits = logits[:, -1].expand(expansion_size, -1) # [batch size, vocab size]
                tokens = tokens.expand(expansion_size, -1)
                tokens, mems = strategy.forward(logits, tokens, mems)
            if strategy.is_done:
                break
    del mems
    del logits
    torch.cuda.empty_cache()
    return strategy.finalize(tokens, None)
    
    
def eval_title(model,beam,tokenizer):
    decode_tokens = tokenizer.DecodeIds(sentence.cpu().tolist())
    poem=decode_tokens.replace('<|startofpiece|>','').replace('[gMASK]','').replace('[CLS]','')
    
    poem_entry=poem.split(':')[-1]
    author=poem.split(':')[-4].split(' ')[0]
    title=poem.split(' ')[0]
    new_sentence='体裁:诗歌 作者:李白 标题:'
    new_sentence2='[sMASK] 正文:'+poem_entry
    new_sentence3='<|startofpiece|>'+title
    encode_ppl=tokenizer.EncodeAsIds(new_sentence).tokenization
    encode_pp2=tokenizer.EncodeAsIds(new_sentence2).tokenization
    encode_target=tokenizer.EncodeAsIds(new_sentence3).tokenization
    input_st=encode_ppl+encode_ppl2+encode_target
    tokens, attention_mask, position_ids = get_masks_and_position_ids_glm(input_st,len(encode_ppl),len(encode_ppl)+len(encode_ppl2))
    
    
    
    mask=torch.zeros(len(input_st)).cuda()
    input_st=torch.LongTensor(input_st).cuda()
    mask[len(encode_ppl):]=1
    tokens, attention_mask, position_ids = get_masks_and_position_ids_glm(seq)
    ppl=evaluate_perplexity(model,tokens, attention_mask, position_ids, loss_mask,reduction='none')
    return ppl
    
if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy', help='type name of sampling strategy')
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)
