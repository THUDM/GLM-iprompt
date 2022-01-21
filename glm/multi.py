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

from SwissArmyTransformer import mpu, get_args, get_tokenizer, load_checkpoint, initialize_distributed, set_random_seed

from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
from multigen import MultigenStrategy


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

    end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    if args.sampling_strategy == 'BaseStrategy':
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k,end_tokens=end_tokens)
    elif args.sampling_strategy == 'BeamSearchStrategy':
        strategy = BeamSearchStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True, end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=args.min_tgt_length)
    elif args.sampling_strategy == 'MultigenStrategy':
        strategy = MultigenStrategy(args.batch_size, consider_end=True, end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=args.min_tgt_length)
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')
    
    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        # add MASK
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
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = [seq]
        # continually detect the first mark position
    
        seq = output_list[0] # TODO find the best one
            # detect
        mask_tokens = ['MASK', 'sMASK', 'gMASK'] if args.task_mask else ['MASK']
        mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
        mask_position = len(seq)
        for token in mask_tokens:
            try:
                mask_position = min(mask_position, seq.index(token))
            except ValueError:
                pass
        if mask_position == len(seq):
            return 0
            
        get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=len(seq))
        output_list = []
        att_weights=torch.zeros((args.out_seq_length,args.out_seq_length)).cuda()
        ll=mask_position-5
        att_weights[:,:mask_position-2]=5/(5+ll)
        input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
        output = generate_sentence(model, input_seq,
                        batch_size=args.batch_size,
                        strategy=strategy,
                        log_attention_weights=att_weights,
                        get_masks_and_position_ids=get_func
                        ) # we don't use mems, fill back
        print(output)


            # clip -1s and fill back generated things into seq
        output_list=[]
        for i in range(len(output)):
            output = output[i].tolist()
            try:
                unfinished = output.index(-1)
            except ValueError:
                unfinished = len(output)
            if output[unfinished - 1] in end_tokens:
                unfinished -= 1
            bog = output.index(tokenizer.get_command('sop').Id)
            output_list.append(output[bog + 1:unfinished] + output[mask_position + 1:bog])

        # decoding
        txts = []
        for seq in output_list:
            decode_tokens = tokenizer.DecodeIds(seq)
            txts.append(decode_tokens)
        return txts
        
    def read_codes():
        path='HumanEval.jsonl'
        f=open(path,'r')
        lines=f.readlines()
        import json
        all_pr=[]
        all_ids=[]
        for line in lines:
            js=json.loads(line)
            pr=js['prompt'].replace('    ','  ')
            all_pr.append(pr)
            all_ids.append(js['task_id'])
        f.close()
        return all_pr,all_ids
    all_pr,all_ids=read_codes()
    
    



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
