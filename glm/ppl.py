import os
import sys
import math
import random
import torch


def getbc(seq, mask_position, context_length):
    length=len(seq[0])
    tokens = torch.LongTensor(seq).cuda()

    attention_mask = torch.ones((1, length, length), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(2, length, device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, length - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids
    
    
def ppl(model, input_tokens, target_tokens,mask_position,sop=50006,additional_attention=None):
    assert len(input_tokens)==len(target_tokens)
    bc=len(input_tokens)
    input_len=0
    for i in input_tokens:
        if len(i)>input_len:
            input_len=len(i)
    target_len=0
    for i in target_tokens:
        if len(i)>target_len:
            target_len=len(i)
    
    for i in range(bc):
        if len(input_tokens[i])<input_len:
            input_tokens[i]=[-1]*(input_len-len(input_tokens[i]))+input_tokens[i]
    
    for i in range(bc):
        if len(target_tokens[i])<target_len:
            target_tokens[i]=target_tokens[i]+[-1]*(target_len-len(target_tokens[i]))
    
    tokens=[]
    for i in range(len(input_tokens)):
        tkl=input_tokens[i]+[sop]+target_tokens[i]
        tokens.append(tkl)
    st=input_len+1
    context_length=input_len+target_len+1
    loss_mask=torch.zeros((bc,context_length)).cuda()
    loss_mask[:,st:]=1
    #print(st,context_length)
    '''
    todo: add pad
    pad_pos = tokens < 0
    if pad_pos.any():
        print('Find -1 in tokens, automatically ignore them.')
        tokens[pad_pos] = 0
        loss_mask[pad_pos] = 0
    '''
    
    #tokens=torch.LongTensor(tokens).cuda()
    tokens,attention_mask,position_ids=getbc(tokens,mask_position,input_len)
    
    attention_mask = attention_mask.type_as(next(model.parameters()))
    
    logits = model(tokens, position_ids, attention_mask,log_attention_weights=additional_attention)[0]
    
    logits = logits.float()
    
    log_probs = torch.log(torch.nn.functional.softmax(logits, dim=-1))

    pred = log_probs[:, :-1, :]
    target = tokens[:, 1:].unsqueeze(-1)
    loss_mask = loss_mask[..., 1:]
    
    scores = torch.gather(pred, dim=2, index=target).squeeze(-1) # [batch_size, seq_len-1]
    #print(scores,loss_mask)
    score=(scores * loss_mask).sum(dim=-1)
    
    return score
   
