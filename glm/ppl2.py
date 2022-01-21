import os
import sys
import math
import random
import torch


def getbc(seq, mask_position, context_length):
    length = len(seq[0])
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


def ppl(model, input_tokens, target_tokens, mask_position, sop=50006):
    assert len(input_tokens) == len(target_tokens)
    bc = len(input_tokens)
    input_len = 0
    for i in input_tokens:
        if len(i) > input_len:
            input_len = len(i)
    target_len = 0
    for i in target_tokens:
        if len(i) > target_len:
            target_len = len(i)

    for i in range(bc):
        if len(input_tokens[i]) < input_len:
            input_tokens[i] = [-1] * (input_len - len(input_tokens[i])) + input_tokens[i]

    for i in range(bc):
        if len(input_tokens[i]) < target_len:
            target_tokens[i] = target_tokens[i] + [-1] * (target_len - len(target_tokens[i]))

    tokens = []
    for i in range(len(input_tokens)):
        tkl = input_tokens[i] + [sop] + target_tokens[i]
        tokens.append(tkl)
    st = input_len + 1
    context_length = input_len + target_len + 1
    loss_mask = torch.zeros((bc, context_length)).cuda()
    loss_mask[:, st:] = 1
    # print(st,context_length)
    '''
    todo: add pad
    pad_pos = tokens < 0
    if pad_pos.any():
        print('Find -1 in tokens, automatically ignore them.')
        tokens[pad_pos] = 0
        loss_mask[pad_pos] = 0
    '''

    # tokens=torch.LongTensor(tokens).cuda()
    tokens, attention_mask, position_ids = getbc(tokens, mask_position, input_len)

    attention_mask = attention_mask.type_as(next(model.parameters()))
    logits = model(tokens, position_ids, attention_mask)[0]

    logits = logits.float()

    log_probs = torch.log(torch.nn.functional.softmax(logits, dim=-1))

    pred = log_probs[:, :-1, :]
    target = tokens[:, 1:].unsqueeze(-1)
    loss_mask = loss_mask[..., 1:]

    scores = torch.gather(pred, dim=2, index=target).squeeze(-1)  # [batch_size, seq_len-1]
    # print(scores,loss_mask)
    score = (scores * loss_mask).sum(dim=-1)

    return score


def compute_ip(model, input_tokens, target_tokens, tokenizer, pos_bag, attention_mask, position_ids, inverse_factor,
               factor=0.6):
    detailed_scores_ip, score_ip = inverse_ppl(model, input_tokens, target_tokens, tokenizer, pos_bag, factor)
    detailed_scores_p, score_p = correct_ppl(model, input_tokens, target_tokens, attention_mask, position_ids, factor)

    scores = score_p + inverse_factor * score_ip

    detailed_scores = torch.cat(
        (
            detailed_scores_ip,
            detailed_scores_p
        ), dim=1
    )

    return detailed_scores, scores


def inverse_ppl(model, input_tokens, target_tokens, tokenizer, pos_bag, factor=0.6):
    new_input = []
    new_target = []
    # cls = tokenizer.get_command('ENC').Id
    smask = tokenizer.get_command('sMASK').Id
    eos = tokenizer.get_command('eos').Id
    # sop = tokenizer.get_command('sop').Id
    start_pos = pos_bag[0]
    end_pos = pos_bag[1]
    gen_pos = pos_bag[2]
    sop_pos = pos_bag[3]

    for i in range(len(input_tokens)):
        bm = input_tokens[i]
        new = target_tokens[i]
        new_bm = bm[:start_pos - 1] + [smask] + bm[end_pos:gen_pos] + bm[sop_pos:] + new + [eos]
        new_input.append(new_bm)
        new_target.append(bm[start_pos:end_pos])

    return new_ppl(model, new_input, new_target, start_pos, factor)


def new_ppl(model, input_tokens, target_tokens, mask_position, factor=0.6, sop=50006):
    assert len(input_tokens) == len(target_tokens)
    bc = len(input_tokens)
    input_len = 0
    for i in input_tokens:
        if len(i) > input_len:
            input_len = len(i)
    target_len = 0
    for i in target_tokens:
        if len(i) > target_len:
            target_len = len(i)

    for i in range(bc):
        if len(input_tokens[i]) < input_len:
            input_tokens[i] = [-1] * (input_len - len(input_tokens[i])) + input_tokens[i]

    for i in range(bc):
        if len(input_tokens[i]) < target_len:
            target_tokens[i] = target_tokens[i] + [-1] * (target_len - len(target_tokens[i]))

    tokens = []
    for i in range(len(input_tokens)):
        tkl = input_tokens[i] + [sop] + target_tokens[i]
        tokens.append(tkl)
    st = input_len + 1
    context_length = input_len + target_len + 1
    loss_mask = torch.zeros((bc, context_length)).cuda()
    loss_mask[:, st:] = 1
    # print(st,context_length)
    '''
    todo: add pad
    pad_pos = tokens < 0
    if pad_pos.any():
        print('Find -1 in tokens, automatically ignore them.')
        tokens[pad_pos] = 0
        loss_mask[pad_pos] = 0
    '''

    # tokens=torch.LongTensor(tokens).cuda()
    tokens, attention_mask, position_ids = getbc(tokens, mask_position, input_len)

    attention_mask = attention_mask.type_as(next(model.parameters()))

    logits = model(tokens, position_ids, attention_mask)[0]

    logits = logits.float()

    log_probs = torch.log(torch.nn.functional.softmax(logits, dim=-1))

    pred = log_probs[:, :-1, :]

    target = tokens[:, 1:].unsqueeze(-1)
    loss_mask = loss_mask[..., 1:]

    scores = torch.gather(pred, dim=2, index=target).squeeze(-1)  # [batch_size, seq_len-1]

    # TODO: print out probability of each word

    detailed_score = scores * loss_mask

    div = math.pow(len(target_tokens[0]), factor)

    detailed_score = torch.div(detailed_score, div)

    score = detailed_score.sum(dim=-1)

    return detailed_score, score


def correct_ppl(model, input_tokens, target_tokens, attention_mask, position_ids,
                factor=0.6, threshold=-0.05, need_bench=False):
    input_len = len(input_tokens[0])
    target_len = len(target_tokens[0])
    bc = len(input_tokens)

    tokens = []
    for i in range(len(input_tokens)):
        tkl = input_tokens[i] + target_tokens[i]
        tokens.append(tkl)
    st = input_len
    context_length = input_len + target_len
    loss_mask = torch.zeros((bc, context_length)).cuda()
    loss_mask[:, st:] = 1

    tokens = torch.LongTensor(tokens).cuda()

    logits = model(tokens,
                   position_ids[..., : context_length],
                   attention_mask[..., : context_length, :context_length])[0]

    logits = logits.float()

    log_probs = torch.log(torch.nn.functional.softmax(logits, dim=-1))

    pred = log_probs[:, :-1, :]

    target = tokens[:, 1:].unsqueeze(-1)
    loss_mask = loss_mask[..., 1:]

    scores = torch.gather(pred, dim=2, index=target).squeeze(-1)  # [batch_size, seq_len-1]
    
        # TODO: if we want to use bench_ppl
    if need_bench:
        #operation = 2 * threshold - detailed_score

        detailed_score = torch.where(detailed_score > threshold, threshold, detailed_score)

    detailed_score = scores * loss_mask

    div = math.pow(len(target_tokens[0]), factor)

    detailed_score = torch.div(detailed_score, div)

    

    score = detailed_score.sum(dim=-1)

    return detailed_score, score
