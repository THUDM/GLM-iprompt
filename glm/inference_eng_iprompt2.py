# -*- encoding: utf-8 -*-
'''
@File    :   inference_glm.py
@Time    :   2021/10/22 19:41:58
@Author  :   Ming Ding
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
import pandas as pd
import os
import sys
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import argparse
import stat
import json
import requests
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import urllib.request as req
import random
from math import exp
import nltk
from math import pow, inf
from ppl2 import ppl, compute_ip, correct_ppl
import time

nltk.download('punkt')
import re
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
from poem_verifier import eng_verifier


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def get_search_result(title, urllist, search_sentence_dict, tokenizer):
    my_headers = [
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
        "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
        'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
        'Opera/9.25 (Windows NT 5.1; U; en)',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
        'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
        'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
        'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
        "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
        "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
    ]

    subscription_key = "3dee504568624d3696ab72baafd1fe62"
    assert subscription_key

    search_url = "https://api.bing.microsoft.com/v7.0/search"

    search_term = title

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    for v in search_results["webPages"]["value"]:

        url = v["url"]
        # print(url)
        if url in urllist:
            # print("Duplicated url")
            pass

        urllist.append(url)

        headers = {"User-Agent": random.choice(my_headers)}
        try:
            response = requests.get(url, headers=headers, timeout=2.0)
        except:
            # print("Does not get result")
            continue
        htmlParse = BeautifulSoup(response.text, 'html.parser')

        for para in htmlParse.find_all("p"):
            paragraph = para.get_text().rstrip('\n')

            for sentence in sent_tokenize(paragraph):
                sentence = preprocess_sentence(sentence)
                if sentence:
                    tokenized = tokenizer.EncodeAsIds(sentence).tokenization
                    length = len(tokenized)
                    if length in search_sentence_dict.keys():
                        search_sentence_dict[length].append(tokenized)
                    else:
                        search_sentence_dict[length] = [tokenized]


def preprocess_sentence(sentence):
    sentence = sentence.strip().replace("\n", " ").replace("\r", " ")
    # If the sentence contains a link, we do not consider it
    if "https://www." in sentence:
        return None
    # If the sentence contains no alphabet, we do not consider it
    if re.search('[a-zA-Z]', sentence) == None:
        return None

    return sentence


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
    dict, shengdict, allbu, allsb = cilin()

    raw_end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    if args.sampling_strategy == 'BaseStrategy':
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    elif args.sampling_strategy == 'BeamSearchStrategy':
        strategy = BeamSearchStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True,
                                      end_tokens=raw_end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                      min_tgt_length=0)
    elif args.sampling_strategy == 'iPromptStrategy':
        strategy = iPromptStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True,
                                   end_tokens=raw_end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                   min_tgt_length=0, mode='english', verifier=eng_verifier)
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')
    strategy.set_model(model)

    def qagen(title, factor, re_search, inverse_factor, threshold, need_bench, new_method, desc=None):

        base_path = ""
        if new_method:
            if inverse_factor != 0:
                base_path = './temp/{}/inverse={}.norm={}/'.format(title, inverse_factor, factor)
            elif need_bench:
                base_path = './temp/{}/thres={}.norm={}/'.format(title, threshold, factor)
            else:
                base_path = './temp/{}/norm={}/'.format(title, factor)

            ensureDir(base_path)

            urllist = []
            search_sentence_dict = {}

            get_search_result(title, urllist, search_sentence_dict, tokenizer)

            # Sort the list and see how many groups we need to calculate
            print("How many sentence group we get", len(search_sentence_dict))

        strategy._init_cache()
        desc_str = ''

        if desc is not None:
            desc_str = ' Description:' + desc

        strategy.set_end_tokens(raw_end_tokens)

        strategy.set_ini_pos(1)
        raw_text = 'Question: '
        len1 = len(tokenizer.EncodeAsIds(raw_text).tokenization) + 1
        strategy.set_start(len1)
        raw_text = raw_text + title
        len2 = len(tokenizer.EncodeAsIds(raw_text).tokenization) + 1
        strategy.set_end(len2)
        raw_text = raw_text + desc_str + ' Answer:'
        pretext = raw_text

        # TODO do not add [gMask] here
        generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
        if 'MASK]' not in raw_text:
            raw_text += ' ' + generation_mask
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        seq = [tokenizer.get_command('ENC').Id] + seq
        strategy.set_gen_pos(len(seq) - 1)
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
            mask_position = len(seq) - 1

            # print("Check mask_position", mask_position, tokenizer.EncodeAsIds(generation_mask))
            # print(tokenizer.DecodeIds(seq), seq)

            # input()

            get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=len(seq))
            ll = len(seq)
            output_list = []
            input_seq = torch.cuda.LongTensor(
                seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                device=args.device)

            end_tokens = raw_end_tokens

            vparam = [tokenizer]
            strategy.set_verifier_params(vparam)

            if new_method:

                pos_bag = strategy.get_posbag()

                start = time.time()

                _, attention_mask, position_ids = get_func(input_seq)

                attention_mask = attention_mask.type_as(next(model.parameters()))

                output = generate_from_source_new(attention_mask, position_ids,
                                                  model, seq, search_sentence_dict, args.out_seq_length, args.device,
                                                  tokenizer, factor, urllist, re_search, pos_bag, inverse_factor,
                                                  threshold, base_path, need_bench)

                end = time.time()

                print("Check elapsed time", end - start, output.shape)

                return output

            else:
                output, mems = generate_sentence(model, input_seq,
                                                 args.batch_size, tokenizer,
                                                 strategy=strategy,
                                                 pos=strategy.get_pos(),
                                                 get_masks_and_position_ids=get_func,
                                                 mems=None,
                                                 weight=0.65,
                                                 end_tokens=end_tokens,
                                                 )
                # print(output)
                decoded_output = tokenizer.DecodeIds(output[0].tolist())

                if '<|end' in decoded_output:
                    decoded_output = decoded_output.split('<|end')[0]
                pre_text = decoded_output.split('|>')[0]
                decoded_output = decoded_output[len(pre_text) + 2:]
                decoded_output = decoded_output.replace('<n>', '\n').replace('%25', '%')
                print(decoded_output)
                return decoded_output

    def getfirstsp(answer, pointer):

        endnote = ['\n', '.', '!', '?', '"', "'"]
        nowpt = pointer + 1

        caster = True
        add_charter = 0
        all_space = 0
        while caster:
            nowpt += 1
            if (nowpt < len(answer)) and (answer[nowpt] == ' '):
                all_space += 1
            while (nowpt < len(answer)) and (answer[nowpt] in endnote):
                nowpt += 1
                if (nowpt < len(answer)) and (answer[nowpt] == ' '):
                    all_space += 1
                add_charter += 1
                if all_space >= 4:
                    caster = False
            if nowpt >= len(answer):
                caster = False
        nowpt -= 1
        if add_charter > 1 and answer[nowpt] in ['"', "'", '“', '‘']:
            nowpt -= 1
            add_charter -= 1

        if nowpt >= len(answer):
            nowpt = len(answer) - 1
        # nowpt means the end of current sentence
        return nowpt

    def add_potential_ends():
        endtk = ['\n', '.', '?', '!', '"', "'"]
        end_tokens = raw_end_tokens[:]
        for i in range(50256):
            tk = tokenizer.DecodeIds([i])
            for w in endtk:
                if w in tk:
                    end_tokens.append(i)
                # print(tk)
        # print(end_tokens)
        strategy.set_end_tokens(end_tokens)

    def refine_answer(pretext, answer):

        pointer = 0
        isend = False
        while pointer < len(answer):
            strategy._init_cache()
            nowpt = getfirstsp(answer, pointer) + 1

            if nowpt >= len(answer):
                isend = True

            new_prior = pretext + answer[:pointer]
            if not (isend):
                new_poior = '[sMASK]' + answer[nowpt:]
            else:
                new_poior = '[gMASK]'
                strategy.set_end_tokens(raw_end_tokens)

            encode_prior = [tokenizer.get_command('ENC').Id] + tokenizer.EncodeAsIds(new_prior).tokenization
            encode_poior = tokenizer.EncodeAsIds(new_poior).tokenization
            if not (isend):
                encode_poior = encode_poior + [tokenizer.get_command('eos').Id]
            seq = encode_prior + encode_poior
            strategy.set_gen_pos(len(encode_prior))
            strategy.set_sop_pos(len(seq))
            get_func = partial(get_masks_and_position_ids_glm, mask_position=len(encode_prior), context_length=len(seq))
            output_list = []
            input_seq = torch.cuda.LongTensor(
                seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                device=args.device)
            pos = strategy.get_pos()

            output, mems = generate_sentence(model, input_seq,
                                             args.batch_size, tokenizer,
                                             strategy=strategy,
                                             pos=pos,
                                             get_masks_and_position_ids=get_func,
                                             mems=None,
                                             weight=0.75,
                                             excess_beam=[answer[pointer:nowpt]],
                                             use_ip_sep=0
                                             )

            decoded_output = tokenizer.DecodeIds(output[0].tolist()).replace('<n>', '\n').replace('%25', '%')
            # print(decoded_output)
            new_answer = answer[:pointer] + decoded_output.split('startofpiece|>')[-1] + answer[nowpt:]
            pointer = nowpt
            # print(poem)
        print(new_answer)
        return new_answer

    def process(title, desc=None):

        # TODO: take input argument normalization factor
        data = title.split("|")
        print(data)
        # $Question?|# re-research|normalization factor|reverse flag|threshold
        title = data[0]
        re_search = int(data[1])
        factor = float(data[2])
        inverse_factor = float(data[3])
        threshold = float(data[4])

        new_method = True
        need_bench = True

        if threshold == 0:
            need_bench = False

        if factor == 0:
            new_method = False

        answer = qagen(title, factor, re_search, inverse_factor, threshold, need_bench, new_method)

        desc_str = ''
        if desc is not None:
            desc_str = ' Description:' + desc

        raw_text = 'Question: ' + title + desc_str + ' Answer:'

        prev_answer = answer
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


# TODO: jingnan
def generate_from_source_new(attention_mask, position_ids, model, seq, search_sentence_dict, out_seq_length, device,
                             tokenizer, factor, urllist,
                             re_search, pos_bag, inverse_factor, threshold, base_path, need_bench):
    context_length = len(seq)

    mask_position = len(seq) - 1

    counter = context_length - 1  # Last fixed index is ``counter''

    num_generated = 0

    with torch.no_grad():
        while counter < out_seq_length - 1:

            best_match, best_k = choose_sentences_new(attention_mask, position_ids,
                                                      model, seq, search_sentence_dict, tokenizer, factor,
                                                      pos_bag, inverse_factor, out_seq_length,
                                                      threshold, base_path, need_bench)
            # The process has ended
            if best_k == -1:
                return tokens

            decoded_output = tokenizer.DecodeIds(best_match)
            print(decoded_output, end=" ")

            if (len(best_match) + counter) > out_seq_length:
                break

            # TODO: search more results for the first n generated sentences, new results are updated into
            #  search_sentence_dict

            if num_generated < re_search:
                get_search_result(decoded_output, urllist, search_sentence_dict, tokenizer)
                search_sentence_dict[best_k] = list(filter(best_match.__ne__, search_sentence_dict[best_k]))
                # print("Additional search", len(search_sentence_dict))

            counter += len(best_match)
            tokens = torch.LongTensor(seq).unsqueeze(0).to(device)

            best_match = torch.LongTensor(best_match).unsqueeze(0).cuda(tokens.device)
            tokens = torch.cat((tokens, best_match.to(tokens.device)), dim=1)
            seq = torch.squeeze(tokens).tolist()

            num_generated += 1

    return tokens


def choose_sentences_new(attention_mask, position_ids,
                         model, seq, search_sentence_dict, tokenizer, factor,
                         pos_bag, inverse_factor, out_seq_length,
                         threshold, base_path, need_bench
                         ):
    # Meta data
    best_match = []
    best_ppl = -inf
    best_k = 0

    # Need to print out
    total_scores = []
    total_tokens = []
    total_detailed_scores = []

    for k in search_sentence_dict.keys():
        target_tokens = search_sentence_dict[k]
        if len(target_tokens) == 0:
            continue

        input_tokens = [seq] * len(target_tokens)

        # If adding this sentence means exceeding the maximum size
        if len(target_tokens[0]) + len(seq) > out_seq_length:
            continue

        # TODO: inverse prompting
        if inverse_factor != 0:
            detailed_scores, scores = compute_ip(model, input_tokens, target_tokens, tokenizer,
                                                                       pos_bag, attention_mask, position_ids,
                                                                       inverse_factor,factor)
        else:
            detailed_scores, scores = correct_ppl(model, input_tokens, target_tokens,
                                                  attention_mask, position_ids, factor,
                                                  threshold, need_bench)

        total_scores += scores.tolist()
        total_tokens += target_tokens
        # TODO: 输出detailed score
        total_detailed_scores += detailed_scores.tolist()

        index = torch.argmax(scores).item()
        high_ppl = torch.max(scores).item()

        if high_ppl > best_ppl:
            best_ppl = high_ppl
            best_match = target_tokens[index]
            best_k = k

    # If we does not append anything in this step
    if len(total_scores) == 0:
        return [], -1

    zipped_lists = zip(total_scores, total_tokens, total_detailed_scores)
    sorted_zipped_lists = sorted(zipped_lists, reverse=True)
    sorted_scores = [element for element, _, _ in sorted_zipped_lists]
    sorted_tokens = [element for _, element, _ in sorted_zipped_lists]
    sorted_detailed_scores = [element for _, _, element in sorted_zipped_lists]

    readable_text = [tokenizer.DecodeIds(element) for element in sorted_tokens]

    # TODO: Score for each token
    detailed_tokens = []
    for each_line in sorted_tokens:
        line = []
        for each_token in each_line:
            line.append(tokenizer.DecodeIds([each_token]))

        detailed_tokens.append(line)

    d = {'scores': sorted_scores, 'readable_text': readable_text, 'detailed_scores': sorted_detailed_scores,
         'detailed_tokens': detailed_tokens}
    df = pd.DataFrame(data=d)
    df.to_csv("{}/{}.csv".format(base_path, len(seq)))

    # 如果一句话被选中之后就从dictionary里面删掉
    search_sentence_dict[best_k] = list(filter(best_match.__ne__, search_sentence_dict[best_k]))

    return best_match, best_k


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
        use_ip_sep=4
):
    assert len(seq.shape) == 1

    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1  # [0, context_length-1] are given
    cl = context_length
    assert context_length > 0

    tokens, attention_mask, position_ids = get_masks_and_position_ids(seq)

    if len(end_tokens) > 0:
        strategy.set_end_tokens(end_tokens)

    # TODO:Check with tokens

    tokens = tokens[..., :context_length]
    attention_mask = attention_mask.type_as(next(model.parameters()))  # if fp16
    # initialize generation
    counter = context_length - 1  # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2]  # Next forward starting index, also the length of cache.
    # step-by-step generation
    num_generated = 0
    with torch.no_grad():
        while counter < len(seq) - 1:

            # Now, we want to generate seq[counter + 1],
            # token[:, index: counter+1] needs forwarding.

            if seq[counter + 1] >= 0:  # provided
                # TODO: check if we have come to this step
                tokens = torch.cat(
                    (
                        tokens,
                        seq[counter + 1: counter + 2].expand(tokens.shape[0], 1)
                    ), dim=1
                )
                counter += 1
                continue

            # forward
            if pos is not None:
                ini_pos = pos[0]
                log_attention_weights_part = torch.zeros(counter + 1).cuda()
                st_pos = pos[1]
                end_pos = pos[2]

                log_attention_weights_part[1:ini_pos] = weight
                log_attention_weights_part[st_pos:end_pos] = weight
            else:
                log_attention_weights_part = None

            # for piece in tokens:
            #    w=piece.cpu().tolist()
            #    decode_tokens = tokenizer.DecodeIds(w)
            #    print(w,decode_tokens)
            num_generated += 1
            use_ip = False
            max_length = 375
            if use_ip_sep != 0:
                if num_generated % use_ip_sep == 0:
                    use_ip = True
            else:
                max_length = 450

            # TODO: check inputs to the transformer model

            logits, *mem_kv = model(
                tokens[:, index:],
                position_ids[..., index: counter + 1],
                attention_mask[..., index: counter + 1, :counter + 1],  # TODO memlen
                mems=mems,
                log_attention_weights=log_attention_weights_part,
            )
            mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
            counter += 1
            index = counter
            expansion_size = len(logits)

            # sampling
            ban_end = False
            if num_generated < 4:
                ban_end = True

            if expansion_size > 0:
                logits = logits[:, -1].expand(expansion_size, -1)  # [batch size, vocab size]
                tokens = tokens.expand(expansion_size, -1)

                tokens, mems = strategy.forward(logits, tokens, mems, use_ip=use_ip, ban_end=ban_end,
                                                max_length=max_length)
            # print(counter,strategy.is_done,strategy.end_tokens)
            if strategy.is_done:
                break
    del mems
    del logits
    torch.cuda.empty_cache()
    if excess_beam is not None:
        for st in excess_beam:
            encodedst = tokenizer.EncodeAsIds(st).tokenization
            # print(st)
            new_beam = torch.cat((seq[:cl], torch.LongTensor(encodedst).cuda()), dim=0)
            # print(new_beam)
            strategy._add_end_beams(0, -0.2, new_beam)

    return strategy.finalize(tokens, None)


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy',
                           help='type name of sampling strategy')
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    with torch.no_grad():
        main(args)

'''
def generate_from_source(model, seq, max_memory_length=100000, pos=None,
                         mems=None, weight=1.25, search_sentence_list=[], sentence_list=[]):
    assert len(seq.shape) == 1

    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1  # [0, context_length-1] are given

    assert context_length > 0
    # tokens, attention_mask, position_ids = get_masks_and_position_ids(seq)

    counter = context_length - 1  # Last fixed index is ``counter''

    max_ID_list = []

    with torch.no_grad():
        while counter < len(seq) - 1:
            print(counter + 1)
            best_match, max_ID = choose_one_sentence(counter + 1, model, seq,
                                                     max_memory_length, pos, mems, weight, search_sentence_list,
                                                     max_ID_list)

            max_ID_list.append(max_ID)
            print(sentence_list[max_ID])

            if (len(best_match) + counter) > len(seq):
                break

            counter += len(best_match)
            tokens = seq.unsqueeze(0)
            best_match = torch.LongTensor(best_match).unsqueeze(0).cuda(tokens.device)
            tokens = torch.cat((tokens, best_match.to(tokens.device)), dim=1)
            seq = torch.squeeze(tokens)

    return tokens


def choose_one_sentence(context_length, model, seq,
                        max_memory_length=100000, pos=None,
                        mems=None, weight=1.25, sentence_list=[], max_ID_list=[]):
    max_ppl = -inf
    max_ID = 0
    i = 0

    tokens, attention_mask, position_ids = get_masks_and_position_ids_glm(seq, context_length - 1, context_length)

    for given_sentence in sentence_list:

        if i in max_ID_list:
            i += 1
            print("If we have come to here", i)
            continue

        temp_ppl = calculate_sentence_ppl(context_length, model, given_sentence, max_memory_length, pos,
                                          tokens, attention_mask, position_ids, mems, weight)

        print("Check what is the problem", temp_ppl, i)

        if temp_ppl > max_ppl:
            max_ppl = temp_ppl
            max_ID = i

        i += 1

    best_match = sentence_list[max_ID]

    return best_match, max_ID


def calculate_sentence_ppl(context_length, model, given_sentence, max_memory_length, pos=None,
                           tokens=None, attention_mask=None, position_ids=None, mems=None, weight=1.25):
    tokens = tokens[..., :context_length]
    attention_mask = attention_mask.type_as(next(model.parameters()))  # if fp16
    # initialize generation
    counter = context_length - 1  # Last fixed index is ``counter''
    original_counter = counter
    index = 0 if mems is None else mems.shape[2]  # Next forward starting index, also the length of cache.
    # step-by-step generation
    total_perplexity = 1
    with torch.no_grad():
        while counter < len(given_sentence) + context_length - 1:

            if pos is not None:
                ini_pos = pos[0]
                log_attention_weights_part = torch.zeros(counter + 1).cuda()
                st_pos = pos[1]
                end_pos = pos[2]

                log_attention_weights_part[1:ini_pos] = weight
                log_attention_weights_part[st_pos:end_pos] = weight
            else:
                log_attention_weights_part = None

            logits, *mem_kv = model(
                tokens[:, index:],
                position_ids[..., index: counter + 1],
                attention_mask[..., index: counter + 1, :counter + 1],
                mems=mems,
                log_attention_weights=log_attention_weights_part,
            )

            mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
            counter += 1
            index = counter

            logits = logits.float()

            log_probs = torch.log(F.log_softmax(logits, dim=-1))

            temp_prob = log_probs[0, -1, given_sentence[counter - original_counter - 1]]

            total_perplexity += temp_prob
            # expand tokens with the given sentence

            expand = torch.LongTensor(given_sentence[counter - original_counter - 1: counter - original_counter])
            tokens = torch.cat((tokens, expand.unsqueeze(0).cuda(tokens.device)), dim=1)

    print("So where does this None come from", total_perplexity)
    total_perplexity = total_perplexity / pow(len(given_sentence), 0.6)
    print(total_perplexity)

    return total_perplexity


'''
