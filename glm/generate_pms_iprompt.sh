#!/bin/bash
CHECKPOINT_PATH=/dataset/fd5061f6/sat_pretrained/glm

source $1
MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

python inference_poems_iprompt.py \
       --mode inference \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --num-beams 6 \
       --no-repeat-ngram-size 0 \
       --length-penalty 0.7 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --output-path samples_glm \
       --batch-size 6 \
       --out-seq-length 100 \
       --mode inference \
       --input-source interactive \
       --sampling-strategy iPromptSearchStrategy \
       --device $2 \
