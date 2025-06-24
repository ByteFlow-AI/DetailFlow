# !/bin/bash
set -x

CUR_DIR=$(dirname $(dirname $(realpath "$0")))

source init.sh

VQ_MODLE=$1

filename=$(basename "$VQ_MODLE" .pt)
parent_dir=$(basename "$(dirname "$(dirname "$VQ_MODLE")")")

CODE_DIR=./logs/extracted/${parent_dir}/CKPT${filename}_No_EMA_imagenet_code_c2i_flip_ten_crop

# use a tokenizer to extract code.
torchrun \
--nnodes $ARNOLD_WORKER_NUM \
--node_rank $ARNOLD_ID \
--nproc_per_node $ARNOLD_WORKER_GPU \
--master_addr $METIS_WORKER_0_HOST \
--master_port $CUR_PORT \
autoregressive/train/extract_codes_c2i.py \
--data-path ${IMAGENET_PATH} \
--vq-path ${VQ_MODLE} \
--code-path ${CODE_DIR} \
--mixed-precision ${MIX_PRECISION} \
--ten-crop \
--vq-ema False

sleep 3m

TASK_NAME=ar_${parent_dir}_ckpt${filename}
OUTDIR=./logs/task/${TASK_NAME}_$2

# train the ar model
torchrun \
--nnodes $ARNOLD_WORKER_NUM \
--node_rank $ARNOLD_ID \
--nproc_per_node $ARNOLD_WORKER_GPU \
--master_addr $METIS_WORKER_0_HOST \
--master_port $CUR_PORT \
autoregressive/train/train_c2i.py \
--config configs/ar_l.yaml \
--mixed-precision ${MIX_PRECISION} \
--code-path ${CODE_DIR} \
--vq-ema False \
--compile ${MODEL_COMPILE} \
--results-dir ${OUTDIR} ${@:3} 



