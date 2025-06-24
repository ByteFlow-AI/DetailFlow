# !/bin/bash
set -x

CUR_DIR=$(dirname $(dirname $(realpath "$0")))

source init.sh

TASK_NAME=$1
OUTDIR=./logs/task/${TASK_NAME}

torchrun \
--nnodes $ARNOLD_WORKER_NUM \
--node_rank $ARNOLD_ID \
--nproc_per_node $ARNOLD_WORKER_GPU \
--master_addr $METIS_WORKER_0_HOST \
--master_port $CUR_PORT \
train/train_tokenizer.py \
--data-path ${IMAGENET_PATH} \
--mixed-precision ${MIX_PRECISION} \
--results-dir ${OUTDIR} ${@:2} 

