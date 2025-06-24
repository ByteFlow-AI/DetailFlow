# !/bin/bash
set -x

source init.sh


VQ_MODLE=$1


filename=$(basename "$VQ_MODLE" .pt)
parent_dir=$(basename "$(dirname "$(dirname "$VQ_MODLE")")")
SAMPLE_DIR=./logs/eval/${parent_dir}/CKPT-${filename}


torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node $ARNOLD_WORKER_GPU \
--master_addr localhost \
--master_port 12738 \
inference/reconstruct_vq.py --mixed-precision ${MIX_PRECISION} --data-path ${IMAGENET_VAL_PATH} --vq-model ${VQ_MODLE} --sample-dir ${SAMPLE_DIR} --per-proc-batch-size 4 ${@:2} 