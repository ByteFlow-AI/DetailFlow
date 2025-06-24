# !/bin/bash
set -x

source init.sh

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[-1]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

CFG=${3:-1.5}
TOP_K=8192
TOP_P=1
VQ_EMA=False
GPT_EMA=True
GLOBAL_SEED=${4:-0}

VQ_MODLE=$1
RQ_MODEL=$2

vq_filename=$(basename "$VQ_MODLE" .pt)
vq_parent_dir=$(basename "$(dirname "$(dirname "$VQ_MODLE")")")

ar_filename=$(basename "$RQ_MODEL" .pt)
ar_parent_dir=$(basename "$(dirname "$(dirname "$RQ_MODEL")")")

SAMPLE_DIR=./logs/ar_eval/${vq_parent_dir}-${vq_filename}/${ar_parent_dir}-${ar_filename}/CFG-${CFG}-TOPK-${TOP_K}-TOPP-${TOP_P}-VQEMA-${VQ_EMA}-GPTEMA-${GPT_EMA}-Seed-${GLOBAL_SEED}

torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node $ARNOLD_WORKER_GPU \
--master_addr localhost \
--master_port 12738 \
autoregressive/sample/sample_c2i_ddp.py \
--vq_checkpoint ${VQ_MODLE} \
--gpt_checkpoint ${RQ_MODEL} \
--image_size 256 \
--cfg_scale ${CFG} \
--per_process_batch_size 4 \
--sample_output_dir ${SAMPLE_DIR}/result \
--precision ${MIX_PRECISION} \
--top_k ${TOP_K} \
--top_p ${TOP_P} \
--use_vq_ema ${VQ_EMA} \
--use_gpt_ema ${GPT_EMA} \
--global_seed ${GLOBAL_SEED}


export CUDA_VISIBLE_DEVICES=0 
npz_file_path=$(find "$SAMPLE_DIR" -type f -name "*.npz" | head -n 1)

python3 evaluations/c2i/evaluator.py ${IMAGENET_EVAL_NPZ} ${npz_file_path}