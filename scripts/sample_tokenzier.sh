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


VQ_MODLE=$1


filename=$(basename "$VQ_MODLE" .pt)
parent_dir=$(basename "$(dirname "$(dirname "$VQ_MODLE")")")
SAMPLE_DIR=./logs/samples/${parent_dir}/CKPT-${filename}


torchrun \
--nnodes $ARNOLD_WORKER_NUM \
--node_rank $ARNOLD_ID \
--nproc_per_node $ARNOLD_WORKER_GPU \
--master_addr $METIS_WORKER_0_HOST \
--master_port $port \
inference/sample_vq.py --mixed-precision ${MIX_PRECISION} --data-path ${IMAGENET_VAL_PATH} --vq-model ${VQ_MODLE} --sample-dir ${SAMPLE_DIR} --per-proc-batch-size 4 --vq-ema False