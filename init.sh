if [ ! -f /tmp/init_detailflow_cloud_env_marker ]
then
    echo "Install pip dependencies"
    
    pip3 install -r requirements.txt

    pip3 install --no-cache-dir git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2

    echo "Prepare transformers library"
    python3 -c "import transformers"

    echo "Write initialized marker"
    touch /tmp/init_detailflow_cloud_env_marker

fi

if [[ "$ARNOLD_DEVICE_TYPE" == *"Ascend"* ]]; then
    echo "RUNNING on $ARNOLD_DEVICE_TYPE"
    export MIX_PRECISION="bf16"
    export MODEL_COMPILE="True"
elif [[ "$ARNOLD_DEVICE_TYPE" == *"V100"* ]]; then
    echo "RUNNING on $ARNOLD_DEVICE_TYPE"
    export MIX_PRECISION="none"
    export MODEL_COMPILE="True"
else
    echo "RUNNING on $ARNOLD_DEVICE_TYPE"
    export MIX_PRECISION="bf16"
    export MODEL_COMPILE="True"
fi

export IMAGENET_PATH="/mnt/bn/cloud-project-lq/code/liuyh/data/vq_data/train" 
export IMAGENET_VAL_PATH="/mnt/bn/cloud-project-lq/code/liuyh/data/vq_data/val"
export CACHE_DATA_PATH="/mnt/bn/cloud-project-lq/data/detailflow/model"
export TORCH_HOME="/mnt/bn/cloud-project-lq/data/detailflow/model/torch_cache"
export IMAGENET_EVAL_NPZ="/mnt/bn/cloud-project-lq/code/liuyh/data/DetailFlow/imagenet_eval/VIRTUAL_imagenet256_labeled.npz"


export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

export CUR_PORT=12738

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${CUR_PORT}"