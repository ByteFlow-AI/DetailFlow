# DetailFlow🚀: 1D Coarse-to-Fine Autoregressive Image Generation via Next-Detail Prediction
<div align="center">

[![DetailFlow](https://img.shields.io/badge/Paper-DetailFlow-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2505.21473)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-DetailFlow-yellow)](https://huggingface.co/collections/ByteFlow-AI/detailflow-6843f28683925bfdb4806cf6)&nbsp;
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=ByteFlow-AI.DetailFlow)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/ByteFlow-AI/DetailFlow?color=blue&label=Issues)](https://github.com/ByteFlow-AI/DetailFlow/issues?q=is%3Aissue+is%3Aclosed) 


</div>


## 🌿 Introduction

We present DetailFlow, a coarse-to-fine 1D autoregressive (AR) image generation method that models images through a novel next-detail prediction strategy. By learning a resolution-aware token sequence supervised with progressively degraded images, DetailFlow enables the generation process to start from the global structure and incrementally refine details. 


<div align='center'>
<img src="./assets/demo.png" class="interpolation-image" alt="teasor." height="100%" width="100%" />
</div>

DetailFlow encodes tokens with an inherent semantic ordering, where each subsequent token contributes additional high-resolution information.  On the ImageNet 256×256 benchmark, our method achieves 2.96 gFID with 128 tokens, outperforming VAR (3.3 FID) and FlexVAR (3.05 FID), which both require 680 tokens in their AR models. Moreover, due to the significantly reduced token count and parallel inference mechanism, our method runs nearly 2× faster inference speed compared to VAR and FlexVAR. 

<div align='center'>
<img src="./assets/method.png" class="interpolation-image" alt="method." height="50%" width="50%" />
</div>

## 📰 News

**2025.06.24**: 🎉🎉🎉 The code and model weights of Detailflow have been open-sourced.

**2025.06.16**: The code and model weights are finalized and currently undergoing legal review. We expect to release them soon.

**2025.05.28**:  🎉🎉🎉 DetailFlow is released! 🎉🎉🎉  See our [paper](https://arxiv.org/abs/2505.21473) .


## Model zoo

[Pre-trained siglip2](https://huggingface.co/google/siglip2-so400m-patch16-naflex)

|   Tokenizer  | reso. |   rFID    | AR model | gFID |
|:----------:|:-----:|:--------:|:---------:|:-------:|
|  [DetailFlow-16](https://huggingface.co/ByteFlow-AI/DetailFlow-16)   |  256  |   1.13   |    [DetailFlow-16-GPT-L](https://huggingface.co/ByteFlow-AI/DetailFlow-16-GPT-L)    |  2.96   |
|  [DetailFlow-32](https://huggingface.co/ByteFlow-AI/DetailFlow-32)   |  256  |   0.80   |    [DetailFlow-32-GPT-L](https://huggingface.co/ByteFlow-AI/DetailFlow-32-GPT-L)    |  2.75   |
|  [DetailFlow-64](https://huggingface.co/ByteFlow-AI/DetailFlow-64)   |  256  |   0.52   |    [DetailFlow-64-GPT-L](https://huggingface.co/ByteFlow-AI/DetailFlow-64-GPT-L)    |  2.59   |

## Installation

1. Install `torch>=2.1.2`.
2. Install other pip packages via `bash init.sh`.
3. Prepare the [ImageNet](http://image-net.org/) dataset
    <details>
    <summary> assume the ImageNet is in `/path/to/imagenet`. It should be like this:</summary>

    ```
    /path/to/imagenet/:
        train/:
            n01440764: 
                many_images.JPEG ...
            n01443537:
                many_images.JPEG ...
        val/:
            n01440764:
                ILSVRC2012_val_00000293.JPEG ...
            n01443537:
                ILSVRC2012_val_00000236.JPEG ...
    ```
    </details>
4. Modify the path-related environment variables in the `init.sh` file.

## Training Scripts

### Tokenizer

50 epochs can already yield good results, worth a try.

```shell
# 128 tokens, group size 8
bash scripts/train_tokenizer.sh detailflow_128token --config configs/vq_8k_siglip_b_res_p02_pw15_enc.yaml --num-latent-tokens 128 --group-size 8 --epochs 250 --global-token-loss-weight 1

# 256 tokens, group size 8
bash scripts/train_tokenizer.sh detailflow_256token --config configs/vq_8k_siglip_b_res_p02_pw15_enc.yaml --num-latent-tokens 256 --group-size 8 --epochs 250 --global-token-loss-weight 1

# 512 tokens, group size 8
bash scripts/train_tokenizer.sh detailflow_512token --config configs/vq_8k_siglip_b_res_p02_pw15_enc.yaml --num-latent-tokens 512 --group-size 8 --epochs 250 --global-token-loss-weight 1
```

### AR Model

```shell
# DetailFlow-16 and DetailFlow-32 can better utilize GPU resources by using a batch size of 512.
bash scripts/train_c2i_token.sh /xxx/DetailFlow-16/checkpoints/128.pt demo_task_name --global-batch-size 512 --epochs 300

```

The model will be stored in the directory ./logs/task/job_name.

* config.json and config.yaml: files contain some configuration information about the model and are necessary for loading the model during subsequent inference.

* checkpoints: These store the ckpt files, with only the most recent 10 ckpt files being retained.

## Evaluation

* If you encounter the error `torch._dynamo.exc.CacheLimitExceeded: cache_size_limit reached`, you can try setting `args.compile = False`, but this will slow down the inference speed.
* Due to randomness, evaluation metrics may vary by up to 0.3; we recommend running with multiple seeds and reporting the average value.

```shell
# Tokenizer
bash scripts/eval_tokenizer.sh /path_to_ckpt/xxx.pt

# ar model, 128 tokens
bash scripts/eval_c2i.sh /xxx/DetailFlow-16/checkpoints/128.pt /xxx/DetailFlow-16-GPT-L/checkpoints/128.pt 1.45

# ar model, 256 tokens
bash scripts/eval_c2i.sh /xxx/DetailFlow-32/checkpoints/256.pt /xxx/DetailFlow-32-GPT-L/checkpoints/256.pt 1.5

# ar model, 512 tokens
bash scripts/eval_c2i.sh /xxx/DetailFlow-64/checkpoints/512.pt /xxx/DetailFlow-64-GPT-L/checkpoints/512.pt 1.32

```


## Acknowledgement

We thank the great work from [SoftVQ-VAE](https://github.com/Hhhhhhao/continuous_tokenizer/tree/main), [LlamaGen](https://github.com/FoundationVision/LlamaGen) and [PAR](https://github.com/YuqingWang1029/PAR/tree/main).

## 📄 Citation

If our work assists your research, feel free to give us a star ⭐ or cite us using

```
@article{liu2025detailflow,
  title={DetailFlow: 1D Coarse-to-Fine Autoregressive Image Generation via Next-Detail Prediction},
  author={Liu, Yiheng and Qu, Liao and Zhang, Huichao and Wang, Xu and Jiang, Yi and Gao, Yiming and Ye, Hu and Li, Xian and Wang, Shuai and Du, Daniel K and others},
  journal={arXiv preprint arXiv:2505.21473},
  year={2025}
}
```

## 🔥 Open positions
We are hiring interns and full-time researchers at the ByteFlow Group, ByteDance, with a focus on multimodal understanding and generation (preferred base: Beijing, Shenzhen and Hangzhou). If you are interested, please contact yolomemos@gmail.com.