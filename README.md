# Benchmark RWKV on Long Range Arena


## Data Preparation
First download and unpack the LRA dataset
```
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar xvf lra_release.gz
```
## Training Commands
- listops: RWKV_T_MAX=2048 CUDA_VISIBLE_DEVICES=0,5,6,7 RWKV_FLOAT_MODE=fp32 python -m train wandb=null experiment=lra/rwkv-listops trainer.devices=4
- cifar:
- aan: RWKV_FLOAT_MODE=fp16 python -m train trainer.devices=8 experiment=lra/rwkv-aan wandb=null
