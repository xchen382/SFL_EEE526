num_users=10
frac=1
local_bs=100
lr=0.01
server_lr=0.02
model='vgg11_bn'
norm='group_norm' #'batch_norm'
dataset='cifar10'
cut=5
epochs=200


CUDA_VISIBLE_DEVICES=0 python main_sfl.py  --save_dir=SFL/${dataset}_${model}_num_users${num_users}_cut${cut}_frac${frac}_iid \
--epochs=$epochs --num_users=$num_users \
--frac=${frac} \
--local_bs=${local_bs} --lr=${lr} --server_lr=${server_lr} \
--model=${model} --norm=${norm} \
--dataset=${dataset} --cut=${cut} --iid 

