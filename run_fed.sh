num_users=500
frac=0.1
local_ep=10
local_bs=50
lr=0.01
model='resnet'
norm='batch_norm'
dataset='cifar100'

for epochs in 200 400 
do
    CUDA_VISIBLE_DEVICES=0 python main_fed.py  \
    --epochs=$epochs --num_users=$num_users \
    --frac=${frac} --local_ep=${local_ep} \
    --local_bs=${local_bs} --lr=${lr} \
    --model=${model} --norm=${norm} \
    --dataset=${dataset}
done