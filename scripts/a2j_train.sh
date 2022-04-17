
CUDA_VISIBLE_DEVICES=1 nohup python3 trainval_net_a2j.py --model_name=dexycb --net=a2j -bs=64 --lr=0.00035 --s=1 --epoch=30 --wd=1e-4 --lr-gamma=0.2 -j=8 --amp --optimizer=adamw > scripts/output/a2j_train.log 2>&1 &
echo $! > scripts/output/a2j_train_pid.txt