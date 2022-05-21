
CUDA_VISIBLE_DEVICES=1 nohup python3 trainval_net_a2j.py --model_name=dexycb --net=a2j --amp --test-only -r=models/a2j_dexycb_1/a2j_45.pth > scripts/output/a2j_eval.log 2>&1 &
echo $! > scripts/output/a2j_eval_pid.txt