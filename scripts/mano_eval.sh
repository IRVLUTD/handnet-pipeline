CUDA_VISIBLE_DEVICES=0 nohup python3 trainval_net_mano.py --model_name=dexycb --net=e2e -r=models/e2e_dexycb_2/e2e_handnet_2_45.pth --s=2 --test-only > scripts/output/mano_eval.log 2>&1 &
echo $! > scripts/output/mano_eval_pid.txt