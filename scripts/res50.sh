    nohup python3 trainval_net_fpn.py --model_name=handobj_100K --net=res50 -bs=2 --lr=0.0025 --s=3 --amp -r models/res50_handobj_100K_fpn/detector_3_10.pth > scripts/output/res50.log 2>&1 &
echo $! > scripts/output/res50_pid.txt