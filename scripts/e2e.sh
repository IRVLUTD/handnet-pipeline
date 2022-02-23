nohup python3 trainval_net_e2e.py --model_name=e2e_dexycb --net=e2e -bs=32 --lr=1e-3 --s=1 --optimizer=adamw --epoch=35 --wd=0 --lr-steps=30 -j=8 -r=models/e2e_e2e_dexycb/faster_rcnn_1_9.pth > scripts/output/e2e.log 2>&1 &
echo $! > scripts/output/e2e.txt