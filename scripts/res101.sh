DISPLAY=:0 nvidia-settings -a '[gpu:0]/GPUFanControlState=1'
DISPLAY=:0 nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=100'

nohup python3 trainval_net_fpn.py --model_name=handobj_100K --net=res101 -bs=2 --lr=0.0025 --epochs=26 --s=4 --amp -r models/res101_handobj_100K_fpn/faster_rcnn_4_18.pth > scripts/output/res101.log 2>&1 &
echo $! > scripts/output/res101_pid.txt

wait
DISPLAY=:0 nvidia-settings -a '[gpu:0]/GPUFanControlState=0'