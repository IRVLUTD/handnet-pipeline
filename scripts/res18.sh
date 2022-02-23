DISPLAY=:0 nvidia-settings -a '[gpu:1]/GPUFanControlState=1'
DISPLAY=:0 nvidia-settings -a '[fan:1]/GPUTargetFanSpeed=100'

CUDA_VISIBLE_DEVICES=1 nohup python3 trainval_net_fpn.py --model_name=handobj_100K --net=res18 -bs=2 --lr=0.0025 --s=2 --amp -r models/res18_handobj_100K_fpn/faster_rcnn_2_22.pth > scripts/output/res18.log 2>&1 &
echo $! > scripts/output/res18_pid.txt

wait
DISPLAY=:0 nvidia-settings -a '[gpu:1]/GPUFanControlState=0'