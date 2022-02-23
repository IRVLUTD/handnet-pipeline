DISPLAY=:0 nvidia-settings -a '[gpu:1]/GPUFanControlState=1'
DISPLAY=:0 nvidia-settings -a '[fan:1]/GPUTargetFanSpeed=100'

CUDA_VISIBLE_DEVICES=1 nohup python3 trainval_net_fpn.py --model_name=handobj_100K --net=fcos --test-only -r models/fcos_handobj_100K_fpn/faster_rcnn_5_25.pth > scripts/output/fcos_eval.log 2>&1 &
echo $! > scripts/output/fcos_eval_pid.txt

wait
DISPLAY=:0 nvidia-settings -a '[gpu:1]/GPUFanControlState=0'