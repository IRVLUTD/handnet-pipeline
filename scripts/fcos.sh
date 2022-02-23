# DISPLAY=:0 nvidia-settings -a '[gpu:1]/GPUFanControlState=1'
# DISPLAY=:0 nvidia-settings -a '[fan:1]/GPUTargetFanSpeed=100'

nohup python3 trainval_net_fpn.py --model_name=handobj_100K_res18 --net=fcos -bs=2 --lr=0.00125 --s=1 --amp > scripts/output/fcos.log 2>&1 &
echo $! > scripts/output/fcos_pid.txt

# wait
# DISPLAY=:0 nvidia-settings -a '[gpu:1]/GPUFanControlState=0'