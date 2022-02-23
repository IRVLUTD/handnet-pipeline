


DISPLAY=:0 nvidia-settings -a '[gpu:0]/GPUFanControlState=1'
DISPLAY=:0 nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=100'

nohup python3 pretrain_fcos.py --model_name dexycb --net=fcos -bs=2 --lr=0.00125 --s=1 --amp --epoch=26 > scripts/output/fcos_pretrain.log 2>&1 &
echo $! > scripts/output/fcos_pretrain_pid.txt

wait
DISPLAY=:0 nvidia-settings -a '[gpu:0]/GPUFanControlState=0'