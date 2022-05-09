nohup python trainval_net_a2j_lightning.py fit > scripts/output/a2j_lightning_train.log 2>&1 &
echo $! > scripts/output/a2j_lightning_train_pid.txt