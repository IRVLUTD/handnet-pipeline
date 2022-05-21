#!/bin/bash

mkdir models
wget https://utdallas.box.com/shared/static/xgusayqw8xgmxnfzzqi1r0hvrxukknzo.pth -O "models/a2j.pth"
wget https://utdallas.box.com/shared/static/s6igmdvxsis8sib101xko7o44gwompgz.pth -O "models/fcos.pth"
mkdir experiment
mkdir experiment/pose2mesh_manoJ_train_freihand
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1Ub5WtF5oesIzGPLTLaNSN2TxHJDjYuX2&confirm=t" -O "experiment/pose2mesh_manoJ_train_freihand/final.pth.tar" 