#!/bin/bash
conda env create
eval "$(conda shell.bash hook)"
conda activate handnet_pipeline


ln -s ../../../misc/mano dex-ycb-toolkit/manopth/mano/models
ln -s ../mano misc/mano/models

cd dex-ycb-toolkit
pip install -e .

cd bop_toolkit
pip install -r requirements.txt
cd ..

cd manopth
pip install -e .

# rospy dependency with conda
echo 'export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages' >> ~/.zshrc

conda install pytorch=1.10.2 torchvision==0.11.3 torchtext cudatoolkit=11.3 pytorch-lightning=1.5.10 -c pytorch -c conda-forge --yes