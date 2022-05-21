#!/bin/bash
conda install mamba -n base -c conda-forge --yes
mamba env create
eval "$(conda shell.bash hook)"
conda activate handnet_pipeline

ln -s ../../../misc/mano dex-ycb-toolkit/manopth/mano/models
ln -s ../mano misc/mano/models

cd dex-ycb-toolkit
pip install -e .

cd manopth
pip install -e .
cd ../../pose2mesh/lib
pip install -e .

cd ../../lib
pip install -e .

# rospy dependency with conda
echo 'export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages' >> ~/.zshrc

mamba install pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 pytorch-lightning=1.5.10 -c conda-forge -c pytorch --yes