#!/bin/bash
conda env create
eval "$(conda shell.bash hook)"
conda activate handnet
conda install pytorch=1.10.2 torchvision cudatoolkit=11.3 -c pytorch --yes


ln -s ../../../misc/mano dex-ycb-toolkit/manopth/mano/models
ln -s ../mano misc/mano/models
cd lib
python setup.py build develop
cd ..

cd dex-ycb-toolkit
pip install -e .

cd bop_toolkit
pip install -r requirements.txt
cd ..

cd manopth
pip install -e .