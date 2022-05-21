# HandNet-Pipeline

A pipeline for hand detection and hand pose estimation using FCOS and A2J. Training and inference scripts are provided.

## Environment setup
1. Make sure to have Anaconda installed before initializing the environment.
2. Download MANO models and code (`mano_v1_2.zip`) from the [MANO website](https://mano.is.tue.mpg.de), unzip, and place `models/MANO_*.pkl` files under `misc/mano`.
3. Run the following command to install the dependencies:
    ```
    ./scripts/init_env.sh
    ```
4. Activate the environment using the following command:
    ```
    conda activate handnet_pipeline
    ```
5. Download the pretrained models using the following command:
    ```
    ./scripts/download_models.sh
    ```
6. Run index preprocessing for DexYCB
   ```
   python3 refine_idx_gen.py
   ```

## ROS Demo
1. Make sure ROS environment variables are exported before running the demo.
2. Run the following command to start the demo (with pretrained models):
    ```
    python ros_demo.py
    ```

# Training

## Detector Training
 - Download [100DOH](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html) (Pascal VOC format) and unzip into `data` directory
 - Run the following command to train the detector:
    ```
    python trainval_net_fcos.py
    ```
## Detector Testing
 - Run the following command to test the detector:
    ```
    python trainval_net_fcos.py --test-only --resume=/path/to/checkpoint
    ```
## A2J Training
 - Download [DexYCB](https://dex-ycb.github.io/) (subject 1 required), unzip, and run the following command
    ```Shell
    export DEX_YCB_DIR=/path/to/dex-ycb
    ```
    `$DEX_YCB_DIR` should be a folder with the following structure:

    ```Shell
    ├── 20200709-subject-01/
    ├── 20200813-subject-02/
    ├── ...
    ├── calibration/
    └── models/
    ```
 - Run the following command to train the A2J model:
    ```
    python trainval_net_a2j.py fit
    ```
## A2J Testing
 - Modify `config/a2j.yaml` to set the path to the trained A2J checkpoint.
 - Run the following command to test the A2J model:
    ```
    python trainval_net_a2j.py test
    ```
 - If using pretrained model, load pretrained weights in LightningModule `init` instead (line 273).
