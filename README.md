# E2E-HandNet

**In development**
 - `demo.py`
 - `trainval_net_e2e.py` still needs FCOS training loop (currently implemented in `pretrain_fcos.py`)
 - Naming of output files for `trainval_net_e2e.py` is inaccurate
 - DexYCB metrics of E2E-HandNet
 - Reorganization of file names and functions

**Completed**
 - `trainval_net_fpn.py` & `pretrain_fcos.py` (detection training + test)
 - MANO network training from FPN outputs (training)
 - Data preprocessing and caching
    - Note: caching of feature maps is not implemented yet