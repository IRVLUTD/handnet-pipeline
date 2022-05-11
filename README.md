# E2E-HandNet

**In development**
 - `trainval_net_e2e.py` (end-to-end training)
    - Naming of output files for `trainval_net_e2e.py` is inaccurate
 - lightning implementation of fcos
 - custom naming of checkpoints for `trainval_net_a2j_lightning.py`

**Completed**
 - `trainval_net_fpn.py` & `pretrain_fcos.py` (detection training + test)
 - `trainval_net_a2j.py` (a2j training + test + lightning)
 - Data preprocessing and caching
 - `trainval_net_e2e.py` (end-to-end test)
 - `ros_demo.py` (robot demo)