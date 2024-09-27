# Fixes and Notices on GlobalMapNet

## 20240927: Fix the environment
If you meet one of the following problems, try the methods below:

### 1. version problems of dependencies
**solution: try install the following version:**
```python
yapf                    0.40.1 # must below 0.40.2
numpy                   1.23.5
numba                   0.53.0
ipython
```

for python3.9
```python
networkx                    3.2.1 # >=2.5
```

### 2. mmcv install - fatal error: cuda_runtime_api.h: No such file or directory
**solution: manually export env variables**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
```

### 3. NCCL timing out
terminal out:
```
Watchdog caught collective operation timeout: 
WorkNCCL(OpType=BROADCAST, Timeout(ms)=1800000) ran for 1804406 milliseconds before timing out.
```

**solution: try to set the following global environments:**
```bash
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### 4. ModuleNotFoundError: No module named 'plugin'
**solution: manually set PYTHONPATH**
```bash
PYTHONPATH=. python plugin/models/globalmapnet/map_utils/nusc_gt_map.py --location all
```

## 20240927: Notice on experimental setups

### 1. training and evaluation on multiple GPUs
Our experiments are done on a single GPU with Gradient Accumulation. If you have multiple GPUs, simply revise the config files, and run with the number of GPUs:
```
bash tools/dist_train.sh plugin/configs/nusc_newsplit_480_60x30_24e_global_train.py ${GPUS}
```

### 2. "update_kwargs" in global_map_config for training and evaluation
The "update_kwargs" gives the arguments in global map update. We use the same value in StreamMapNet and GlobalMapNet, and across nuScene and Argoverse2. 

However, we empirically find using different "threshold" (in map matching) and "buffer_distance" (in Map NMS) in training and evaluation gives better results. Thus, all the checkpoints are trained under the following setting:
```
update_kwargs={'threshold': [0.05, 1.0, 1.0], 'sample_num': 100, 'simplify': True, 'buffer_distance': 0.5, 'biou_threshold': 0.1}
```

For the choice of evaluation parameters, see **ablation on parameters of GlobalMapBuilder** in the paper.

### 3. parameters of GlobalMapBuilder in visualization
We use evaluation parameters for visualization as default. For video presentation of cross-scene evaluation, we set Map NMS purge mode to PURGE_BUFFER_ZONE and adjust the buffer_distance to generate a cleaner result. However, the mGAP will be lower under such setting.