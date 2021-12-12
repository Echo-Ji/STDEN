# STDEN

This is the implementation of Spatio-temporal Differential Equation Network (STDEN) in the following paper:
Jiahao Ji, Jingyuan Wang, Zhe Jiang, Jiawei Jiang, and Hu Zhang, Towards Physics-guided Neural Networks for Traffic Flow Prediction, AAAI 2022. 

Thanks [chnsh](https://github.com/chnsh/DCRNN_PyTorch) for the model training framework of this project.

## Requirement

* scipy>=1.5.2
* numpy>=1.19.1
* pandas>=1.1.5
* pyyaml>=5.3.1
* pytorch>=1.7.1
* future>=0.18.2
* torchdiffeq>=0.2.0

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```

## Model Traning and Evaluation

One can run the code by
```bash
# traning for dataset GT-221
python stden_train.py --config_filename=configs/stden_gt.yaml

# testing for dataset GT-221
python stden_eval.py --config_filename=configs/stden_gt.yaml
```
The configuration file of all datasets are as follows:

|dataset|config file|
|:--|:--|
|GT-221|stden_gt.yaml|
|WRS-393|stden_wrs.yaml|
|ZGC-564|stden_zgc.yaml|

Note the data is not public and I am not allowed to distribute it.