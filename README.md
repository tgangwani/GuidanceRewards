This repository contains code for our paper [Learning Guidance Rewards with Trajectory-space Smoothing](https://arxiv.org/abs/2010.12718), published at the Conference on Neural Information Processing Systems (NeurIPS 2020).

The code reuses the Pytorch SAC code from [this awesome repository](https://github.com/denisyarats/pytorch_sac). It was tested with the following packages:

* python 3.6.6
* pytorch 0.4.1
* gym 0.10.8
* [hydra](https://github.com/facebookresearch/hydra) 0.11.3


## Running command
To run the SAC experiments on MuJoCo, use the command below. The hyperparameters are mentioned in the ```config``` folder. Check the file _run_cmds.sh_ for further commands.

```
python main.py env_name="Hopper-v2" seed=$RANDOM 
```

## Credits
1. [denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac)
