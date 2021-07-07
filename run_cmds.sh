#!/usr/bin/env bash

python main.py env_name="Hopper-v2" seed=$RANDOM 
python main.py env_name="Walker2d-v2" seed=$RANDOM 
python main.py env_name="HalfCheetah-v2" seed=$RANDOM 
python main.py env_name="Swimmer-v2" seed=$RANDOM 

# for Ant and Humanoid, we train for 3M timesteps and also employ extra exploration
python main.py env_name="Ant-v2" num_train_steps=3e6 exploration.num_periodic_explr=3 seed=$RANDOM 
python main.py env_name="Humanoid-v2" num_train_steps=3e6 exploration.num_periodic_explr=3 mh_buffer_capacity=50 seed=$RANDOM 
