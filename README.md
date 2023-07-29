# Rainbow DDPG with ResNet18 Backbone

This repository contains a modified version of the Rainbow DDPG algorithm from the paper "Sim-to-Real Reinforcement Learning for Deformable Object Manipulation". In this version, the backbone of actor has been changed to ResNet18. This repository also includes a toy pushing task to demonstrate how to use the code.

## Instructions

The code was tested on Ubuntu20.04 with Python3.7. Use of virtualenvs is recommended. To run the training:

```
pip install -r requirements.txt
python main.py
```

Original model: Running a full training (250 epochs) may take more than 24 hours on a machine with Nvidia Titan GPU and use a considerable amount of memory (by authors of the paper).
After replacing the actor backbone with ResNet18: It takes more than 4 days for running 150 epochs on a machine with RTX 3090.

Results:
This modification has led to some instability in the learning process compared to the original version (See figure below). Further tuning and optimization may be required to achieve stable learning with the ResNet18 backbone.

Besides, to run a demonstration of the toy task:

```
pip install -r requirements.txt
python run_demo.py
```

Please note that the hyper parameters are not necessarily optimised for the task.



## References

For a complete list of references, please see the accompanying paper.

The learning algorithm is based on OpenAI baselines (https://github.com/openai/baselines), the perlin noise file is heavily based on https://github.com/nikagra/python-noise/blob/master/noise.py and robot meshes are generated from https://github.com/Kinovarobotics/kinova-ros. 
