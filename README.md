# GAIL-for-mountain-car-Tensorflow
GAIL implementation for Mountain Car on gym environment in Tensorflow 2 [link](https://gym.openai.com/envs/MountainCar-v0/)


![](images/mountaincar.jpg)


GAIL: Generative Adversarial Imitation Learning [link](https://arxiv.org/abs/1606.03476)

## This Repo Include:
1. State-Action pair data from Expert Behavior
2. GAIL model to train
3. Output Sequence of Action from trained Model

(Note: THe starting point for the car is random, so the current output action might not work for a different starting point)

## To Train:
Simply Run train_gail.py

The output model and actions for best score will be saved in the out folder.

You can also use the trained model for inference.

## To DO:
1. Add Inference Model
2. Add gif for trained model inference
