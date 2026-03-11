# ML_practice
Various practices for ML. Since some of the  projects originated from NTU ML courses or courses from other universities, commercial use should be avoided.

# MNIST_Classifier_and_GAN: 
This folder contains several projects: Classifier, GAN and WGAN. The default ML libraries of the first two are Keras and TensorFlow, since I did them a few years ago. 
WGAN is coded in PyTorch. And, indeed, training WGAN is much simpler than training the traditional GAN.  

# ML_NN_architecture_NTU_2021:
Currently, there are four completed projects in this folder. 
They are 
Food classifiers with ResNet, image augmentations and CutMix (HW3) 
Specker's voice identifiers with transformer, conformer, self-attentive pooling and additive margin softmax (HW4).
English to Chinese translator with transformer encoder and decoder.
Rocker lander with reinforcement learning. Three methods of RL are included: 
A) on-policy gradient with accumulated decay reward
B) on-policy gradient with accumulated decay reward, accompanied by the actor-critic method, whose value function neural network is estimated by Monte Carlo
C) similar to B), but the value function nn is estimated by one-step temporal difference
Method B demonstrates the best performance. It is stable: the landing process is smooth, and the rocker almost always lands in the landing zone.
