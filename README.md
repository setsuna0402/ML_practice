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

# Generative_ML_NTU_2024
This folder contains projects focusing on generative ML methods.
Current topics:
A) Fine-tuning LLM (Qwen2.5-3B) to a poem generator.
Key features: Use Low-Rank Adaptation (LoRA) with NF4 to do parameter fine-tuning.
              Adopt 4-bit quantisation to pre-trained model to allow performing LLM fine-tuning in a gaming GPU, like RTX 3060Ti (8GB).
