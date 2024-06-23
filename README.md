# Aspect-based-Few-shot

We generalize the formulation of few-shot learning by introducing the concept of an aspect. In the traditional formulation of few-shot learning, there is an underlying assumption that a single \"true\" label defines the content of each data point. This label serves as a basis for the comparison between the query object and the objects in the support set. However, when a human expert is asked to implement the same task without being given a predefined set of labels, they typically consider the rest of the data points in the support set as context. This context helps to understand at which level of abstraction and from which aspect to implement the comparison. In this work, we introduce a novel architecture and training procedure that develops a context given the query and support set and implements aspect-based few-shot learning that is not limited to a predetermined set of classes. We demonstrate that our method is capable of forming and using an aspect for few-shot learning on the Geometrics Shapes and Sprites dataset. The results validate the feasibility of our approach compared to traditional few-shot learning.

The paper will be linked when done. Currently, the master thesis itself is included in the repo.

## Shapes Data Generation

The original code of generation originates from the following paper:

[Knowledge Elicitation Using Deep Metric Learning and Psychometric Testing](https://link.springer.com/chapter/10.1007/978-3-030-67661-2_10) by [Lu Yin](https://github.com/luuyin), Vlado Menkovski and Mykola Pechenizkiy

We alterned the code to include the generation of a npy file of the individual images

## Sprites Data Generation

The original code of generation originates from the following paper:

[Disentangled Sequential Autoencoder](https://arxiv.org/abs/1803.02991) by [Yingzhen Li](http://yingzhenli.net) and [Stephan Mandt](http://www.stephanmandt.com)

We altered the code to only include single frames rather than a sequence of frames.

The sprite sheets are collected from the following open-source projects:

[Liberated Pixel Cup](http://lpc.opengameart.org)

[Universal-LPC-spritesheet](https://github.com/sanderfrenken/Universal-LPC-Spritesheet-Character-Generator)

We do NOT claim the ownership of the original sprite sheets. But if you will be using the code in this repo to generate the single frame sprite data, then consider citing the two original open-source projects, and our paper.

## Create the dataset
Clone the repo to your working directory. Then first run

    python data/Sprites/character_generation.py
    
This will create a folder and generate .png files of 72000 unique characters with different features. It will also turn a generate numpy data file .npy

Then run each cell in the python notebook

    data/Shapes/Creat_simple.ipynb
    
This will again create a folder and generate .png files of 3072 unique shapes with different features. It will also turn a generate numpy data file .npy

## Run training and testing

The notebook "Training and Testing".ipynb contain the training of the model with the respective experiments
    
