# Aspect-based-Few-shot

# Shapes Data Generation

The original code of generation originates from the following paper:
[Knowledge Elicitation Using Deep Metric Learning and Psychometric Testing](https://link.springer.com/chapter/10.1007/978-3-030-67661-2_10) by [Lu Yin](https://github.com/luuyin), Vlado Menkovski and Mykola Pechenizkiy

We alterned the code to include the generation of a npy file of the individual images

# Sprites Data Generation

The original code of generation originates from the following paper:

[Disentangled Sequential Autoencoder](https://arxiv.org/abs/1803.02991) by [Yingzhen Li](http://yingzhenli.net) and [Stephan Mandt](http://www.stephanmandt.com)

We altered the code to only include single frames rather than a sequence of frames.

The sprite sheets are collected from the following open-source projects:

[Liberated Pixel Cup](http://lpc.opengameart.org)
[Universal-LPC-spritesheet](https://github.com/sanderfrenken/Universal-LPC-Spritesheet-Character-Generator)

We do NOT claim the ownership of the original sprite sheets. But if you will be using the code in this repo to generate the single frame sprite data, then consider citing the two original open-source projects, and our paper.

## Create the dataset
You need to install python packages [Pillow](https://pillow.readthedocs.io/) and [imageio](https://imageio.github.io) first. Using pip should be sufficient.

Then clone the repo to your working directory. Then first run

    python random_character.py
    
This will create frames/ folder and generate .png files of 1296 unique characters with different actions.

Then run

    python frame_to_npy.py
    
This will read the .png files in frames/, create path npy/, and generate numpy data files .npy.

After that if you don't want to retain the .png files, simply run

    rm -rf frames/
    
