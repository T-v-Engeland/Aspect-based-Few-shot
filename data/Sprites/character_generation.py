from PIL import Image
import sys, os
import numpy as np
import imageio
from scipy import ndimage
import itertools

from PIL import Image
import os, sys
import cv2
import numpy as np
import pandas as pd


def load_dataset():
    # Append images to a list
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item).convert("RGB")
            im = np.array(im)
            x_train.append(im)

def gen_char(body, bottom, top, hair):

    np.random.seed(seed)

    # then randomly sample the components
    attributes = {'body': str(body), 'bottomwear': str(bottom),
                  'topwear': str(top), 'hair': str(hair)}
    img_list = []
    for attr in ['body', 'bottomwear', 'topwear', 'hair']:
        path = attr + '/'
        filename = attributes[attr] + '.png'
        #print path+filename
        img_list.append(Image.open(path + filename))
    # shoes
    img_list.append(Image.open('shoes/0.png'))

    # then merge all!
    f = Image.new('RGBA', img_list[0].size, 'white')
    for i in range(len(img_list)):
        f = Image.alpha_composite(f, img_list[i].convert('RGBA'))

    # save image
    classname = str(body)+str(bottom)+str(top)+str(hair)#+str(weapon)
    f.save('%s.png' % classname)

    img = Image.open('%s.png' % classname)
    # crop to 64 * 64
    width = 64; height = 64
    imgwidth, imgheight = img.size
    N_width = imgwidth / width
    N_height = imgheight / height
    path = 'dataSprites112/'
    if not os.path.exists(path):
        os.makedirs(path)

    actions = {
    'spellcard': {'back': range(0, 7), 'left': range(13, 20),
                  'front': range(26, 33), 'right': range(39, 46)},
    'thrust': {'back': range(52, 60), 'left': range(65, 73),
               'front': range(78, 86), 'right': range(91, 99)},
    'walk': {'back': range(104, 113), 'left': range(117, 126),
             'front': range(130, 139), 'right': range(143, 152)},
    'slash': {'back': range(156, 162), 'left': range(169, 175),
              'front': range(182, 188), 'right': range(195, 201)},
    'shoot': {'back': range(208, 221), 'left': range(221, 234),
              'front': range(234, 247), 'right': range(247, 260)},
    'hurt': {'front': range(260, 266)}
    }


    for j in range(int(N_height)):
        for i in range(int(N_width)):
            ind = j * N_width + i
            if ind in [26, 28, 29, 30, 80, 184, 187, 236,240]:
                box = (i*width, j*height, (i+1)*width, (j+1)*height)
                a = img.crop(box)
                a.convert('RGB')
                a = a.resize((112,112), Image.ANTIALIAS)
                a.save(path + '_%s_%d.png' % (classname, ind))


    # now remove the png files
    os.remove('%s.png' % classname)

if __name__ == '__main__':
    # Generate images with combinations of features
    a = [range(8), range(10), range(10), range(10)]
    seed_list = list(itertools.product(*a))
    for i, seed in enumerate(seed_list):
        body, bottom, top, hair = seed

        gen_char(body, bottom, top, hair)

    path = "dataSprites112/"
    dirs = os.listdir( path )
    dirs.sort()
    x_train=[]
    load_dataset()
    
    # Convert and save the list of images in '.npy' format
    imgset=np.array(x_train)
    np.save("imgs112_sprites.npy",imgset)
    
    # Create dataframe of features
    df_attribute = pd.DataFrame(columns=["Body", "Bottom", 'Top', "Hair", "Stance"])

    for item in dirs:
        item = item[:-4].replace('_','')
        keys = ["Body", "Bottom", 'Top', "Hair", "Stance"]
        values =  [item[i] if i <4 else item[i:] for i in range(5)]
        item_dict = {keys[i]: values[i] for i in range(len(keys))}
        df_attribute = pd.concat([df_attribute, pd.DataFrame(item_dict, index=[0])], ignore_index = True)
    
    df_attribute.to_csv('attribute_sprites.csv')
        
