from torch.utils.data import Dataset
import numpy as np
from numpy import *
import torch
import torchvision.transforms as transforms

class QuerySplit(Dataset):
  def __init__(self, df, df_complete, data, same_aspect, data_queries = None, transform = None, support_size = 4, img_size = 112):
    self.df = df
    self.df_complete = df_complete
    self.data = data
    self.data_queries = data_queries
    self.aspects = self.find_aspects()
    self.columns = list(self.df.columns)
    self.transform = transform
    self.num_same = len(same_aspect)
    self.support_size = support_size
    self.img_size = img_size
    self.same_aspect = same_aspect

  def __len__(self):
    return len(self.df)

  def find_aspects(self):
    df_without_image = self.df_complete.iloc[:, :]
    aspects = df_without_image.apply(lambda col: list(col.unique())).to_dict()
    return aspects

  def __getitem__(self, idx):
    anchor_aspects = list(self.df.values[:, :][idx])
    anchor = self.data_queries[idx]

    same_asp = list(random.choice(self.same_aspect, size=self.num_same, replace = False))
    anchor_aspects_dict = dict(zip(self.columns, anchor_aspects))
    different = [value for value in self.columns if value not in same_asp]

    for value in different:
      possible = [p for p in self.aspects[value] if p != anchor_aspects_dict[value]]
      anchor_aspects_dict[value] = random.choice(possible, 1)[0]

    my_vars = {}
    new = anchor_aspects_dict
    for i in range(self.num_same):
        support = "support%d" % i
        Y = "Y%d" % i
        my_vars[support], my_vars[Y], new = self.get_support(new, same_asp[i])
        my_vars[support] = self.index_to_tensor(my_vars[support])

    anc = self.read_img(anchor)

    return anc, my_vars

  def get_support(self, anchor_aspects_dict, same_anchor):
    place = random.choice(range(self.support_size), size = self.support_size-1, replace=False)

    possible_choices = [value for value in self.aspects[same_anchor] if value != anchor_aspects_dict[same_anchor]]
    changing_aspect = random.choice(possible_choices, size = self.support_size-1, replace = False)

    j = 0
    images = []

    for i in range(self.support_size):
      current = anchor_aspects_dict.copy()
      if i == place[0]:
        pass
      else:
        current[same_anchor] = changing_aspect[j]
        j += 1
        if i == place[1]:
          new = current

      params = list(current.values())
      images.append(self.find_image(params))

    Y = np.zeros(self.support_size)
    Y[place[0]] = 1.

    return images, Y, new

  def index_to_tensor(self, indexes):
    image_tensor = torch.zeros((self.support_size, 3, self.img_size, self.img_size))

    for i, index in enumerate(indexes):
      img = self.read_img(self.data[index])
      image_tensor[i] = img

    return image_tensor

  def find_image(self, params):
    copy_df = self.df_complete.copy()
    columns = list(copy_df.columns)
    for i, value in enumerate(params):
      column = columns[i]
      copy_df = copy_df[copy_df[column] == value]

    return copy_df.index[0]

  def read_img(self, image):
    if self.transform:
      image = self.transform(image)
    return image
    
class DataUniqueSplit(Dataset):
  def __init__(self, df, df_complete, data, same_aspect, data_queries = None, transform = None, train = True, support_size = 4, img_size = 112):
    self.df = df
    self.df_complete = df_complete
    self.data = data
    self.data_queries = data_queries
    self.aspects = self.find_aspects()
    self.columns = list(self.df.columns)
    self.transform = transform
    self.train = train
    self.num_same = len(same_aspect)
    self.support_size = support_size
    self.img_size = img_size
    self.same_aspect = same_aspect
    if not self.train:
      self.seeded_anchor, self.seeded_support_sets = self.seeded_generation()

  def __len__(self):
    return len(self.df)

  def find_aspects(self):
    df_without_image = self.df_complete.iloc[:, :]
    aspects = df_without_image.apply(lambda col: list(col.unique())).to_dict()
    return aspects

  def seeded_generation(self):
    np.random.seed(154)
    seeds = list(random.choice(range(len(self.df)*5), len(self.df), replace=False))

    seeded_anchor = []
    seeded_support_sets = {**{"seeded_support%d" % i: [] for i in range(self.num_same)},
                           **{"seeded_Y%d" % i: [] for i in range(self.num_same)}}

    for i, seed in enumerate(seeds):

      np.random.seed(seed)
      anchor_aspects = list(self.df.values[:, :][i])
      anchor = self.data_queries[i]

      same_asp = list(random.choice(self.same_aspect, size=self.num_same, replace = False))
      anchor_aspects_dict = dict(zip(self.columns, anchor_aspects))
      different = [value for value in self.columns if value not in same_asp]

      for value in different:
        possible = [p for p in self.aspects[value] if p != anchor_aspects_dict[value]]
        anchor_aspects_dict[value] = random.choice(possible, 1)[0]

      new = anchor_aspects_dict
      for i in range(self.num_same):

          seeded_support = "seeded_support%d" % i
          seeded_Y = "seeded_Y%d" % i
          sup, Y, new = self.get_support(new, same_asp[i])
          seeded_support_sets[seeded_support].append(sup)
          seeded_support_sets[seeded_Y].append(Y)

      seeded_anchor.append(anchor)

    return seeded_anchor, seeded_support_sets

  def __getitem__(self, idx):
    if self.train:
      anchor_aspects = list(self.df.values[:, :][idx])
      anchor = self.data[idx]

      same_asp = list(random.choice(self.same_aspect, size=self.num_same, replace = False))
      anchor_aspects_dict = dict(zip(self.columns, anchor_aspects))
      different = [value for value in self.columns if value not in same_asp]

      for value in different:
        possible = [p for p in self.aspects[value] if p != anchor_aspects_dict[value]]
        anchor_aspects_dict[value] = random.choice(possible, 1)[0]

      my_vars = {}
      new = anchor_aspects_dict
      for i in range(self.num_same):
          support = "support%d" % i
          Y = "Y%d" % i
          my_vars[support], my_vars[Y], new = self.get_support(new, same_asp[i])
          my_vars[support] = self.index_to_tensor(my_vars[support])

    else:
      anchor = self.seeded_anchor[idx]
      my_vars = {}
      for i in range(self.num_same):
          support = "support%d" % i
          Y = "Y%d" % i
          seeded_support = "seeded_support%d" % i
          seeded_Y = "seeded_Y%d" % i
          my_vars[support], my_vars[Y] = self.seeded_support_sets[seeded_support][idx], self.seeded_support_sets[seeded_Y][idx]
          my_vars[support] = self.index_to_tensor(my_vars[support])

    anc = self.read_img(anchor)

    return anc, my_vars

  def get_support(self, anchor_aspects_dict, same_anchor):
    place = random.choice(range(self.support_size), size = self.support_size-1, replace=False)

    possible_choices = [value for value in self.aspects[same_anchor] if value != anchor_aspects_dict[same_anchor]]
    changing_aspect = random.choice(possible_choices, size = self.support_size-1, replace = False)

    j = 0
    images = []

    for i in range(self.support_size):
      current = anchor_aspects_dict.copy()
      if i == place[0]:
        pass
      else:
        current[same_anchor] = changing_aspect[j]
        j += 1
        if i == place[1]:
          new = current

      params = list(current.values())
      images.append(self.find_image(params))

    Y = np.zeros(self.support_size)
    Y[place[0]] = 1.

    return images, Y, new

  def index_to_tensor(self, indexes):
    image_tensor = torch.zeros((self.support_size, 3, self.img_size, self.img_size))

    for i, index in enumerate(indexes):
      img = self.read_img(self.data[index])
      image_tensor[i] = img

    return image_tensor

  def find_image(self, params):
    copy_df = self.df_complete.copy()
    columns = list(copy_df.columns)
    for i, value in enumerate(params):
      column = columns[i]
      copy_df = copy_df[copy_df[column] == value]

    return copy_df.index[0]

  def read_img(self, image):
    if self.transform:
      image = self.transform(image)
    return image

class QuerySplitTest(Dataset):
  def __init__(self, df, df_complete, data, same_aspect, data_queries = None, transform = None, support_size = 4, img_size = 112):
    self.df = df
    self.df_complete = df_complete
    self.data = data
    self.data_queries = data_queries
    self.aspects = self.find_aspects()
    self.columns = list(self.df.columns)
    self.transform = transform
    self.num_same = len(same_aspect)
    self.support_size = support_size
    self.img_size = img_size
    self.same_aspect = same_aspect

  def __len__(self):
    return len(self.df)

  def find_aspects(self):
    df_without_image = self.df_complete.iloc[:, :]
    aspects = df_without_image.apply(lambda col: list(col.unique())).to_dict()
    return aspects

  def __getitem__(self, idx):
    anchor_aspects = list(self.df.values[:, :][idx])
    anchor = self.data_queries[idx]

    same_asp = list(random.choice(self.same_aspect, size=self.num_same, replace = False))
    anchor_aspects_dict = dict(zip(self.columns, anchor_aspects))
    different = [value for value in self.columns if value not in same_asp]

    for value in different:
      possible = [p for p in self.aspects[value] if p != anchor_aspects_dict[value]]
      anchor_aspects_dict[value] = random.choice(possible, 1)[0]

    my_vars = {}
    new = anchor_aspects_dict
    for i in range(self.num_same):
        support = "support%d" % i
        Y = "Y%d" % i
        my_vars[support], my_vars[Y], new = self.get_support(new, same_asp[i])
        my_vars[support] = self.index_to_tensor(my_vars[support])

    anc = self.read_img(anchor)

    return anc, my_vars, same_asp

  def get_support(self, anchor_aspects_dict, same_anchor):
    place = random.choice(range(self.support_size), size = self.support_size-1, replace=False)

    possible_choices = [value for value in self.aspects[same_anchor] if value != anchor_aspects_dict[same_anchor]]
    changing_aspect = random.choice(possible_choices, size = self.support_size-1, replace = False)

    j = 0
    images = []

    for i in range(self.support_size):
      current = anchor_aspects_dict.copy()
      if i == place[0]:
        pass
      else:
        current[same_anchor] = changing_aspect[j]
        j += 1
        if i == place[1]:
          new = current

      params = list(current.values())
      images.append(self.find_image(params))

    Y = np.zeros(self.support_size)
    Y[place[0]] = 1.

    return images, Y, new

  def index_to_tensor(self, indexes):
    image_tensor = torch.zeros((self.support_size, 3, self.img_size, self.img_size))

    for i, index in enumerate(indexes):
      img = self.read_img(self.data[index])
      image_tensor[i] = img

    return image_tensor

  def find_image(self, params):
    copy_df = self.df_complete.copy()
    columns = list(copy_df.columns)
    for i, value in enumerate(params):
      column = columns[i]
      copy_df = copy_df[copy_df[column] == value]

    return copy_df.index[0]

  def read_img(self, image):
    if self.transform:
      image = self.transform(image)
    return image