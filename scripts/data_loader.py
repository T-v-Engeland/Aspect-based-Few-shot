from torch.utils.data import Dataset
import numpy as np
from numpy import *
import torch
import torchvision.transforms as transforms
import pandas as pd

class QuerySplit(Dataset):
  '''
  This class allows for the query split of the data
  '''
  
  def __init__(self, df: pd.DataFrame, df_complete: pd.DataFrame, data: np.ndarray, same_aspect: list, 
               data_queries: np.ndarray = None, transform: transforms.Compose = None, support_size: int = 4, 
               img_size: int = 112):
    self.df = df
    self.df_complete = df_complete
    self.data = data
    self.data_queries = data_queries

    # Create a dictionary of each aspect presented in the data
    self.aspects = self.find_aspects()
    
    self.columns = list(self.df.columns)
    self.transform = transform
    self.num_same = len(same_aspect)
    self.support_size = support_size
    self.img_size = img_size
    self.same_aspect = same_aspect

  def __len__(self):
    return len(self.df)

  def find_aspects(self) -> dict:
    '''
    This function extracts the unique aspects presented in the data

    Returns:
        dict: dictionary with the unique aspects for each feature of the data
    '''
    
    df_without_image = self.df_complete.iloc[:, :]

    # Create the dictionary of each features with unique values
    aspects = df_without_image.apply(lambda col: list(col.unique())).to_dict()
    return aspects

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
    '''
    This function returns a query image with the respective support sets

    Args:
        idx (int): An index where 0 <= idx < len(self)

    Returns:
        torch.Tensor: The query image
        dict: A dictionary containing the support sets with the information of the correct match
    '''

    # Find the features of query
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

  def get_support(self, anchor_aspects_dict: dict, same_anchor: str) -> tuple[list, np.ndarray, dict]:
    '''
    This function finds a random support set

    Args:
        anchor_aspects_dict (dict): The features of the support set image that matches the query
        same_anchor (str): The aspect of the support set

    Returns:
        list: The indexes in the data of the support set images
        np.ndarray: The index of the matching image
        dict: The features of the next support set image that matches the query
    '''
    
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

  def index_to_tensor(self, indexes: list) -> torch.Tensor:
    '''
    This function finds the corresponding images based on the index

    Args:
        indexes (list): A list of the indexes corresponding with images

    Returns:
        torch.Tensor: A Tensor containing the images of the support set
    '''
    
    image_tensor = torch.zeros((self.support_size, 3, self.img_size, self.img_size))

    for i, index in enumerate(indexes):
      img = self.read_img(self.data[index])
      image_tensor[i] = img

    return image_tensor

  def find_image(self, params: list) -> int:
    '''
    This function finds the index based on the values of the features

    Args:
        params (list): A list with the feature values

    Returns:
        int: The index
    '''
    
    copy_df = self.df_complete.copy()
    columns = list(copy_df.columns)
    for i, value in enumerate(params):
      column = columns[i]
      copy_df = copy_df[copy_df[column] == value]

    return copy_df.index[0]

  def read_img(self, image: np.ndarray) -> torch.Tensor:
    '''
    This function transforms the numpy array to a torch Tensor

    Args:
        image (np.ndarray): A numpy array representing the image

    Returns:
        torch.Tensor: A torch Tensor representing the image
    '''
   
    if self.transform:
      image = self.transform(image)
    return image
    
class DataUniqueSplit(Dataset):
  '''
  This class allows for the data unique split of the data
  '''
  
  def __init__(self, df: pd.DataFrame, df_complete: pd.DataFrame, data: np.ndarray, same_aspect: list, 
               data_queries: np.ndarray = None, transform: transforms.Compose = None, train: bool = True,
               support_size: int = 4, img_size: int = 112):
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

  def find_aspects(self) -> dict:
    '''
    This function extracts the unique aspects presented in the data

    Returns:
        dict: dictionary with the unique aspects for each feature of the data
    '''
    
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

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
    '''
    This function returns a query image with the respective support sets

    Args:
        idx (int): An index where 0 <= idx < len(self)

    Returns:
        torch.Tensor: The query image
        dict: A dictionary containing the support sets with the information of the correct match
    '''
    
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

  def get_support(self, anchor_aspects_dict: dict, same_anchor: str) -> tuple[list, np.ndarray, dict]:
    '''
    This function finds a random support set

    Args:
        anchor_aspects_dict (dict): The features of the support set image that matches the query
        same_anchor (str): The aspect of the support set

    Returns:
        list: The indexes in the data of the support set images
        np.ndarray: The index of the matching image
        dict: The features of the next support set image that matches the query
    '''
        
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

  def index_to_tensor(self, indexes: list) -> torch.Tensor:
    '''
    This function finds the corresponding images based on the index

    Args:
        indexes (list): A list of the indexes corresponding with images

    Returns:
        torch.Tensor: A Tensor containing the images of the support set
    '''
    
    image_tensor = torch.zeros((self.support_size, 3, self.img_size, self.img_size))

    for i, index in enumerate(indexes):
      img = self.read_img(self.data[index])
      image_tensor[i] = img

    return image_tensor

  def find_image(self, params: list) -> int:
    '''
    This function finds the index based on the values of the features

    Args:
        params (list): A list with the feature values

    Returns:
        int: The index
    '''
    
    copy_df = self.df_complete.copy()
    columns = list(copy_df.columns)
    for i, value in enumerate(params):
      column = columns[i]
      copy_df = copy_df[copy_df[column] == value]

    return copy_df.index[0]

  def read_img(self, image: np.ndarray) -> torch.Tensor:
    '''
    This function transforms the numpy array to a torch Tensor

    Args:
        image (np.ndarray): A numpy array representing the image

    Returns:
        torch.Tensor: A torch Tensor representing the image
    '''
    
    if self.transform:
      image = self.transform(image)
    return image

class QuerySplitTest(Dataset):
  '''
  This class allows for the query split of the test data
  '''
  def __init__(self, df: pd.DataFrame, df_complete: pd.DataFrame, data: np.ndarray, same_aspect: list, 
               data_queries: np.ndarray = None, transform: transforms.Compose = None, support_size: int = 4, 
               img_size: int = 112):
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

  def find_aspects(self) -> dict:
    '''
    This function extracts the unique aspects presented in the data

    Returns:
        dict: dictionary with the unique aspects for each feature of the data
    '''
    
    df_without_image = self.df_complete.iloc[:, :]
    aspects = df_without_image.apply(lambda col: list(col.unique())).to_dict()
    return aspects

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict, list]:
    '''
    This function returns a query image with the respective support sets

    Args:
        idx (int): An index where 0 <= idx < len(self)

    Returns:
        torch.Tensor: The query image
        dict: A dictionary containing the support sets with the information of the correct match
        list: The aspects for each support set
    '''
    
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

  def get_support(self, anchor_aspects_dict: dict, same_anchor: str) -> tuple[list, np.ndarray, dict]:
    '''
    This function finds a random support set

    Args:
        anchor_aspects_dict (dict): The features of the support set image that matches the query
        same_anchor (str): The aspect of the support set

    Returns:
        list: The indexes in the data of the support set images
        np.ndarray: The index of the matching image
        dict: The features of the next support set image that matches the query
    '''
    
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

  def index_to_tensor(self, indexes: list) -> torch.Tensor:
    '''
    This function finds the corresponding images based on the index

    Args:
        indexes (list): A list of the indexes corresponding with images

    Returns:
        torch.Tensor: A Tensor containing the images of the support set
    '''
    
    image_tensor = torch.zeros((self.support_size, 3, self.img_size, self.img_size))

    for i, index in enumerate(indexes):
      img = self.read_img(self.data[index])
      image_tensor[i] = img

    return image_tensor

  def find_image(self, params: list) -> int:
    '''
    This function finds the index based on the values of the features

    Args:
        params (list): A list with the feature values

    Returns:
        int: The index
    '''
    
    copy_df = self.df_complete.copy()
    columns = list(copy_df.columns)
    for i, value in enumerate(params):
      column = columns[i]
      copy_df = copy_df[copy_df[column] == value]

    return copy_df.index[0]

  def read_img(self, image: np.ndarray) -> torch.Tensor:
    '''
    This function transforms the numpy array to a torch Tensor

    Args:
        image (np.ndarray): A numpy array representing the image

    Returns:
        torch.Tensor: A torch Tensor representing the image
    '''
   
    if self.transform:
      image = self.transform(image)
    return image
