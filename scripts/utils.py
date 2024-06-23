import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import torch
import torch.nn as nn
import time
import math

def load_data(dataset):
  if dataset == 'Sprites':
      images = np.load('data/Sprites/imgs112_sprites.npy')
      df_attributes = pd.read_csv('data/Sprites/attribute_sprites.csv', index_col=0)
      support_size = 4
      batch_size_train = 100
      images_train, images_val, images_test, attributes_train, attributes_val, attributes_test = split_data(dataset, df_attributes, images)
      same_aspect = ["Bottom", 'Top', "Hair", "Stance"]
  elif dataset == 'Shapes':
      images = np.load('data/Shapes/imgs112_shapes.npy')
      df_attributes = pd.read_csv('data/Shapes/attribute_shapes.csv')
      df_attributes = df_attributes.drop('image', axis=1)
      df_attributes.fillna('No', inplace=True)
      support_size = 3
      batch_size_train = 32
      images_train, images_val, images_test, attributes_train, attributes_val, attributes_test = split_data(dataset, df_attributes, images)
      same_aspect = ["Thickness", "Color", 'Pattern']
  else:
      raise ValueError("No dataset found")
  return (images, df_attributes, support_size, images_train, images_val, images_test, 
          attributes_train, attributes_val, attributes_test, same_aspect, batch_size_train)
  
def split_data(dataset, attributes, images):
    np.random.seed(879)
    
    if dataset == 'Sprites':
        aspects = attributes.copy().apply(lambda col: list(col.unique())).to_dict()
        train_test_splits = {key: np.random.choice(value, 6, replace=False) if key == 'Stance' else np.random.choice(value, 6, replace=False) for key, value in aspects.items()}
        del train_test_splits['Body']

        attributes_train = attributes.copy()
        attributes_test = attributes.copy()

        for key in train_test_splits.keys():
            attributes_train = attributes_train[attributes_train[key].isin(train_test_splits[key])]
            attributes_test = attributes_test[~attributes_test[key].isin(train_test_splits[key])]

        train_val_split = {(key): (np.random.choice(value, 5, replace=False))  if key == 'Stance' else np.random.choice(value, 5, replace=False) for key, value in train_test_splits.items()}
        attributes_val = attributes_train.copy()

        for key in train_val_split.keys():
            attributes_train = attributes_train[attributes_train[key].isin(train_val_split[key])]
            attributes_val = attributes_val[~attributes_val[key].isin(train_val_split[key])]

        images_train = images[attributes_train.index]
        images_val = images[attributes_val.index]
        images_test = images[attributes_test.index]

        attributes_train, attributes_val, attributes_test = attributes_train.reset_index(drop=True), attributes_val.reset_index(drop=True), attributes_test.reset_index(drop=True)

    elif dataset =='Shapes':
        aspects = attributes.copy().apply(lambda col: list(col.unique())).to_dict()
        train_test_splits = {key: np.random.choice(value, 5, replace=False) for key, value in aspects.items()}
        del train_test_splits['Shape']

        attributes_train = attributes.copy()
        attributes_test = attributes.copy()

        for key in train_test_splits.keys():
          attributes_train = attributes_train[attributes_train[key].isin(train_test_splits[key])]
          attributes_test = attributes_test[~attributes_test[key].isin(train_test_splits[key])]

        train_val_split = {(key): (np.random.choice(value, 4, replace=False))  for key, value in train_test_splits.items()}
        attributes_val = attributes_train.copy()

        for key in train_val_split.keys():
          attributes_train = attributes_train[attributes_train[key].isin(train_val_split[key])]
          attributes_val = attributes_val[~attributes_val[key].isin(train_val_split[key])]

        images_train = images[attributes_train.index]
        images_val = images[attributes_val.index]
        images_test = images[attributes_test.index]

        attributes_train, attributes_val, attributes_test = attributes_train.reset_index(drop=True), attributes_val.reset_index(drop=True), attributes_test.reset_index(drop=True)
 
    return images_train, images_val, images_test, attributes_train, attributes_val, attributes_test
    
def visualize_support_set(loader, num_sets, support_size):
    anchor, support_sets, = next(iter(loader))[:2]
    for j in range(num_sets):
        columns = support_size+1
        rows = 1
        for k in range(support_size):
          fig = plt.figure(figsize=(columns*rows*3, columns*rows*3))
          img = np.transpose(anchor[j], (1, 2, 0))
          ax = fig.add_subplot(rows, columns, 1)
          plt.imshow(img)
          support = "support%d" % k
          Y = "Y%d" % k

          for axis in ['top', 'bottom', 'left', 'right']:
              ax.spines[axis].set_linewidth(5)  # change width
              ax.spines[axis].set_color('red')    # change color

          for i in range(2, columns*rows+1):
              img = np.transpose(support_sets[support][j][i-2],(1,2,0))
              ax = fig.add_subplot(rows, columns, i)
              if support_sets[Y][j][i-2] == 1:
                  for axis in ['top', 'bottom', 'left', 'right']:

                      ax.spines[axis].set_linewidth(5)  # change width
                      ax.spines[axis].set_color('red')    # change color

              plt.imshow(img)

        plt.show()

class AspectBasedTupletLoss(nn.Module):
    def __init__(self, support_size, device):
        super(AspectBasedTupletLoss, self).__init__()
        self.support_size = support_size
        self.device = device

    def forward(self, anchor, support, targets):
        batch_size = anchor.shape[0]
        pdist = nn.PairwiseDistance(p=2)
        dist_mat = torch.zeros(batch_size, support.shape[1], device=self.device)

        for i in range(support.shape[1]):
          dist_mat[:,i] = pdist(anchor, support[:,i])


        pos = dist_mat[targets == 1].reshape(batch_size, 1)
        neg = dist_mat[targets == 0].reshape(batch_size, support.shape[1]-1)

        loss = torch.log(1 + torch.exp(pos - neg).sum(dim=1, keepdim=True))
        return loss.mean()
        

def train(model, train_loader, val_loader, num_epochs, optimizer, criterion, model_name, device, support_size):
    train_losses = []
    val_losses = []
    cur_step = 0
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0

        i = 1
        model.train()
        print("Starting epoch " + str(epoch+1))
        for anchor, support_sets in train_loader:
            # Forward
            batch_size_tr = anchor.shape[0]
            anchor = anchor.to(device)
            loss = 0.0
            losses =[]
            for i in range(support_size):
                support = "support%d" % i
                Y = "Y%d" % i
                support, labels = support_sets[support].to(device), support_sets[Y].to(device)
                anchor_x, support_x = model(anchor, support)
                anchor_x, support_x = anchor_x.view(batch_size_tr, -1), support_x.view(batch_size_tr, support_size, -1)
                loss_support = criterion(anchor_x, support_x, labels)
                loss += loss_support
                losses.append(loss_support)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            i += 1

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_running_loss = 0.0
        with torch.no_grad():
            model.eval()
            for anchor, support_sets in val_loader:
                # Forward
                batch_size_tr = anchor.shape[0]
                anchor = anchor.to(device)
                loss = 0.0
                losses =[]
                for i in range(support_size):
                    support = "support%d" % i
                    Y = "Y%d" % i
                    support, labels = support_sets[support].to(device), support_sets[Y].to(device)
                    anchor_x, support_x = model(anchor, support)
                    anchor_x, support_x = anchor_x.view(batch_size_tr, -1), support_x.view(batch_size_tr, support_size, -1)
                    loss_support = criterion(anchor_x, support_x, labels)
                    loss += loss_support
                    losses.append(loss_support)

                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
          if best_val_loss != np.inf:
            print('Improved by ', best_val_loss-avg_val_loss)
          torch.save(model.state_dict(), model_name)
          best_val_loss = avg_val_loss

        time_taken = time.time() - start

        print('Epoch [{}/{}], Train Loss: {:.4f}, val Loss: {:.4f}, Time: {:.4f}'
            .format(epoch+1, num_epochs, avg_train_loss, avg_val_loss, time_taken))

    print("Finished Training")
    return train_losses
    
def distance(anchor, support, support_size, device):
  batch_size = anchor.shape[0]
  pdist = nn.PairwiseDistance(p=2)
  dist_mat = torch.zeros(batch_size, support_size, device=device)

  for i in range(support.shape[1]):
    dist_mat[:,i] = pdist(anchor, support[:,i])
  return dist_mat
  
def interval_95(df):
      ci95_hi = []
      ci95_lo = []
      plus_minus = []
    
      for i in df.index:
          m, c, s = df.loc[i]
          ci95_hi.append(m + 1.96*s/math.sqrt(c))
          ci95_lo.append(m - 1.96*s/math.sqrt(c))
          plus_minus.append(1.96*s/math.sqrt(c))
    
      df['plus_minus'] = plus_minus
      df['ci95_hi'] = ci95_hi
      df['ci95_lo'] = ci95_lo
    
      return(df)
  
def predict_distance(loader, num_runs, device, models, model_names, support_size):
    df_dist = pd.DataFrame()
    for i in range(num_runs):
        for anchor, support_sets, same_asp in loader:
            batch_size_tr = anchor.shape[0]
            anchor = anchor.to(device)

            for model_name, modelz in zip(model_names, models):
                new = pd.DataFrame()
                for j in range(support_size):
                    k=j+1
                    support = "support%d" % j
                    Y = "Y%d" % j
                    distancePD = 'distance%d' % k
                    support, labels = support_sets[support].to(device), support_sets[Y].to(device)
                    anchor_x, support_x = modelz(anchor, support)
                    anchor_x, support_x = anchor_x.view(batch_size_tr, -1), support_x.view(batch_size_tr, support_size, -1)

                    new[distancePD] = distance(anchor_x, support_x, support_size, device).cpu().detach().numpy().flatten()
                    Y = "Y%d" % k
                    new[Y] = labels.flatten()

                if model_name[-5:] == 'class':
                    new['split'] = 'class'
                    new['model'] = model_name[:-6]
                else:
                    new['split'] = 'query'
                    new['model'] = model_name

                for j in range(support_size):
                    k = j+1
                    aspect = 'aspect%d' % k
                    new[aspect] = np.repeat(same_asp[j], support_size)
                   
                new['run'] = i

                df_dist = pd.concat([df_dist, new])
    return df_dist
