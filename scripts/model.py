import torch.nn as nn
import torch.nn.functional as F
import torch
import einops

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,4,kernel_size=3,padding=1),
                        nn.BatchNorm2d(4, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(4,8,kernel_size=3,padding=1),
                        nn.BatchNorm2d(8, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(8,16,kernel_size=3,padding=1),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self,x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        return out
        
class VGG(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(VGG, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,4,kernel_size=3,padding=1),
                        nn.BatchNorm2d(4, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(4,4,kernel_size=3,padding=1),
                        nn.BatchNorm2d(4, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(4,8,kernel_size=3,padding=1),
                        nn.BatchNorm2d(8, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(8,8,kernel_size=3,padding=1),
                        nn.BatchNorm2d(8, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(8,16,kernel_size=3,padding=1),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, last = nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.last = last
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.last(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10, last = nn.ReLU()):
        super(ResNet, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 8, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(8),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 8, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 16, layers[1], stride = 2, last=last)

    def _make_layer(self, block, planes, blocks, stride=1, last = nn.ReLU()):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        if blocks == 1:
            layers.append(block(self.inplanes, planes, stride, downsample, last=last))
            self.inplanes = planes
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes
            for i in range(1, blocks):
                if i == blocks-1:
                    layers.append(block(self.inplanes, planes, last=last))
                else:
                    layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)


        return x
        
class ContextIdentity(nn.Module):
  def __init__(self, *args, **kwargs):
    super(ContextIdentity, self).__init__()
    pass

  def forward(self, anchor, support):
    return anchor, support

class AspectBasedModel(nn.Module):
  def __init__(self, represent_model, support_size, fsl_params ={}, context_params ={}, context_model = ContextIdentity, fsl_model = ContextIdentity, img_size = 112,):
    super(AspectBasedModel, self).__init__()
    self.represent_model = represent_model
    self.img_size = img_size
    self.support_size = support_size

    x = torch.randn(1, 3, img_size, img_size)
    x = self.represent_model(x)

    self.context_model = context_model(x.shape, support_size=support_size, **context_params)
    support = einops.repeat(x, 'm n k l -> m s n k l', s=support_size)

    x, support = self.context_model(x, support)
    self.fsl = fsl_model(x.shape, **fsl_params)

  def forward(self, anchor, support_set):
    batch_size, support_size, C, H, W = support_set.shape
    anchor_x = self.represent_model(anchor)

    support_x = self.represent_model(support_set.view(-1,C,H,W))
    C, H, W = support_x.shape[1:]
    support_x = support_x.view(batch_size, support_size, C,H,W)

    anchor_x, support_x = self.context_model(anchor_x, support_x)

    if type(self.fsl) == ContextIdentity:
      return anchor_x, support_x

    else:
      output = torch.zeros((batch_size, support_size), device = self.device)

      for i in range(support_size):
        output[:, i:i+1] = self.fsl(anchor_x, support_x[:,i])

      output = F.softmax(output, dim = 1)
      return output



  @property
  def device(self):
    return next(self.parameters()).device
    
class DeepSetTraversalModule(nn.Module):
  def __init__(self, shape, support_size):
    super(DeepSetTraversalModule, self).__init__()

    self.support_size = support_size

    self.concentrator = nn.Sequential(
                          nn.Conv2d(16,16,kernel_size=3,padding=1),
                          nn.BatchNorm2d(16, momentum=1, affine=True),
                          nn.ReLU(),
    )

    self.perm_equi = nn.Sequential(
                        nn.Conv2d(32, 16, kernel_size=1, padding=0),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),

    )

    self.projector = nn.Sequential(
                        nn.Conv2d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.Softmax(dim=1),
                        nn.MaxPool2d(2)
    )

    self.reshaper = nn.Sequential(
                        nn.Conv2d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
    )

  def forward(self, anchor, support):
    anchor_x = self.reshaper(anchor)

    batch_size, support_size, C, H, W = support.shape
    support_x_reshape = self.reshaper(support.view(-1,C,H,W))
    C, H, W = support_x_reshape.shape[1:]
    support_x_reshape = support_x_reshape.view(batch_size, support_size, C,H,W)

    batch_size, support_size, C, H, W = support.shape
    support_x = self.concentrator(support.view(-1,C,H,W))
    C, H, W = support_x.shape[1:]
    support_x = support_x.view(batch_size, support_size, C,H,W)

    support_x_deep_set = torch.zeros(*support.shape, device = self.device)

    for i in range(support_size):
      mask = torch.ones(support_size, dtype=bool)
      mask[i] = False
      sum_tensor = support_x[:, mask].sum(dim=1)
      concatenation = torch.concat([support[:,i], sum_tensor], dim=1)
      support_x_deep_set[:, i:i+1] = self.perm_equi(concatenation)[:,None,:,:,:]

    support_x_deep_set = torch.mean(support_x_deep_set, dim=1)
    projector = self.projector(support_x_deep_set)

    anchor_x = torch.mul(anchor_x, projector)
    support_x_reshape_m = torch.zeros(*support_x_reshape.shape, device=self.device)

    for i in range(support_size):
      support_x_reshape_m[:, i:i+1] = torch.mul(support_x_reshape[:, i], projector)[:, None, :, :, :]

    return anchor_x, support_x_reshape_m

  @property
  def device(self):
    return next(self.parameters()).device

class DeepSetTraversalModuleResnet(nn.Module):
  def __init__(self, shape, support_size):
    super(DeepSetTraversalModuleResnet, self).__init__()

    self.support_size = support_size
    self.inplanes = 16
    self.concentrator = self._make_layer(ResidualBlock, 16, 1, stride = 1)

    self.perm_equi = nn.Sequential(
                        nn.Conv2d(32, 16, kernel_size=1, padding=0),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),
    )

    self.projector = self._make_layer(ResidualBlock, 16, 1, stride = 2, last=nn.Softmax())

    self.reshaper = nn.Sequential(
                        nn.Conv2d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm2d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
    )

  def forward(self, anchor, support):
    anchor_x = self.reshaper(anchor)

    batch_size, support_size, C, H, W = support.shape
    support_x_reshape = self.reshaper(support.view(-1,C,H,W))
    C, H, W = support_x_reshape.shape[1:]
    support_x_reshape = support_x_reshape.view(batch_size, support_size, C,H,W)

    batch_size, support_size, C, H, W = support.shape
    support_x = self.concentrator(support.view(-1,C,H,W))
    C, H, W = support_x.shape[1:]
    support_x = support_x.view(batch_size, support_size, C,H,W)

    support_x_deep_set = torch.zeros(*support.shape, device = self.device)

    for i in range(support_size):
      mask = torch.ones(support_size, dtype=bool)
      mask[i] = False
      sum_tensor = support_x[:, mask].sum(dim=1)
      concatenation = torch.concat([support[:,i], sum_tensor], dim=1)
      support_x_deep_set[:, i:i+1] = self.perm_equi(concatenation)[:,None,:,:,:]

    support_x_deep_set = torch.mean(support_x_deep_set, dim=1)
    projector = self.projector(support_x_deep_set)

    anchor_x = torch.mul(anchor_x, projector)
    support_x_reshape_m = torch.zeros(*support_x_reshape.shape, device=self.device)

    for i in range(support_size):
      support_x_reshape_m[:, i:i+1] = torch.mul(support_x_reshape[:, i], projector)[:, None, :, :, :]

    return anchor_x, support_x_reshape_m

  @property
  def device(self):
    return next(self.parameters()).device

  def _make_layer(self, block, planes, blocks, stride=1, last = nn.ReLU()):
    downsample = None
    if stride != 1 or self.inplanes != planes:

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
            nn.BatchNorm2d(planes),
        )
    layers = []
    if blocks == 1:
        layers.append(block(self.inplanes, planes, stride, downsample, last=last))
        self.inplanes = planes
    else:
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            if i == blocks-1:
                layers.append(block(self.inplanes, planes, last=last))
            else:
                layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)