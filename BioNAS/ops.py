import torch.nn as nn

# NxN conv -> IN -> Relu
class ConvNxN_IN_RELU(nn.Module):

  def __init__(self, C_in, C_out, kernel_size):
    super(ConvNxN_IN_RELU, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding='same'),
      nn.InstanceNorm2d(C_out, affine=True),
      nn.ReLU(inplace=False),
      )

  def forward(self, x):
    return self.op(x)

# NxN conv -> IN -> MISH
class ConvNxN_IN_MISH(nn.Module):

  def __init__(self, C_in, C_out, kernel_size):
    super(ConvNxN_IN_MISH, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding='same'),
      nn.InstanceNorm2d(C_out, affine=True),
      nn.Mish()
      )

  def forward(self, x):
    return self.op(x)

# IN -> MISH -> NxN conv
class IN_MISH_NxNConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size):
    super(IN_MISH_NxNConv, self).__init__()
    self.op = nn.Sequential(
      nn.InstanceNorm2d(C_out, affine=True),
      nn.Mish(),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding='same'),
      )

  def forward(self, x):
    return self.op(x)

# RELU -> NxN conv
class RELU_NxNConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size):
    super(RELU_NxNConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding='same'),
      )

  def forward(self, x):
    return self.op(x)

# MISH -> NxN conv
class MISH_NxNConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size):
    super(MISH_NxNConv, self).__init__()
    self.op = nn.Sequential(
      nn.Mish(),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding='same'),
      )

  def forward(self, x):
    return self.op(x)


OPS = {
    'Conv3x3_IN_RELU' : lambda in_c, out_c: ConvNxN_IN_RELU(in_c, out_c, 3),
    'Conv5x5_IN_RELU' : lambda in_c, out_c: ConvNxN_IN_RELU(in_c, out_c, 5),
    'Conv7x7_IN_RELU' : lambda in_c, out_c: ConvNxN_IN_RELU(in_c, out_c, 7),
    'Conv3x3_IN_MISH' : lambda in_c, out_c: ConvNxN_IN_MISH(in_c, out_c, 3),
    'Conv5x5_IN_MISH' : lambda in_c, out_c: ConvNxN_IN_MISH(in_c, out_c, 5),
    'Conv7x7_IN_MISH' : lambda in_c, out_c: ConvNxN_IN_MISH(in_c, out_c, 7),
    'IN_MISH_3x3Conv' : lambda in_c, out_c: IN_MISH_NxNConv(in_c, out_c, 3),
    'IN_MISH_5x5Conv' : lambda in_c, out_c: IN_MISH_NxNConv(in_c, out_c, 5),
    'IN_MISH_7x7Conv' : lambda in_c, out_c: IN_MISH_NxNConv(in_c, out_c, 7),
    'RELU_3x3Conv' : lambda in_c, out_c: RELU_NxNConv(in_c, out_c, 3),
    'RELU_5x5Conv' : lambda in_c, out_c: RELU_NxNConv(in_c, out_c, 5),
    'RELU_7x7Conv' : lambda in_c, out_c: RELU_NxNConv(in_c, out_c, 7),
    'MISH_3x3Conv' : lambda in_c, out_c: MISH_NxNConv(in_c, out_c, 3),
    'MISH_5x5Conv' : lambda in_c, out_c: MISH_NxNConv(in_c, out_c, 5),
    'MISH_7x7Conv' : lambda in_c, out_c: MISH_NxNConv(in_c, out_c, 7),
}

OPS_Keys = ['Conv3x3_IN_RELU', 'Conv5x5_IN_RELU', 'Conv7x7_IN_RELU', 'Conv3x3_IN_MISH', 'Conv5x5_IN_MISH', 'Conv7x7_IN_MISH', 'IN_MISH_3x3Conv', 'IN_MISH_5x5Conv', 'IN_MISH_7x7Conv', 
'RELU_3x3Conv', 'RELU_5x5Conv', 'RELU_7x7Conv', 'MISH_3x3Conv', 'MISH_5x5Conv', 'MISH_7x7Conv']