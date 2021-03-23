# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

import torch.nn.functional as F
import torch.nn as nn
import torch as pt
import numpy as np
import math

class VanillaMLP(nn.Module):
  def __init__(self, mlp, hidden, pos_level, depth_level, lrelu_slope, out_node, first_gpu = 0):
    super().__init__()
    self.activation = nn.LeakyReLU(lrelu_slope)

    ls = [nn.Linear(pos_level * 2 * 2 + depth_level * 2, hidden)]
    ls.append(self.activation)
    for i in range(mlp):
      ls.append(nn.Linear(hidden, hidden))
      ls.append(self.activation)
    ls.append(nn.Linear(hidden, out_node))
    self.seq1 = nn.Sequential(*ls)

  def forward(self, x):
    return self.seq1(x)

class ReluMLP(nn.Module):
  def __init__(self, mlp, hidden_dim, in_node=3, lrelu_slope = 0.01, out_node=1, first_gpu = 0):
    super().__init__()
    self.activation = nn.LeakyReLU(lrelu_slope)
    ls = [nn.Linear(in_node, hidden_dim)]
    ls.append(self.activation)
    for i in range(mlp):
      ls.append(nn.Linear(hidden_dim, hidden_dim))
      ls.append(self.activation)

    ls.append(nn.Linear(hidden_dim, out_node))
    self.seq1 = nn.Sequential(*ls).cuda('cuda:{}'.format(first_gpu))

  def forward(self, x):
    return self.seq1(x)