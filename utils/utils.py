# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
from skimage import io
import re
import time
import traceback
import sys
import json
from torch.utils.data import SubsetRandomSampler,DataLoader
import torch as pt
import http.server
import socketserver
import webbrowser
from threading import Timer

def is_deepview(dpath):
  # check if dataset is deepview dataset
  return os.path.exists(dpath + "/models.json")

def is_llff(dpath):
  # check if dataset is LLFF dataset
  return os.path.exists(dpath + '/poses_bounds.npy')

def getDatasetScale(dpath, deepview_width, llff_width):
  if is_deepview(dpath):
      with open(os.path.join(dpath, "models.json"), "r") as fi:
        js = json.load(fi)
      return float(deepview_width / js[0][0]['width'])
  elif is_llff(dpath):
    img0 = [os.path.join(dpath, 'images', f) for f in sorted(os.listdir(os.path.join(dpath, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = io.imread(img0).shape
    return float(llff_width / sh[1])
  else:
    return 1

def generateSubsetSamplers(dataset_size, ratio=0.8, random_seed=0):
  indices = list(range(dataset_size))
  np.random.seed(random_seed)
  np.random.shuffle(indices)

  split = int(np.round(ratio * dataset_size))
  train_indices, val_indices = indices[:split], indices[split:]

  return SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

def prepareDataloaders(dataset, dpath, random_split=False, train_ratio=1, num_workers=8):
  if random_split:
    sampler_train, sampler_val = generateSubsetSamplers(len(dataset), ratio=train_ratio, num_workers = num_workers)
    dataloader_train = DataLoader(dataset, batch_size=1, sampler=sampler_train)
    dataloader_val = DataLoader(dataset, batch_size=1, sampler=sampler_val)
    print('TRAINING IMAGES: {}'.format(len(dataloader_train)))
    print('VALIDATE IMAGES: {}'.format(len(dataloader_val)))
  else:
    def get_indices(ty):
      if os.path.exists(dpath + "/{}_image.txt".format(ty)):
        data = []
        with open(dpath + "/{}_image.txt".format(ty), "r") as fi:
          for line in fi.readlines():
            count = 0
            for img in dataset.imgs:
              if line.strip() in img['path']:
                data.append(count)
                break
              count += 1
        return data
      else:
        raise('No CONFIG TRAINING FILE')
    def clean_path(list_of_path):
      return list(map(lambda x: str(os.path.basename(x)).lower(), list_of_path))

    if os.path.exists(os.path.join(dpath,'poses_bounds.npy')):
      #LLFF dataset which is use every 8 images to be training data
      indices_total = list(range(len(dataset.imgs)))
      indices_val = indices_total[::8]
      indices_train = list(filter(lambda x: x not in indices_val, indices_total))
    elif os.path.exists(os.path.join(dpath,'transforms_train.json')):
      indices_train = dataset.sfm.index_split[0]
      indices_val = dataset.sfm.index_split[1]
    else:
      indices_train = get_indices('train')
      indices_val = get_indices('val')
      # save indices to sfm for render propose
      dataset.sfm.index_split = [
        indices_train,
        indices_val
      ]

    # set cam rotation and translation for kalantari
    ref_camtxt = os.path.join(dpath,'ref_cameramatrix.txt')
    if os.path.exists(ref_camtxt):
      cam_matrix = np.zeros((4,4),np.float32)
      with open(ref_camtxt) as fi:
        lines = fi.readlines()
        for i in range(4):
          line = lines[i].strip().split(' ')
          for j in range(4):
            cam_matrix[i,j] = float(line[j])
        dataset.sfm.ref_rT = pt.from_numpy(cam_matrix[:3,:3]).t()
        dataset.sfm.ref_t = pt.from_numpy(cam_matrix[:3,3:4])
    sampler_train = SubsetRandomSampler(indices_train)
    sampler_val = SubsetRandomSampler(indices_val)
    dataloader_train = DataLoader(dataset, batch_size=1, sampler = sampler_train, num_workers = num_workers)
    dataloader_val = DataLoader(dataset, batch_size=1, sampler = sampler_val, num_workers = num_workers)
    print('TRAINING IMAGES: {}'.format(len(dataloader_train)))
    print('VALIDATE IMAGES: {}'.format(len(dataloader_val)))
  return sampler_train, sampler_val, dataloader_train, dataloader_val

def drawBottomBar(status):
  def print_there(x, y, text):
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()

  def move (y, x):
    print("\033[%d;%dH" % (y, x))

  columns, rows = os.get_terminal_size()

  # status += "\x1B[K\n"
  status += " " * ((columns - (len(status) % columns)) % columns)
  # status += " " * (columns)

  lines = int(len(status) / columns)
  print("\n" * (lines), end="")
  print_there(rows - lines, 0, " " * columns)
  print_there(rows - lines + 1, 0, "\33[38;5;72m\33[48;5;234m%s\33[0m" % status)
  move(rows - lines - 1, 0)

class TrainingStatus:
  def __init__(self,
               num_steps,
               eta_interval=25,
               statusbar=""):

    self.eta_interval = eta_interval
    self.num_steps = num_steps

    self.etaCount = 0
    self.etaStart = time.time()
    self.duration = 0

    self.statusbar = " ".join(sys.argv)

  def tic(self):
    self.start = time.time()

  def toc(self, iter, loss):
    self.end = time.time()

    self.etaCount += 1
    if self.etaCount % self.eta_interval == 0:
      self.duration = time.time() - self.etaStart
      self.etaStart = time.time()

    etaTime = float(self.num_steps - iter) / self.eta_interval * self.duration
    m, s = divmod(etaTime, 60)
    h, m = divmod(m, 60)
    etaString = "%d:%02d:%02d" % (h, m, s)
    msg = ("%.2f%% (%d/%d): %.3e  t %.3f  @ %s (%s)" % (iter * 100.0 / self.num_steps, iter, self.num_steps, loss, self.end - self.start, time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + etaTime)), etaString))

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
      barText = "Command: CUDA_VISIBLE_DEVICES=%s python %s" % (os.environ['CUDA_VISIBLE_DEVICES'], self.statusbar)
    else:
      barText = "Command: python %s" % (self.statusbar)
    try:
      drawBottomBar(barText)
    except:
      pass #skip bottombar if it no output
    return msg

# use in render_depth
def Rainbow(val):
  rainbow = [
  [0.18995, 0.07176, 0.23217], [0.22500, 0.16354, 0.45096],
  [0.25107, 0.25237, 0.63374], [0.26816, 0.33825, 0.78050],
  [0.27628, 0.42118, 0.89123], [0.27543, 0.50115, 0.96594],
  [0.25862, 0.57958, 0.99876], [0.21382, 0.65886, 0.97959],
  [0.15844, 0.73551, 0.92305], [0.11167, 0.80569, 0.84525],
  [0.09267, 0.86554, 0.76230], [0.12014, 0.91193, 0.68660],
  [0.19659, 0.94901, 0.59466], [0.30513, 0.97697, 0.48987],
  [0.42778, 0.99419, 0.38575], [0.54658, 0.99907, 0.29581],
  [0.64362, 0.98999, 0.23356], [0.72596, 0.96470, 0.20640],
  [0.80473, 0.92452, 0.20459], [0.87530, 0.87267, 0.21555],
  [0.93301, 0.81236, 0.22667], [0.97323, 0.74682, 0.22536],
  [0.99314, 0.67408, 0.20348], [0.99593, 0.58703, 0.16899],
  [0.98360, 0.49291, 0.12849], [0.95801, 0.39958, 0.08831],
  [0.92105, 0.31489, 0.05475], [0.87422, 0.24526, 0.03297],
  [0.81608, 0.18462, 0.01809], [0.74617, 0.13098, 0.00851],
  [0.66449, 0.08436, 0.00424], [0.47960, 0.01583, 0.01055]]

  ind = val * (len(rainbow)-1)
  color0 = np.array(rainbow[int(ind)])
  color1 = np.array(rainbow[min(int(ind) + 1, len(rainbow)-1)])

  intt = ind - int(ind)
  color = color0 * (1-intt) + color1 * intt
  return color

def colored_hook(home_dir):
  """Colorizes python's error message.

  Args:
    home_dir: directory where code resides (to highlight your own files).
  Returns:
    The traceback hook.
  """

  def hook(type_, value, tb):
    def colorize(text, color, own=0):
      """Returns colorized text."""
      endcolor = "\x1b[0m"
      codes = {
          "green": "\x1b[0;32m",
          "green_own": "\x1b[1;32;40m",
          "red": "\x1b[0;31m",
          "red_own": "\x1b[1;31m",
          "yellow": "\x1b[0;33m",
          "yellow_own": "\x1b[1;33m",
          "black": "\x1b[0;90m",
          "black_own": "\x1b[1;90m",
          "cyan": "\033[1;36m",
      }
      return codes[color + ("_own" if own else "")] + text + endcolor

    for filename, line_num, func, text in traceback.extract_tb(tb):
      basename = os.path.basename(filename)
      own = (home_dir in filename) or ("/" not in filename)

      print(colorize("\"" + basename + '"', "green", own) + " in " + func)
      print("%s:  %s" % (
          colorize("%5d" % line_num, "red", own),
          colorize(text, "yellow", own)))
      print("  %s" % colorize(filename, "black", own))

    print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
  return hook

class ServeFilesHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        http.server.SimpleHTTPRequestHandler.end_headers(self)
    @classmethod
    def Creator(cls, *args, **kwargs):
        def _HandlerCreator(request, client_address, server):
            cls(request, client_address, server, *args, **kwargs)
        return _HandlerCreator

def open_webgl_on_nexmpi(address,port,model_dir):
    webbrowser.open_new('https://nex-mpi.github.io/viewer/viewer.html?scene=http://{}:{}/{}'.format(address,port,model_dir))
    
def serve_files(model_dir='',web_path='runs/html/'):
    print(web_path)
    with socketserver.TCPServer(("localhost", 0), ServeFilesHandler.Creator(directory=web_path)) as http:
        address = http.server_address[0]
        port = http.server_address[1]
        print("serving real-time demo at http://{}:{}/{}".format(address,port,model_dir))
        #need delay for waiting http server start listening
        Timer(2.0, open_webgl_on_nexmpi, (address,port,model_dir)).start() 
        http.serve_forever()
