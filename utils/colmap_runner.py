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
import glob
from scipy import misc
import sys
import json
import argparse
from utils.colmap_read_model import *
import shutil

def cmd(s):
  print(s)
  os.system(s)

def runner(dataset):
  '''
    Use colmap
      1. feature extractor
      2. exhaustive matcher
      3. Mapper
      4. Image undistorter
  '''
  cmd("colmap feature_extractor \
   --database_path " + dataset + "/database.db \
   --image_path " + dataset + "/images \
   --ImageReader.single_camera 1\
   --ImageReader.camera_model SIMPLE_RADIAL")

  cmd("colmap exhaustive_matcher \
   --database_path " + dataset + "/database.db " \
   "--SiftMatching.guided_matching 1 " \
   )

  cmd("mkdir " + dataset + "/sparse")

  cmd("colmap mapper \
    --database_path " + dataset + "/database.db \
    --image_path " + dataset + "/images \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.num_threads 16 \
    --Mapper.extract_colors 0 \
    --output_path " + dataset + "/sparse")

  cmd("mkdir " + dataset + "/dense")

  cmd("cp -r " + dataset + "/sparse " + dataset +"/sparse_before_undistort")
  cmd("cp -r " + dataset + "/database.db " + dataset +"/sparse_before_undistort/database.db")
  cmd("colmap image_undistorter \
    --image_path " + dataset + "/images \
    --input_path " + dataset + "/sparse/0 \
    --output_path " + dataset + "/dense \
    --output_type COLMAP")
  cmd("mv " + dataset + "/images " + dataset +"/images_distort")
  cmd("mv " + dataset +"/dense/images" +" " + dataset +'/images')

def load_colmap_data(realdir):
  '''
    copy from Local light field fusion
    https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py

    make a little change in principle points
    LLFF assumes cx, cy to locate at the center of image (H/2, W/2)
    Whereas we get it from colmap prediction
  '''
  camerasfile = os.path.join(realdir, 'dense/sparse/cameras.bin')
  camdata = read_cameras_binary(camerasfile)

  # cam = camdata[camdata.keys()[0]]
  list_of_keys = list(camdata.keys())
  cam = camdata[list_of_keys[0]]

  h, w, fx, fy, cx, cy = cam.height, cam.width, cam.params[0], cam.params[1], cam.params[2], cam.params[3]

  hwf_cxcy = np.array([h, w, fx, fy, cx, cy]).reshape([6,1])

  imagesfile = os.path.join(realdir, 'dense/sparse/images.bin')
  imdata = read_images_binary(imagesfile)

  w2c_mats = []
  bottom = np.array([0,0,0,1.]).reshape([1,4])

  names = [imdata[k].name for k in imdata]
  print( 'Images #', len(names))
  perm = np.argsort(names)
  for k in imdata:
    im = imdata[k]
    R = im.qvec2rotmat()
    t = im.tvec.reshape([3,1])
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    w2c_mats.append(m)

  w2c_mats = np.stack(w2c_mats, 0)
  c2w_mats = np.linalg.inv(w2c_mats)

  poses = c2w_mats[:, :3, :4].transpose([1,2,0])

  #poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)

  points3dfile = os.path.join(realdir, 'dense/sparse/points3D.bin')
  pts3d = read_points3d_binary(points3dfile)

  # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
  #poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
  poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :]], 1)

  return poses, pts3d, perm, hwf_cxcy

def save_poses(basedir, poses, pts3d, perm, hwf_cxcy):
  pts_arr = []
  vis_arr = []
  for k in pts3d:
    pts_arr.append(pts3d[k].xyz)
    cams = [0] * poses.shape[-1]
    for ind in pts3d[k].image_ids:
      if len(cams) < ind - 1:
        print('ERROR: the correct camera poses for current points cannot be accessed')
        return
      cams[ind-1] = 1
    vis_arr.append(cams)

  pts_arr = np.array(pts_arr)
  vis_arr = np.array(vis_arr)
  print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )

  zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
  valid_z = zvals[vis_arr==1]
  print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )

  save_arr = []
  for i in perm:
    vis = vis_arr[:, i]
    zs = zvals[:, i]
    zs = zs[vis==1]
    close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
    # print( i, close_depth, inf_depth )

    save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
  save_arr = np.array(save_arr)

  np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
  np.save(os.path.join(basedir, 'hwf_cxcy.npy'), hwf_cxcy)

def need_run_coolmap(basedir):
  files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
  if os.path.exists(os.path.join(basedir, 'poses_bounds.npy') and os.path.join(basedir, 'hwf_cxcy.npy')):
    return False
  if os.path.exists(os.path.join(basedir, 'sparse/0')):
    files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
  elif os.path.exists(os.path.join(basedir, 'dense/sparse')):
    files_had = os.listdir(os.path.join(basedir, 'dense/sparse'))
  else:
    files_had = []
  if not all([f in files_had for f in files_needed]):
    print( 'Need to run COLMAP' )
    return True
  else:
    return False

def colmapGenPoses(dpath):
  files = os.listdir(dpath)
  #no need colmap on deepview
  if 'models.json' in files:
    return False
  #no need colmap on blender
  if 'transforms_train.json' in files:
    return False
  
  if need_run_coolmap(dpath):
    '''
      Automatically run colmap
      Get near, far planes, ref_image using code from LLFF
    '''
    if shutil.which('colmap') is None:
      print('You need to install COLMAP in this machine')
      raise Exception("No COLMAP found in this machine")

    runner(dpath)
    print( 'Post-colmap')
    poses, pts3d, perm, hwf_cxcy = load_colmap_data(dpath)
    save_poses(dpath, poses, pts3d, perm, hwf_cxcy)
    print( 'Done with imgs2poses' )
    return True
  else: 
    return False