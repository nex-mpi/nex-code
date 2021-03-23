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
from skimage import io
import numpy as np
import os
import re
import struct
import json
import glob
from scipy.spatial.transform import Rotation
import copy
from utils.load_llff import load_llff_data

class SfMData:
  def __init__(self, dataset, ref_img='', scale=1, dmin=0,
    dmax=0, invz=0, render_style='', offset=200):
    self.scale = scale
    self.ref_cam = None
    self.ref_img = None
    self.render_poses = None
    self.dmin = dmin
    self.dmax = dmax
    self.invz = invz
    self.dataset = dataset
    self.dataset_type = 'unknown'
    self.render_style = render_style
    self.white_background = False #change background to white if transparent.
    self.index_split = [] #use for split dataset in blender
    self.offset = 200
    # Detect dataset type
    can_hanle = self.readDeepview(dataset) \
      or self.readLLFF(dataset, ref_img) \
      or self.readColmap(dataset) 
    if not can_hanle:
      raise Exception('Unknow dataset type')
    # Dataset processing
    self.cleanImgs()
    self.selectRef(ref_img)
    self.scaleAll(scale)
    self.selectDepth(dmin, dmax, offset)
    if self.dataset_type != 'llff': self.webgl = None

  def cleanImgs(self):
    """
    Remvoe non exist image from self.imgs
    """
    todel = []
    for image in self.imgs:
      img_path = self.dataset + "/" + self.imgs[image]['path']
      if "center" not in self.imgs[image] or not os.path.exists(img_path):
        todel.append(image)
    for it in todel:
      del self.imgs[it]

  def selectRef(self, ref_img):
    """
    Select Reference image
    """
    if ref_img == "" and self.ref_cam is not None and self.ref_img is not None:
      return
    for img_id, img in self.imgs.items():
      if ref_img in img["path"]:
        self.ref_img = img
        self.ref_cam = self.cams[img["camera_id"]]
        return
    raise Exception("reference view not found")

  def selectDepth(self, dmin, dmax, offset):
    """
    Select dmin/dmax from planes.txt / bound.txt / argparse
    """
    if self.dmin < 0 or self.dmax < 0:
      if os.path.exists(self.dataset + "/bounds.txt"):
        with open(self.dataset + "/bounds.txt", "r") as fi:
          data = [np.reshape(np.matrix([float(y) for y in x.split(" ")]), [3, 1]) for x in fi.readlines()[3:]]
        ls = []
        for d in data:
          v = self.ref_img['r'] * d + self.ref_img['t']
          ls.append(v[2])
        self.dmin = np.min(ls)
        self.dmax = np.max(ls)
        self.invz = 0

      elif os.path.exists(self.dataset + "/planes.txt"):
        with open(self.dataset + "/planes.txt", "r") as fi:
          data = [float(x) for x in fi.readline().split(" ")]
          if len(data) == 3:
            self.dmin, self.dmax, self.invz = data
          elif len(data) == 2:
            self.dmin, self.dmax = data
          elif len(data) == 4:
            self.dmin, self.dmax, self.invz, self.offset = data
            self.offset = int(self.offset)
            print(f'Read offset from planes.txt: {self.offset}')
          else:
            raise Exception("Malform planes.txt")
      else:
        print("no planes.txt or bounds.txt found")
    if dmin > 0:
      print("Overriding dmin %f-> %f" % (self.dmin, dmin))
      self.dmin = dmin
    if dmax > 0:
      print("Overriding dmax %f-> %f" % (self.dmax, dmax))
      self.dmax = dmax
    if offset != 200:
      print(f"Overriding offset {self.offset}-> {offset}")
      self.offset = offset
    print("dmin = %f, dmax = %f, invz = %d, offset = %d" % (self.dmin, self.dmax, self.invz, self.offset))

  def readLLFF(self, dataset, ref_img = ""):
    """
    Read LLFF
    Parameters:
      dataset (str): path to datasets
      ref_img (str): ref_image file name
    Returns:
      bool: return True if successful load LLFF data
    """
    if not os.path.exists(os.path.join(dataset,'poses_bounds.npy')):
      return False
    image_dir = os.path.join(dataset,'images')
    if not os.path.exists(image_dir) and not os.path.isdir(image_dir):
      return False
    # load R,T
    train_poses, reference_depth, reference_view_id, render_poses, poses, intrinsic, self.webgl = load_llff_data(dataset,factor=None, split_train_val = 8, render_style = self.render_style)
    # get all image of this dataset
    images_path = [os.path.join('images', f) for f in sorted(os.listdir(image_dir))]

    #LLFF dataset has only single camera in dataset
    #H, W, focal = train_poses[reference_view_id,0,-1],train_poses[reference_view_id,1,-1],train_poses[reference_view_id,2,-1]
    if len(intrinsic) == 3:
      H, W, f = intrinsic
      cx = W / 2.0
      cy =  H / 2.0
      fx = f
      fy = f
    else:
      H, W, fx, fy, cx, cy = intrinsic


    self.cams = {0 : buildCamera(W,H,fx,fy,cx,cy) }

    # create render_poses for video render
    self.render_poses = buildNerfPoses(render_poses)

    # create imgs pytorch dataset
    # we store train and validation together
    # but it will sperate later by pytorch dataloader
    self.imgs = buildNerfPoses(poses, images_path)

    # if not set ref_cam, use LLFF ref_cam
    if ref_img == "":
      # restore image id back from reference_view_id
      # by adding missing validation index
      image_id = reference_view_id + 1 #index 0 alway in validation set
      image_id = image_id  + (image_id // 8) #every 8 will be validation set
      self.ref_cam = self.cams[0]

      self.ref_img = self.imgs[image_id] # here is reference view from train set

    # if not set dmin/dmax, use LLFF dmin/dmax
    if (self.dmin < 0 or self.dmax < 0) and (not os.path.exists(dataset + "/planes.txt")):
      self.dmin = reference_depth[0]
      self.dmax = reference_depth[1]
    self.dataset_type = 'llff'
    return True


  def scaleAll(self, scale):
    self.ocams = copy.deepcopy(self.cams) # original camera
    for cam_id in self.cams.keys():
      cam = self.cams[cam_id]
      ocam = self.ocams[cam_id]

      nw = round(ocam['width'] * scale)
      nh = round(ocam['height'] * scale)
      sw = nw / ocam['width']
      sh = nh / ocam['height']
      cam['fx'] = ocam['fx'] * sw
      cam['fy'] = ocam['fy'] * sh
      cam['px'] = (ocam['px']+0.5) * sw - 0.5
      cam['py'] = (ocam['py']+0.5) * sh - 0.5
      cam['width'] = nw
      cam['height'] = nh

  def readDeepview(self, dataset):
    if not os.path.exists(os.path.join(dataset, "models.json")):
      return False

    self.cams, self.imgs = readCameraDeepview(dataset)
    self.dataset_type = 'deepview'
    return True

  def readColmap(self, dataset):
    sparse_folder = dataset +"/dense/sparse/"
    image_folder = dataset + "/dense/images/"
    if (not os.path.exists(image_folder)) or (not os.path.exists(sparse_folder)):
      return False

    self.imgs = readImagesBinary(os.path.join(sparse_folder, "images.bin"))
    self.cams = readCamerasBinary(sparse_folder + "/cameras.bin")
    self.dataset_type = 'colmap'
    return True

  
def readCameraDeepview(dataset):
  cams = {}
  imgs = {}
  with open(os.path.join(dataset, "models.json"), "r") as fi:
    js = json.load(fi)
    for i, cam in enumerate(js):
      for j, cam_info in enumerate(cam):
        img_id = cam_info['relative_path']
        cam_id = img_id.split('/')[0]

        rotation = Rotation.from_rotvec(np.float32(cam_info['orientation'])).as_matrix().astype(np.float32)
        position = np.array([cam_info['position']], dtype='f').reshape(3, 1)

        if i ==  0:
          cams[cam_id] = {
            'width': int(cam_info['width']),
            'height': int(cam_info['height']),
            'fx': cam_info['focal_length'],
            'fy': cam_info['focal_length'] * cam_info['pixel_aspect_ratio'],
            'px': cam_info['principal_point'][0],
            'py': cam_info['principal_point'][1]
          }
        imgs[img_id] = {
          "camera_id": cam_id,
          "r": rotation,
          "t": -np.matmul(rotation, position),
          "R": rotation.transpose(),
          "center": position,
          "path": cam_info['relative_path']
        }
  return cams, imgs

def readImagesBinary(path):
  images = {}
  f = open(path, "rb")
  num_reg_images = struct.unpack('Q', f.read(8))[0]
  for i in range(num_reg_images):
    image_id = struct.unpack('I', f.read(4))[0]
    qv = np.fromfile(f, np.double, 4)

    tv = np.fromfile(f, np.double, 3)
    camera_id = struct.unpack('I', f.read(4))[0]

    name = ""
    name_char = -1
    while name_char != b'\x00':
      name_char = f.read(1)
      if name_char != b'\x00':
        name += name_char.decode("ascii")


    num_points2D = struct.unpack('Q', f.read(8))[0]

    for i in range(num_points2D):
      f.read(8 * 2) # for x and y
      f.read(8) # for point3d Iid

    r = Rotation.from_quat([qv[1], qv[2], qv[3], qv[0]]).as_dcm().astype(np.float32)
    t = tv.astype(np.float32).reshape(3, 1)

    R = np.transpose(r)
    center = -R @ t
    # storage is scalar first, from_quat takes scalar last.
    images[image_id] = {
      "camera_id": camera_id,
      "r": r,
      "t": t,
      "R": R,
      "center": center,
      "path": "dense/images/" + name
    }

  f.close()
  return images

def readCamerasBinary(path):
  cams = {}
  f = open(path, "rb")
  num_cameras = struct.unpack('Q', f.read(8))[0]

  # becomes pinhole camera model , 4 parameters
  for i in range(num_cameras):
    camera_id = struct.unpack('I', f.read(4))[0]
    model_id = struct.unpack('i', f.read(4))[0]

    width = struct.unpack('Q', f.read(8))[0]
    height = struct.unpack('Q', f.read(8))[0]

    fx = struct.unpack('d', f.read(8))[0]
    fy = struct.unpack('d', f.read(8))[0]
    px = struct.unpack('d', f.read(8))[0]
    py = struct.unpack('d', f.read(8))[0]

    cams[camera_id] = {
      "width": width,
      "height": height,
      "fx": fx,
      "fy": fy,
      "px": px,
      "py": py
    }
    # fx, fy, cx, cy
  f.close()
  return cams

def nerf_pose_to_ours(cam):
  R = cam[:3, :3]
  center = cam[:3, 3].reshape([3,1])
  center[1:] *= -1
  R[1:, 0] *= -1
  R[0, 1:] *= -1

  r = np.transpose(R)
  t = -r @ center
  return R, center, r, t

def buildCamera(W,H,fx,fy,cx,cy):
  return {
    "width": int(W),
    "height": int(H),
    "fx": float(fx),
    "fy": float(fy),
    "px": float(cx),
    "py": float(cy)
  }

def buildNerfPoses(poses, images_path = None):
  output = {}
  for poses_id in range(poses.shape[0]):
    R, center, r, t = nerf_pose_to_ours(poses[poses_id].astype(np.float32))
    output[poses_id] = {
      "camera_id": 0,
      "r": r,
      "t": t,
      "R": R,
      "center": center
    }
    if images_path is not None:
      output[poses_id]["path"] = images_path[poses_id]

  return output
