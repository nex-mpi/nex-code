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
from torch.utils.data import Dataset

import numpy as np
import os
from utils.sfm_utils import SfMData
from utils.video_path import webGLspiralPath, deepviewInnerCircle
import torch as pt
from collections import deque
from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.transform import resize
import cv2
import lpips
import pandas as pd
import shutil

class OrbiterDataset(Dataset):
  def __init__(self, dataset, ref_img, scale, dmin,
      dmax, invz, transform=None,
      render_style='', offset=200, cv2resize=False):
    self.scale = scale
    self.dataset = dataset
    self.transform = transform
    self.sfm = SfMData(dataset,
                       ref_img=ref_img,
                       dmin=dmin,
                       dmax=dmax,
                       invz=invz,
                       scale=scale,
                       render_style=render_style,
                       offset = offset)

    self.sfm.ref_rT = pt.from_numpy(self.sfm.ref_img['r']).t()
    self.sfm.ref_t = pt.from_numpy(self.sfm.ref_img['t'])
    self.cv2resize = cv2resize

    self.imgs = []

    self.ref_id = -1
    for i, ind in enumerate(self.sfm.imgs):
      img = self.sfm.imgs[ind]
      self.imgs.append(img)
      if ref_img in img['path']:
        self.ref_id = len(self.imgs) - 1

    self.cache = {}
    self.cache_queue = deque()
    self.cache_size = 50


  def fromCache(self, img_path, scale):
    p = (img_path, scale)
    if p in self.cache:
      return self.cache[p]

    img = io.imread(img_path)

    if img.shape[2] > 3 and self.sfm.white_background:
      img[img[:, :, 3] == 0] = [255, 255, 255, 0]

    img = img[:, :, :3]

    if scale != 1:
      h, w = img.shape[:2]
      if self.sfm.dataset_type == 'deepview':
        newh = int(h * scale) #always floor down height
        neww = round(w * scale)
      else:
        newh = round(h * scale)
        neww = round(w * scale)
        
      if self.cv2resize:
        img = cv2.resize(img, (neww, newh),interpolation=cv2.INTER_AREA)
      else:
        img = resize(img, (newh, neww))

    if len(self.cache) == self.cache_size:
      dp = self.cache_queue.popleft()
      del self.cache[dp]

    self.cache_queue.append(p)
    self.cache[p] = img
    return img

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    if pt.is_tensor(idx):
        idx = idx.tolist()

    img = self.fromCache(self.dataset + "/" + self.imgs[idx]['path'], self.scale)

    img = np.transpose(img, [2, 0, 1]).astype(np.float32)
    if np.max(img) > 1:
      img /= 255.0

    im = self.imgs[idx]
    cam = self.sfm.cams[im['camera_id']]
    feature = {
      'image': img,
      'height': img.shape[1],
      'width': img.shape[2],
      'r': im['r'],
      't': im['t'],
      'center':im['center'],
      'fx': cam['fx'],
      'fy': cam['fy'],
      'px': cam['px'],
      'py': cam['py'],
      'path': im['path']
    }

    return feature

def generateViewer(outputFile, w, h, planes, f, px, py,
      nSubPlanes, invz, version, extrinsics, maxcol=0,
      offset=0, layers=0, sublayers=0,
      rads = None, focal = None):
  print("Generating WebGL viewer")
  fo = open(outputFile, "w")

  replacer = {}
  replacer["WIDTH"] = w;
  replacer["HEIGHT"] = h;
  replacer["SCALE"] = 1;
  replacer["PLANES"] = "[[" + ",".join([str(x) for x in planes]) + "]]"
  replacer["F"] = f
  replacer["NAMES"] = "[\"\"]"
  replacer["PX"] = px
  replacer["PY"] = py
  replacer["INVZ"] = "true" if invz else "false"
  replacer["NSUBPLANES"] = nSubPlanes
  replacer["LAYERS"] = layers
  replacer["SUBLAYERS"] = sublayers
  replacer["MAXCOL"] = maxcol
  replacer["VERSION"] = 'learned_basis8'
  replacer["OFFSET"] = offset
  replacer["EXTRINSICS"] = "[[" + ",".join([str(x) for x in extrinsics]) + "]]"
  if focal is not None:
    replacer["RADS"] = "[[" + ",".join([str(x) for x in rads]) + "]]"
    replacer["FOCAL"] = focal
  else:
    replacer["RADS"] = None
    replacer["FOCAL"] = None


  st = """
  const w = {WIDTH};
  const h = {HEIGHT};
  const scale = {SCALE};
  const nSubPlanes = {NSUBPLANES};
  const layers = {LAYERS};
  const sublayers = {SUBLAYERS};

  const planes = {PLANES};

  const f = {F} * scale;
  var names = {NAMES};
  var nMpis = names.length;

  const py = {PY} * scale;
  const px = {PX} * scale;
  const invz = {INVZ};

  const offset = {OFFSET};
  const maxcol = {MAXCOL};
  const version = "{VERSION}";
  const extrinsics = {EXTRINSICS};

  const rads = {RADS};
  const focal  = {FOCAL};

  """
  for k in replacer:
    st = st.replace("{" + k + "}", str(replacer[k]))

  fo.write(st + '\n')
  fo.close()

def makegrid(imgs, maxcol, first, last):
  big = []
  cols = []
  for i in range(first, last):
    cols.append(imgs[i, :, :, :4])

    if len(cols) == maxcol:
      big.append(np.concatenate(cols, 1))
      cols = []

  if len(cols):
    cols += [np.zeros_like(cols[-1])] * (maxcol - len(cols))
    big.append(np.concatenate(cols, 1))

  return np.concatenate(big, 0)

def gridOne(imgs, maxcol):
  if imgs.shape[0] %3 != 0:
    imgs = np.concatenate([imgs, np.zeros((3 - imgs.shape[0]%3, imgs.shape[1], imgs.shape[2], 1))], 0)
  indices = [x * int(imgs.shape[0] / 3) for x in range (3)] + [imgs.shape[0]]
  out = []
  for i in range(3):
    out.append(makegrid(imgs, maxcol, indices[i], indices[i + 1]))
  return np.concatenate(out, -1)


# mpi_c [lay, h, w, 3], mpi_a [lay, h, w, 1]
def outputCompactMPI(mpi_reflection, sfm, planes, model_dir_nolp, layers, sublayers, offset, invz, webpath = "runs/html/", web_width = 16000):
  outputFile = "/".join(model_dir_nolp.split("/")[1:])

  maxcol = int(web_width / mpi_reflection['mpi_c'].shape[-1])

  #mpi_c [layers, 3, h, w]
  #mpi_a [layer * sublayers, 1, h, w]
  #mpi_v [layers, 3, h, w, 8]
  #mpi_b [num_basis, 1, 400, 400]
  if not os.path.exists(webpath + outputFile) or True:
    os.makedirs(webpath + outputFile, exist_ok=True)
    for key, value in mpi_reflection.items():
      print(f'predict {key}')
      if key == 'mpi_b':
        value = (value + 1) / 2
        if value.shape[1] == 1:
          value = value.repeat([1, 3, 1, 1])
      if len(value.shape) == 4:
        out = value.permute([0, 2, 3, 1]).cpu().detach().numpy()
        if key == "mpi_a":
          mpi_reflection[key] = gridOne(out, maxcol)
        elif key =='mpi_b':
          mpi_reflection[key] = makegrid(out, out.shape[0], 0, out.shape[0])
        else:
          mpi_reflection[key] = makegrid(out, maxcol, 0, out.shape[0])
        io.imsave(webpath + outputFile + f"/{key}.png",
              np.floor(255 * mpi_reflection[key]).astype(np.uint8))
      elif len(value.shape) > 4:
        #bound basis[-1 to 1] to [0 to 1]
        value = (value + 1) / 2
        out = value.permute([4, 0, 2, 3, 1]).cpu().detach().numpy()

        for i in range(out.shape[0]):
          basis = makegrid(out[i], maxcol, 0, out.shape[1])
          io.imsave(webpath + outputFile + f"/basis_{i + 1}.png",
                np.floor(255 * basis).astype(np.uint8))
    def np2list(x):
      ex = []
      for i in range(x.shape[1]):
        for j in range(x.shape[0]):
          ex.append(x[j, i])
      return ex

    ext = np.concatenate([np.concatenate([sfm.ref_img['r'], sfm.ref_img['t']], 1), np.array([[0.0, 0.0, 0.0, 1.0]])], 0)
    exls = np2list(ext)
    if sfm.webgl is not None:
      rads = [x for x in np.nditer(sfm.webgl['rads'])]
      focal = sfm.webgl['focal']
      generateViewer(webpath + outputFile + "/config.js", sfm.ref_cam['width'], sfm.ref_cam['height'],
          planes, sfm.ref_cam['fx'], sfm.ref_cam['px'], sfm.ref_cam['py'], 1, invz,
          "sharedrgb", exls, maxcol, offset, layers, sublayers,
          rads = rads, focal = focal)
    else:
      generateViewer(webpath + outputFile + "/config.js", sfm.ref_cam['width'], sfm.ref_cam['height'],
          planes, sfm.ref_cam['fx'], sfm.ref_cam['px'], sfm.ref_cam['py'], 1, invz,
          "sharedrgb", exls, maxcol, offset, layers, sublayers)
    print("WebGL saved to {}".format(webpath + outputFile))

def evaluation(model, dataset, dataloader, ray, model_dir_nolp, webpath = 'runs/html/', write_csv = True):
  outputFile = "/".join(model_dir_nolp.split("/")[1:])

  info_dict = measurement(model, dataset.sfm, dataloader, ray, webpath + outputFile + '/rendered_val')
  df = pd.DataFrame.from_dict(info_dict)

  print("===================================")
  print("Measurement Result")
  print("===================================")
  print(df.to_string(index=False))
  print("-----------------------------------")
  print(df.mean().to_string())
  if write_csv:
    df.to_csv(webpath + outputFile +'/ssim_psnr.csv', index=False)
  return df.mean().to_dict()

def render_video(model, dataset, ray, video_path = 'runs/video/', render_type='default', dataloader = None):
  print("render_video: {}".format(render_type))

  os.makedirs(video_path,exist_ok=True)

  feature = dataset.__getitem__(0)
  # require to pad batch dimension to compatible with torch

  for key in feature.keys():

    if key != 'path':
      feature[key] = pt.from_numpy(np.expand_dims(feature[key], axis=0))

  # render is video frame from render_poses
  if dataset.sfm.dataset_type == 'deepview':
    render_poses = deepviewInnerCircle(dataset.sfm)
  elif dataset.sfm.render_poses is not None:
    render_poses = dataset.sfm.render_poses
  else:
    render_poses = webGLspiralPath(
      dataset.sfm.ref_img['r'],
      dataset.sfm.ref_img['t'],
      dataset.sfm.dmin,
      dataset.sfm.dmax
    )
  total_frame = len(render_poses)
  print('Rendering video frame...')

  for pose_id in render_poses:
    pose = render_poses[pose_id]

    if render_type == 'viewing':
      def getCenter(pose):
        feature['r'] = pt.from_numpy(np.expand_dims(pose['r'], axis=0))
        feature['t'] = pt.from_numpy(np.expand_dims(pose['t'], axis=0))
        feature['R'] = feature['r'][0].t()[None]
        return (-feature['R'][0] @ feature['t'][0])[None]
      feature['center'] = getCenter(pose)
      feature['r'] = pt.from_numpy(np.expand_dims(dataset.sfm.ref_rT.T, axis=0))
      feature['t'] = pt.from_numpy(np.expand_dims(dataset.sfm.ref_t, axis=0))

      feature['R'] = feature['r'][0].t()[None]
    else:
      feature['r'] = pt.from_numpy(np.expand_dims(pose['r'], axis=0))
      feature['t'] = pt.from_numpy(np.expand_dims(pose['t'], axis=0))
      feature['R'] = feature['r'][0].t()[None]
      feature['center'] = (-feature['R'][0] @ feature['t'][0])[None]
    if not render_type == 'nearest':
      predict_image = patch_render(model, dataset.sfm, feature, ray)
      predict_image = predict_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    else:
      closet = [(pt.mean((feature['center'][0] - feature_train['center'][0])**2), feature_train['image']) for feature_train in dataloader]
      closet.sort(key = lambda x: x[0])
      predict_image = closet[0][1].permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    image_name = "frame_{:04d}.png".format(pose_id)
    filepath = os.path.join(video_path, image_name)
    print('frame_{:04d}.png ({:d}/{:d})'.format(pose_id, pose_id+1, total_frame))
    io.imsave(filepath, (predict_image * 255).astype(np.uint8))
  if shutil.which('ffmpeg') is None:
    print("Skiping create video, No FFmpeg install on this PC.")
    return
  video_filepath = os.path.join(video_path,'video.mp4')
  if os.path.exists(video_filepath):
    os.remove(video_filepath)
  os.system('ffmpeg -r 30 -i {}/frame_%04d.png -c:v libx264 -crf 12 -pix_fmt yuv420p -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" {}/video.mp4 < /dev/null'.format(video_path, video_path))
  print("output video at {}/video.mp4".format(video_path))

def patch_render(model, sfm, feature, ray):
  """ render image in patch style """
  gt = feature['image']
  predicted_image = []
  # ray *= 2.5
  num = np.ceil(gt.shape[2] * gt.shape[3]/ray).astype(np.int64)
  model.eval()
  with pt.no_grad():
    for i in range(num):
      output_shape = feature['image'].shape[-2:]
      if i < num -1 :
        selection = pt.arange(ray * i, ray * (i + 1))
      else:
        selection = pt.arange(ray * i, gt.shape[2] * gt.shape[3])

      output = model(sfm, feature, output_shape, selection)

      predicted_image.append(output)

    predicted_image = pt.cat(predicted_image, 2)
    out = predicted_image.view(gt.shape[0], gt.shape[1], gt.shape[2], gt.shape[3])
  return out

def measurement(model, sfm, dataloader, ray, write_path = ''):
  """ calculate PSNR/SSIM and LPIPS"""
  psnrs = []
  ssims = []
  lpipss = []
  filenames = []
  if write_path != '':
    os.makedirs(write_path,exist_ok=True)
  lpips_model = lpips.LPIPS(net='vgg')
  for i, feature in enumerate(dataloader):
    gt = feature['image'] # "ground-truth"
    predict_image = patch_render(model, sfm, feature, ray)

    # calculate LPIPS. It's require to change image range from [0,1] to [-1,1]
    gt_lpips = gt.clone().cpu() * 2.0 - 1.0
    predict_image_lpips = predict_image.clone().detach().cpu() * 2.0 - 1.0
    lpips_result = lpips_model.forward(predict_image_lpips, gt_lpips).cpu().detach().numpy()
    lpipss.append(np.squeeze(lpips_result))
    # calculate PSNR/SSIM
    predict_image = predict_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    gt = gt.permute(0, 2, 3, 1).numpy()[0]
    ssims.append(structural_similarity(predict_image, gt, win_size=11, multichannel=True, gaussian_weights=True))
    psnrs.append(peak_signal_noise_ratio(predict_image, gt, data_range=1.0))

    # save output image
    filename = '_'.join(feature['path'][0].split('/'))
    filenames.append(filename)
    if write_path != '':
      filename = '{}.png'.format('.'.join(filename.split('.')[:-1]))
      io.imsave(os.path.join(write_path, '{}_ours.png'.format(filename)), (predict_image * 255).astype(np.uint8))
      io.imsave(os.path.join(write_path, '{}_gt.png'.format(filename)), (gt * 255).astype(np.uint8))
      
  return {
    'name': filenames,
    'PSNR': psnrs,
    'SSIM': ssims,
    'LPIPS': lpipss
  }

def getPlanes(sfm, n):

  if sfm.invz:
    return 1/np.linspace(1, sfm.dmin / sfm.dmax, n) * sfm.dmin
  else:
    return np.linspace(sfm.dmin, sfm.dmax, n)

def patchtify(img, size):
  output_shape = img.shape[-2:]
  if size >= output_shape[0] or size >= output_shape[1]:
    select_row = [0]
    select_col = [0]
  else:
    select_row = np.random.randint(0, output_shape[0] - size, 1)
    select_col = np.random.randint(0, output_shape[1] - size, 1)
  y, x = pt.meshgrid([pt.arange(0, size) + select_row[0],
                      pt.arange(0, size) + select_col[0]])
  sel = pt.flatten(y * output_shape[1] + x).to(pt.long)
  img = img.view(img.shape[0], img.shape[1], img.shape[2] * img.shape[3])
  img = img[:, :, sel, None].cuda()

  return img, sel

def Lsel(shape, ray):
  #shape [h, w]
  # o o
  # o
  selRow = pt.tensor(np.random.randint(0, shape[0] - 1, int(ray/3)))
  selCol = pt.tensor(np.random.randint(0, shape[1] - 1, int(ray/3)))
  sel = pt.zeros(3 * int(ray/3)).to(pt.long)
  pick = selRow * shape[1]  + selCol

  sel[0::3] = pick
  sel[1::3] = pick + 1
  sel[2::3] = pick + shape[1]
  return sel