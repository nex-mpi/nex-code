# video_path.py using for generate rendering path for create output video.

# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

import numpy as np 
from scipy.spatial.transform import Rotation, Slerp

def webGLspiralPath(ref_rotation, ref_translation, dmin, dmax, total_frame = 120, spin_radius = 10, total_spin = 1):
  spin_speed = 2*np.pi / total_frame * total_spin
  render_poses = {}
  # matrix conversation helper
  def dcm_to_4x4(r,t):
    camera_matrix = np.zeros((4,4),dtype=np.float32)
    camera_matrix[:3,:3] = r
    if len(t.shape) > 1:
      camera_matrix[:3,3:4] = t
    else:
      camera_matrix[:3,3] = t
    camera_matrix[3,3] = 1.0
    return camera_matrix

  for i in range(total_frame):
    anim_time = spin_speed * i
    leftright = np.sin(anim_time) * spin_radius / 500.0
    updown = np.cos(anim_time) * spin_radius / 500.0
    r = ref_rotation
    t = ref_translation
    cam = dcm_to_4x4(r,t)
    dist = (dmin + dmax) / 2.0
    translation_matrix = dcm_to_4x4(np.eye(3), np.array([0,0, -dist]))
    translation_matrix2 = dcm_to_4x4(np.eye(3), np.array([0,0, dist]))
    euler_3x3 = Rotation.from_euler('yxz', [leftright, updown, 0]).as_dcm()
    euler_4x4 = dcm_to_4x4(euler_3x3, np.array([0.0,0.0,0.0]))
    output = translation_matrix2 @ euler_4x4 @  translation_matrix @ cam
    output = output.astype(np.float32)
    r = output[:3, :3]
    t = output[:3, 3:4]
    render_poses[i] = {'r': r, 't': t}
  return render_poses

def deepviewInnerCircle(sfm, inter_frame = 30):
  '''
  Deepview Inner Circle render
  render across cam 1,2 (training view) and 5,11,10,7 (eval view)
  '''
  indices = sfm.index_split[0] + sfm.index_split[1]
  indices = sorted(indices) # assume, space dataset always sortable 
  images = list(sfm.imgs.values())
  selected_cam = [images[indices[i]] for i in [7,1,2,5,11,10,7]]
  render_poses = {}
  for i in range(len(selected_cam)-1):
    # use Slerp to interpolate between 2 rotation
    rot = Rotation.from_dcm([selected_cam[i]['r'], selected_cam[i+1]['r']])
    slerp = Slerp([0,1], rot) 
    times = np.linspace(0.0, 1.0, num=inter_frame+1)[:-1]
    interp_rots = slerp(times).as_dcm().astype(np.float32)
    for j in range(inter_frame):
      step = j / inter_frame
      t = selected_cam[i]['t'] * (1 - step) + step *  selected_cam[i+1]['t']
      render_poses[i * inter_frame + j] = {
        'r': interp_rots[j],
        't': t
      }

  return render_poses