

def iou_bev(box1, box2):
  """
  boxes are numpy arrays: x_min, y_min, x_max, y_max
  box example: np.array([x_min, y_min, x_max, y_max])
  """
  x1_min, y1_min, x1_max, y1_max = box1
  x2_min, y2_min, x2_max, y2_max = box2

  # boundary of the intersection
  inter_xmin = max(x1_min, x2_min)
  inter_ymin = max(y1_min, y2_min)
  inter_xmax = min(x1_max, x2_max)
  inter_ymax = min(y1_max, y2_max)

  inter_w = max(0, inter_xmax - inter_xmin)
  inter_h = max(0, inter_ymax - inter_ymin)
  inter_area = inter_w * inter_h

  # Areas
  area1 = (x1_max - x1_min) * (y1_max - y1_min)
  area2 = (x2_max - x2_min) * (y2_max - y2_min)

  union_area = area1 + area2 - inter_area

  return inter_area / union_area if union_area > 0 else 0.0


def iou_3d(box1, box2):
  """
  boxes are numpy arrays: x_min, y_min, z_min, x_max, y_max, z_max
  box example: np.array([x_min, y_min, z_min, x_max, y_max, z_max])
  """
  x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
  x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

  # boundary of the intersection
  inter_xmin = max(x1_min, x2_min)
  inter_ymin = max(y1_min, y2_min)
  inter_zmin = max(z1_min, z2_min)
  inter_xmax = min(x1_max, x2_max)
  inter_ymax = min(y1_max, y2_max)
  inter_zmax = min(z1_max, z2_max)

  inter_w = max(0, inter_xmax - inter_xmin)
  inter_l = max(0, inter_ymax - inter_ymin)
  inter_h = max(0, inter_zmax - inter_zmin)
  inter_vol = inter_w * inter_l * inter_h

  vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
  vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)

  union_vol = vol1 + vol2 - inter_vol

  return inter_vol / union_vol if union_vol > 0 else 0.0
