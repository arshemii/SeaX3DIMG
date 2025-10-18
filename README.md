# Branch guide:

### ImgOnly:
* only for camera data [no LiDAR or Radar] - this is the basic branch

### ImgOnly_LT
* only for camera data [no LiDAR or Radar]
* Ground truth is tensor --> loss function and evaluation use vectorization
* Loss function considers also loss for backgrounds with high objectness



### Do we need a 2d BEV?
