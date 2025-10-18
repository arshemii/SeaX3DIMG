# Branch guide:

### ImgOnly:
* only for camera data [no LiDAR or Radar] - this is the basic branch
* Loss weights are [1.0, 1.0, 0.15, 0.05, 0.2]

### ImgOnly_LT
* only for camera data [no LiDAR or Radar]
* Ground truth is tensor --> loss function and evaluation use vectorization
* Loss function considers also loss for backgrounds with high objectness
* Loss function has different weights for each term: loss.weight = [1.0, 1.0, 0.75, 0.65, 0.2]

### ImgOnly_LT_V2 (ToDo)
* Change the calculation of evaluation module
* Increase the grid resolution
* Produce assignment and nearest voxel centers to the dataset module
* Change the head aggregation
* Check if memory is necessary!
