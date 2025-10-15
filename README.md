## Todo list and checks:

### Done
  * Initialize OoB mask outside of runtime ☑
  * Converting ground truth to tensor (currently list) ☑
  * Removal of all lists and convert them to tensor ☑
  * Modification of objectness loss ☑
  * Correction of classification loss by penalizing non valid voxels ☑

### ToDos:
  
  * Modification of offset regression loss (why not including center voxel to the method?)
  * Modification of dimension and yaw loss
  * A strict check on objects to avoid in gtl for gtl[-1] == 0
  * Why out of the boundary not be assigned to -3? why -1???
  * Modifying the architecture (change the voxel shape to conical! from small voxels near to coarse ones at far)

### Architecture overview:
> It is possible to have two different head, with modified disparity. One is to avoid at all classification and put in as binary occupancy grid maps, another one is to have the same head but reduce the tensor channels at disparity.
