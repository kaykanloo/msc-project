% Projects the depth points from the image plane to the 3D world
% coordinates.
%
% Args:
%   imgDepth - depth map which has already been projected onto the RGB
%              image plane, an HxW matrix where H and W are the height and
%              width of the matrix, respectively.
%   params -   camera parameters
%
% Returns:
%   points3d - the point cloud in the world coordinate frame, an Nx3
%              matrix.
function points3d = rgb_plane2rgb_world(imgDepth)

  [H, W] = size(imgDepth);
  
  % RGB Intrinsic Parameters
  fx_rgb = 5.1885790117450188e+02;
  fy_rgb = 5.1946961112127485e+02;
  cx_rgb = 3.2558244941119034e+02;
  cy_rgb = 2.5373616633400465e+02;
  
  % Make the original consistent with the camera location:
  [xx, yy] = meshgrid(1:W, 1:H);

  x3 = (xx - cx_rgb) .* imgDepth / fx_rgb;
  y3 = (yy - cy_rgb) .* imgDepth / fy_rgb;
  z3 = imgDepth;
  
  points3d = [x3(:) -y3(:) z3(:)];
end