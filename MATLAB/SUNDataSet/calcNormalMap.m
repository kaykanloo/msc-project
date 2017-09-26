% Calculates the pixel-wise surface normal map of depth images
% Args:
%   imgRawDepth - HxW raw depth image.
%   params - Camera intrinsic parameters
%   align - true: align the normals 
%
% Returns:
%   normMap - HxWx3 matrix of surface normals at each pixel.

function normMap = calcNormalMap(imgRawDepth, denoise)
    
    imgD = im2double(imgRawDepth);
    sz = size(imgD);
    
    % Project the depth points from the image plane to the 3D world coordinates.
    points3d = rgb_plane2rgb_world(imgD);
    X = points3d(:,1);
    Y = -points3d(:,2); % Y axis pointing upward
    Z = points3d(:,3);
    
    % Computing normals
    normMap = compute_local_planes(X, Y, Z, sz); 
    
    % Valid values mask
    mask = sum(normMap.^2,3).^0.5 > 0.5;
    
    % Denoising
    if denoise == true
        
       addpath('./Denoise/');
       % tv-denoise the surface normals
       denoisedNormMap  = tvNormal(normMap,1);
       
       % Normalisation
       normMap = bsxfun(@rdivide,denoisedNormMap,sum(denoisedNormMap.^2,3).^0.5+eps);
    end
    
    % Valid values mask
    mask = mask .* (imgRawDepth ~= 0);

    % Applying the mask
    normMap = normMap .* mask;
    
end