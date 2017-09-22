%% Converts the ground truth surface normal images provided by Ladicky et al. to normal maps
% Input: RGB image
% Output: Matrix of surface normals
%
% Reference: 
% Lubor Ladicky, Bernhard Zeisl, and Marc Pollefeys. “Discriminatively Trained Dense Surface Normal Estimation”. In: ECCV. 2014
% Data: 
% https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip

function normals = ladickyToNormals(img)
    normals = single(cat(3,img(:,:,3),img(:,:,2),img(:,:,1))) ./ 255; % Convert from [0 255] range to [0 1]
    normals = (-2 * normals) + 1;  % To [-1 +1] range
    normals = bsxfun(@rdivide,normals,sum(normals.^2,3).^0.5+eps); % Normalisation
end
