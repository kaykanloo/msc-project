%% Shows the surface normal map as an image
% Input: Matrix of normals
function showNormalMap(normals)
    imshow((normals+1) / 2);
end