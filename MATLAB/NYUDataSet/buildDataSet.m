%% Building the Data Set 

% loading images and depth maps
load('./Data/nyu_depth_v2_labeled.mat', 'images');
load('./Data/nyu_depth_v2_labeled.mat', 'rawDepths');

for ii = 1:1449 % All images
    normImg = imread(sprintf('./Data/Ladicky/%05d.png', ii)); % load image of normal map 
    normMap = ladickyToNormals(normImg); % convert to normal map
    mask = getValidValuesMask(rawDepths(:,:,ii)); % get valid values
    normals(:,:,:,ii) = normMap .* mask; % apply the mask
end

save('./Data/NYUDataSet.mat', 'images', 'normals','-v7.3');
