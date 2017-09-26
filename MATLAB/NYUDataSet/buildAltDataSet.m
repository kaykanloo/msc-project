%% Building the Data Set (normal maps computed based on the Silberman method)
addpath('./AltMethod/');

% loading images and depth maps
load('./Data/nyu_depth_v2_labeled.mat', 'images');
load('./Data/nyu_depth_v2_labeled.mat', 'rawDepths');

for ii = 1:1449 % All images
    disp(['Computing ',num2str(ii),'/1449 ...']);
    normMap = calcNormalMap(rawDepths(:,:,ii), true); % calculate the normal map
    mask = getValidValuesMask(rawDepths(:,:,ii)); % get valid values
    normals(:,:,:,ii) = single(normMap .* mask); % apply the mask
end

save('./Data/NYUAltDataSet.mat', 'images', 'normals','-v7.3');
