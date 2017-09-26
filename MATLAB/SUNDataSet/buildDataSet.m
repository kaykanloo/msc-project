%% Building the data set

load('NYUMeta.mat'); % Data set meta data

for ii = 1:1449

    disp(['Computing ',num2str(ii),'/1449 ...']);
    
    % loading image and depth map
    imgRGB = imread(['./Data/',Meta(ii).sequenceName,'/fullres/',Meta(ii).rgbname]);
    imgRawDepth = imread(['./Data/',Meta(ii).sequenceName,'/fullres/',Meta(ii).depthname]);
    
    images(:,:,:,ii) = imgRGB;
    
    % Calculating surface normal maps
    normMap = calcNormalMap(imgRawDepth, true);
    normals(:,:,:,ii) = single(normMap);
    
end

save('./Data/SUNDataSet.mat', 'images', 'normals','-v7.3');