%% Returns a mask of valid values based on the valid raw depth data
% Input: Raw depth data
% Output: Logical map of valid values
function mask = getValidValuesMask(rawDepth)
    minDepth = 0; % No distance to camera
    maxDepth = 10; % 10 meters
    mask = rawDepth > minDepth & rawDepth < maxDepth; % valid values
end