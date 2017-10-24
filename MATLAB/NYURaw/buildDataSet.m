load('Meta.mat');

for vid = 1:length(Meta)
   for frame = 1:Meta(vid).Frames
      index = sum([Meta(1:vid-1).Frames])+frame;
      
      try
        rgb = imread([Meta(vid).Vid, num2str(frame, '%06d'),'_rgb.png']);
        rgb = imresize(rgb,0.5);
        imwrite(rgb,['./DataSet/RGB/',num2str(index, '%06d'),'.png']);
      catch
        warning([Meta(vid).Vid, num2str(frame, '%06d'),'_rgb.png', ' NOT FOUND!']);
      end
      
      try
        norm = imread([Meta(vid).Vid, num2str(frame, '%06d'),'_norm.png']);
        mask = any(norm,3);
        mask = repmat(mask,[1,1,3]);
        norm = uint8((single(norm)/255)*254);
        norm(~mask) = 127;
        norm = imresize(norm,0.5);
        imwrite(norm,['./DataSet/NORM/',num2str(index, '%06d'),'.png']);
      catch
        warning([Meta(vid).Vid, num2str(frame, '%06d'),'_norm.png', ' NOT FOUND!']);
      end
      
   end
end

   