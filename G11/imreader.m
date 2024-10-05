function imageDataList = imreader(direc)
% IMREADER 数据为像素坐标
%   读取图片数据( pixel cordinate &RGB-Value H*W*C)，返回字典imageDataList
%   输入I  ：图片文件所在地址 path of the pictures,
%           字典对应i-th图片imageDataList{i}内存有像素+rgb-channel
    
    imageFiles = dir(fullfile(direc,'*.JPG'));
    
    % Init array to store image data
    imageDataList = {};

    % load image file recursively
    for i = 1:numel(imageFiles)
        % Read the image
        imagePath = fullfile(direc, imageFiles(i).name);
        image = imread(imagePath);

        % Append the image data to the list
        imageDataList{i} = image;
    end
   
end

