clc;
clear;
%% Incremental SFM Reconstruction
% Read all images using imageReader, replace with your own dataset directory when using
% imageReader从指定目录读取所有图像，使用时请将路径替换为您自己的数据集目录
image = imreader('D:\poster_conference_hall\images');

% Read camera intrinsic parameters using camera.txt reader
% 使用camera.txt读取相机的内参
camInfo = cameraReader('D:\poster_conference_hall\parameters\cameras.txt');

%% Define Camera Parameters
% Define camera intrinsic matrix
% 定义相机内参矩阵
IntrinMat = [camInfo(3), 0, camInfo(5);
             0, camInfo(4), camInfo(6);
             0, 0, 1];

% Define image size
% 定义图像尺寸
imageSize = [camInfo(1), camInfo(2)];  % 图像宽度和高度

% Create cameraParameters object
% 创建cameraParameters对象
cameraParams = cameraParameters('ImageSize', imageSize, 'K', IntrinMat);
intrinsics = cameraParams.Intrinsics;

%% Test (for convenience, can only use images 1 to 7 for detection)
% Convert images to grayscale and display the stitched image
% 将图像转换为灰度图并显示拼接后的图像
image = image(50:68);
% 创建一个大图并显示拼接后的图像
figure;
montage(image, 'Size', [3, 5]);
title('Montage of Images');

%% Create Connectivity Graph
% Create an empty view set
% 创建一个空的视图集
vSet = imageviewset;

% Create an empty cell array to store feature parameters
% 创建一个空的cell数组来存储特征参数
featuresPrev = cell(1, numel(image));

% Process the first image
% 对第一张图像进行处理
I = im2gray(image{1});
% Undistort the first image (optional, remove distortion)
% 对第一张图像进行去畸变处理（可选，去除畸变）
% I = undistortImage(I, intrinsics);

% Use SIFT feature detector to extract feature points
% 使用SIFT特征检测器提取特征点
points = detectSIFTFeatures(I);%这里面可以调整ContrastThreshold，EdgeThreshold参数
[features, points] = extractFeatures(I, points,"FeatureSize",64);
vSet = addView(vSet, 1, 'Features', features, 'Points', points);

% Iterate through the remaining images
% 遍历剩余的图像
featuresPrev{1} = features; % Save the features of the first image
% count = 0;
all_views_list = 1:numel(image);
for i = 2:numel(image)
    I = im2gray(image{i});
    % Undistort the image (optional, remove distortion)
    % 对图像进行去畸变处理（可选，去除畸变）
    % I = undistortImage(I, intrinsics);

    points = detectSIFTFeatures(I);
    [features, points] = extractFeatures(I, points,"FeatureSize",64);
    vSet = addView(vSet, i, 'Features', features, 'Points', points);
    for j = 1:i-1
        % Perform global matching and establish connections
        % 进行全局匹配并建立连接
        pairsIdx = matchFeatures(featuresPrev{j},features,MaxRatio=0.8,Unique=true,Method="Approximate",MatchThreshold=20);
        disp(['view',' ',num2str(i),' and  view',' ',num2str(j), ...
            ' matching is completed. The total number of matched point pairs is:',num2str(size(pairsIdx,1))])
        

        % Optimize the connectivity graph by removing views with insufficient matching points
        % 通过删除匹配点不足的视图来优化连接图
        if size(pairsIdx,1) >= 300 % Ensure each edge has a minimum of matches, this parameter can be adjusted
            vSet = addConnection(vSet,j,i,'Matches',pairsIdx);
        end
    end
    featuresPrev{i} = features; % Update the features of this image
end
select_view_list = unique([vSet.Connections.ViewId1(:),vSet.Connections.ViewId2(:)]);
% Determine if any images are completely removed from the connectivity graph during the optimization of connections.
diff_element = all_views_list(~ismember(all_views_list,select_view_list));  
if numel(diff_element)
    disp("views"+' '+num2str(diff_element)+' '+"was Removed due to insufficient matching points")
else

    disp("The optimization is complete. No views were deleted.")
end

% ob_gut = 2*count/(numel(image)*(numel(image)-1));

direct_B = 0;



% use for dense Point Cloud
% 用于建稠密点云使用
vSet_test = vSet;

%% Select an edge 'e' in the connectivity graph that is connected to the maximum number of tracks
% 选择连接图中与最多轨迹相关联的边'e'，先使用简单的指标：找到包含最多匹配点的边
connections = vSet.Connections;
match = connections.Matches;

% Initialize variables to store maximum matches and corresponding connection index
% 初始化最大匹配数和对应连接的索引
maxMatches = 0;
maxIndex = 0;

% Iterate through all connections
% 遍历所有连接
for i = 1:size(connections, 1)
    numMatches = numel(match{i,1});
    
    % Check if more matches are found
    % 检查是否找到更多的匹配点
    if numMatches > maxMatches
        maxMatches = numMatches;
        maxIndex = i;
    end
end

% Get the connection with the maximum matches as the initial edge 'e'
% 将具有最多匹配点的连接作为初始边'e'
maxConnection = connections(maxIndex,:);

%% Robustly estimate the essential matrix for the initial edge 'e' and obtain the corresponding extrinsics, then update them in the connectivity graph
% Obtain the corresponding point pairs data
% 使用鲁棒方法估计初始边'e'对应的本质矩阵，并获得相应的外参，然后更新到连接图中
% 获取对应点对的数据
PreFeaturePointset = vSet.Views.Points{maxConnection.ViewId1,1}.Location;
CurrFeaturePointset = vSet.Views.Points{maxConnection.ViewId2,1}.Location;

indexPairs = maxConnection.Matches{1,1};
    
% Find the specific matched points based on the indices
% 根据索引找到具体的匹配点
matchedPoints1 = PreFeaturePointset(indexPairs(:, 1),:);
matchedPoints2 = CurrFeaturePointset(indexPairs(:, 2),:);

% Estimate the camera pose of current view relative to the previous view.
% The pose is computed up to scale, meaning that the distance between
% the cameras in the previous view and the current view is set to 1.
% This will be corrected by the bundle adjustment.
% 估计当前视图相对于前一视图的相机位姿
% 该位姿计算为缩放后的值，即前一视图与当前视图之间的相机距离设置为1。
% 这将由捆绑调整进行校正。

[relPose, inlierIdx] = helperEstimateRelativePose(...
        matchedPoints1, matchedPoints2, intrinsics);

% Get the table containing the previous camera pose.
% 获取前一视图的相机位姿表
prevPose = poses(vSet, maxConnection.ViewId1).AbsolutePose;
        
% Compute the current camera pose in the global coordinate system 
% relative to the first view.
% 计算当前视图相对于第一视图的全局坐标系下的相机位姿
currPose = rigidtform3d(prevPose.A*relPose.A);
    
% Update the pose of the view in the connectivity graph
% 更新连接图中视图的位姿
vSet = updateView(vSet, maxConnection.ViewId2, currPose);
    
% Store the point matches between the previous and the current views.
% 存储前一视图与当前视图之间的点匹配关系
vSet = updateConnection(vSet, maxConnection.ViewId1, maxConnection.ViewId2, relPose, Matches=indexPairs(inlierIdx,:));

%% findTracks() - Find tracks containing three or more views (after the update)
% Since the graph has been updated, find all tracks again
% findTracks() - 找到包含三个或更多视图的轨迹
% 查找所有轨迹
tracks = findTracks(vSet);

% Find the indices of tracks with ViewIds element greater than or equal to 3
% 找到具有大于等于三个视图的轨迹索引
idxToKeep = cellfun(@numel, {tracks.ViewIds}) >= 3;

% Keep only the tracks with the specified indices
% 仅保留具有至少三个视图的轨迹
tracks = tracks(idxToKeep);

%% Triangulate points simultaneously present in both tracks and the edge 'e' 
% Find these points
% Get the view identifiers for the two views used previously
% 对同时存在于轨迹和边e中的点进行三角化
% 找到这些点
% 获取之前使用的两个视图的标识符
viewId1 = maxConnection.ViewId1;
viewId2 = maxConnection.ViewId2;

% Establish an array for recording camera poses with known poses
% 建立一个用于记录具有已知姿态相机的数组
worldviewId = [viewId1,viewId2];

% store the camera index that establishes the world coordinate
% 存储建立世界坐标的相机索引
VieId_first = viewId1;

% Initialize variables to store tracks containing the first and second images
% 初始化变量以存储包含第一张和第二张图的轨迹
matchingTracks = [];
tracks_first = findTracks(vSet,worldviewId);

%% Triangulate points simultaneously present in both tracks and the edge 'e' 
% Re-process the matchingTracks and create a vSet with only the first edge
% 对同时存在于轨迹和边e中的点进行三角化 
% 重新处理matchingTracks并创建仅包含第一条边的vSet
vSet_first = imageviewset;
vSet_first = addView(vSet_first, viewId1, 'Features', vSet.Views.Features{viewId1,1}, 'Points', vSet.Views.Points{viewId1,1});
vSet_first = addView(vSet_first, viewId2, 'Features', vSet.Views.Features{viewId2,1}, 'Points', vSet.Views.Points{viewId2,1});
vSet_first = addConnection(vSet_first,viewId1,viewId2,'Matches',maxConnection.Matches{1,1});

% Update the pose of the view in the connectivity graph
% 更新连接图中视图的姿态
vSet_first = updateView(vSet_first, maxConnection.ViewId2, currPose);
    
% Store the point matches between the previous and the current views.
% 存储前一视图和当前视图之间的点匹配
vSet_first = updateConnection(vSet_first, maxConnection.ViewId1, maxConnection.ViewId2, relPose, Matches=indexPairs(inlierIdx,:));

%% Perform initial triangulation

% Get the table containing camera poses for all views
% 执行初始三角化

% 获取包含所有视图相机姿态的表
camPoses = poses(vSet_first);

% Triangulate initial locations for the 3-D world points.
% 对于3D世界点，三角化初始位置.
[xyzPoints, errors,validvalue] = triangulateMultiview(tracks_first, camPoses, intrinsics);

% Filter out bad values
% Find the tracks and world coordinates that are useful
% This can also be adjusted based on the 'errors' value
% 过滤掉不好的值
% 找到有用的轨迹和世界坐标
% 这也可以根据'errors'值进行调整
xyzPoints = xyzPoints(errors<10,:);% parameter 'errors' can be adjusted
tracks_first = tracks_first(errors<10);
validvalue = validvalue(errors<10);
realxyzPoints = xyzPoints(validvalue == 1,:);
WorldTracks = tracks_first(validvalue == 1);

%% Delete the used connection
% 删除已使用的连接
% Delete the connection corresponding to the maximum number of matches
% 删除与最大匹配数对应的连接
vSet = deleteConnection(vSet,viewId1,viewId2);
disp('Completed the selection of the connection with the highest correlation and reconstructed the associated points in the world coordinate system.')
%% Entering the main loop
% 主循环（第一部分）
disp('Initiating the reconstruction of the remaining connections.')
disp('Reconstruction in progress...')

while numel(worldviewId)<(numel(image))

    % Mark for jump out of the loop
    if direct_B == 1
        break;
    end
    
    % Initialize variables for finding the next edge
    % 初始化下一边
    connections = vSet.Connections;
    maxConnection = [];
    maxtracksnum = 0;
    maxnum_points = 0;
    connections_index = 0;
    
    for numi = 1 : size(connections,1)
        % Get the view identifiers for the two views used previously
        viewId1 = connections(numi,:).ViewId1;
        viewId2 = connections(numi,:).ViewId2;
    
        % Check if at least one camera pose is known for the new connection, if not, skip this iteration
        % 检查新的连接中是否有只有一个图像的相机外参已知，如果没有或全已知，跳过此次迭代
        if (any(worldviewId == viewId1)||any(worldviewId == viewId2)) && (~(any(worldviewId == viewId1))||~(any(worldviewId == viewId2)))
            % Check the condition for viewId1
            % 检查条件是否满足viewId1
            a_viewId1 = any(worldviewId == viewId1);
            
    
            % Initialize variable to store tracks containing the new connection
            % 初始化用于存储包含新连接对应的轨迹的变量
            matchingTr = [];
            matchingTr = findTracks(vSet,[viewId1,viewId2]);
    
            % Initialize count variables
            % 初始化计数变量
            count = 0;
            imagePoints = [];
            sub_realxyzPoints = [];
    
            % Calculate overlap count
            % 计算重叠度     
            count = size(matchingTr,2);
    
            if count > maxtracksnum
                    connections_index = numi;
                    % Get the connection with the highest number of matches as the next edge to process, and store the corresponding image points and world coordinates
                    % Find the corresponding image points
                    % 获取具有最多匹配数的连接作为要处理的下一条边，并存储对应的图像点和世界坐标
                    % 查找对应的图像坐标
                
                for i = 1: size(WorldTracks,2)
                    for j = 1:size(matchingTr,2)
                      if a_viewId1
                            indices_old = find(ismember(WorldTracks(1,i).ViewIds, viewId1));
                            indices_new1 = find(ismember(matchingTr(1,j).ViewIds, viewId1));
                            indices_new2 = find(ismember(matchingTr(1,j).ViewIds, viewId2));
                            imagePoints_old = WorldTracks(1,i).Points(indices_old,:);
                            imagePoints_new1 = matchingTr(1,j).Points(indices_new1,:);
                            imagePoints_new2 = matchingTr(1,j).Points(indices_new2,:);

                            if imagePoints_old == imagePoints_new1
                                imagePoints = [imagePoints; imagePoints_new2,i];
                                sub_realxyzPoints = [sub_realxyzPoints;realxyzPoints(i,:)];
                            end

                     else
                            indices_old = find(ismember(WorldTracks(1,i).ViewIds, viewId2));
                            indices_new1 = find(ismember(matchingTr(1,j).ViewIds, viewId1));
                            indices_new2 = find(ismember(matchingTr(1,j).ViewIds, viewId2));
                            imagePoints_old = WorldTracks(1,i).Points(indices_old,:);
                            imagePoints_new1 = matchingTr(1,j).Points(indices_new1,:);
                            imagePoints_new2 = matchingTr(1,j).Points(indices_new2,:);

                            if imagePoints_old == imagePoints_new2
                                imagePoints = [imagePoints; imagePoints_new1,i];
                                sub_realxyzPoints = [sub_realxyzPoints;realxyzPoints(i,:)];
                            end

                     end
                    end
                end   
                if maxtracksnum < size(imagePoints,1) 
                    maxConnection = connections(connections_index,:);
                    matchingTracks = matchingTr;
                    tempo_imagePoints = imagePoints;
                    tempo_realxyzPoints = sub_realxyzPoints;
                    maxtracksnum =size(imagePoints,1);
                end
    
                
           end
        end
    end

    if isempty(maxConnection)
        disp(['The remaining images do not share common world coordinate points with the existing images, ' ...
         'Now, use the essential matrix to solve for 3D coordinates. ']);
        break;
    end

    % Estimate camera poses using PnP
    % Match the new images with the existing xyzPoints
    % 使用PnP法估计相机外参
    % 将新图像与现有的xyzPoints进行匹配
        
    viewId1 = maxConnection.ViewId1;
    viewId2 = maxConnection.ViewId2;
     
    % Check if the camera pose for viewId1 has been computed, if not, continue with the computation
    % 如果视图 viewId1 的相机位姿尚未计算，则继续计算
    if ~ismember(viewId1, worldviewId)
        worldviewId = [worldviewId viewId1];
        [worldPose,world_inlierIdx,Status] = estworldpose(tempo_imagePoints(:,1:2), tempo_realxyzPoints, intrinsics,MaxNumTrials=2000,Confidence=85,MaxReprojectionError=10);
        
        % Update the pose of the view
        % 更新视图的位姿
        vSet = updateView(vSet, viewId1, worldPose);
    
        % Update the views in vSet_first
        % 在 vSet_first 中更新视图
        vSet_first = addView(vSet_first, viewId1, 'Features', vSet.Views.Features{viewId1,1}, 'Points', vSet.Views.Points{viewId1,1});
        vSet_first = addConnection(vSet_first,viewId1,viewId2,'Matches',maxConnection.Matches{1,1});
        vSet_first = updateView(vSet_first, maxConnection.ViewId1, worldPose);
        disp(['Reconstruction of view ',num2str(viewId1),' is complete. There are ' ...
            , num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);
    end
    
    % Compute camera pose for viewId2
    % 计算视图 viewId2 的相机位姿
    if ~ismember(viewId2, worldviewId)
    
        worldviewId = [worldviewId viewId2];
        
        [worldPose,world_inlierIdx,Status] = estworldpose(tempo_imagePoints(:,1:2), tempo_realxyzPoints, intrinsics,MaxNumTrials=2000,Confidence=85,MaxReprojectionError=10);
        
        % Update the pose of the view
        % 更新视图的位姿
        vSet = updateView(vSet, viewId2, worldPose);
    
        % Update the views in vSet_first
        % 在 vSet_first 中更新视图
        vSet_first = addView(vSet_first, viewId2, 'Features', vSet.Views.Features{viewId2,1}, 'Points', vSet.Views.Points{viewId2,1});
        vSet_first = addConnection(vSet_first,viewId1,viewId2,'Matches',maxConnection.Matches{1,1});
        vSet_first = updateView(vSet_first, maxConnection.ViewId2, worldPose);
        disp(['Reconstruction of view ',num2str(viewId2),' is complete. There are ' ...
            , num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);
    
    end
        
    % Get the table containing camera poses for all views.
    % 获取包含所有视图相机位姿的表格
    camPoses = poses(vSet_first);
    
    % Update the matchingTracks for retriangulation
    % 更新用于三角测量的 matchingTracks
    matchingTracks = findTracks(vSet_first);
    
    % Perform new triangulation
    % 进行新的三角测量
    [xyzPoints, errors,validvalue] = triangulateMultiview(matchingTracks, camPoses, intrinsics);
    
    % Filter out points with validvalue equal to 0 and large errors as noise
    % 过滤掉 validvalue 等于 0 和误差较大的点作为噪声
    xyzPoints = xyzPoints(errors<100,:);% Adjust errors parameter
    matchingTracks = matchingTracks(errors<100);
    validvalue = validvalue(errors<100);
    realxyzPoints = [xyzPoints(validvalue == 1,:)];
    WorldTracks = [matchingTracks(validvalue == 1)];
    
    % Perform bundle adjustment
    % 进行束调整
    [realxyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
        realxyzPoints, WorldTracks, camPoses, intrinsics, FixedViewId=VieId_first, ...
        PointsUndistorted=true);
    
    % Find the useful tracks and corresponding world coordinates, and add them to the final realxyzPoints and WorldTracks
    % Exclude noisy 3D world points
    % 找到有用的轨迹和相应的世界坐标，并将它们添加到最终的 realxyzPoints 和 WorldTracks 中
    % 排除有噪声的 3D 世界点
    goodIdx = (reprojectionErrors < 200);% Adjust reprojectionErrors parameter
    % 调整 reprojectionErrors 参数
    
    realxyzPoints = realxyzPoints(goodIdx, :);
    WorldTracks = WorldTracks(1,goodIdx);
    
    % Update the camera poses3
    % 更新相机位姿
    vSet = updateView(vSet, camPoses);
    vSet_first = updateView(vSet_first, camPoses);
    
    % Delete the used connection
    % 删除已使用的连接
    vSet = deleteConnection(vSet,viewId1,viewId2);
    
    % End of loop1
    % 循环1结束
    
end
%% Estimating Camera Extrinsic Parameters using Essential Matrix (Second Loop)
% 利用求本质矩阵的方法求外参（第二步循环）
while numel(worldviewId)<(numel(image))
    
    % Instantiate the remaining connection
    % 实例化余留连接
    connections = vSet.Connections;
    
    % find ID of the remaining image
    % 找到余留图片的ID
    worldviewId_check = 1:numel(image);
    missingviewId = setdiff(worldviewId_check, worldviewId);
    count = 0;
    maxMatches = 0;
    Connection_index = 0;
    maxConnection = [];
    

    % Find the next edge to be used for creating 3D points
    % 寻找下一个边
    for numi = 1 : size(connections,1)
        % Get the viewIds of the two images used in the connection
        % 获取刚刚使用的两张图像的视图标识符
        viewId1 = connections(numi,:).ViewId1;
        viewId2 = connections(numi,:).ViewId2;

        % Check if the new connection has at least one image whose camera extrinsics are known
        % 判断新的connection中是否有一张图像所对应的相机外参已知
        if (any(worldviewId == viewId1)||any(worldviewId == viewId2)) && ((any(missingviewId == viewId1))||(any(missingviewId == viewId2)))
             
             a_viewId1 = any(worldviewId == viewId1);
             b_viewId2 = any(worldviewId == viewId2);
             % viewId2 corresponds to the ID of the image to be estimated
             % viewId2对应需要求的图像的ID

             count = size(connections.Matches{numi,1},1);
             if count > maxMatches
                 maxMatches = count;
                 maxConnection = connections(numi,:);
                 end_a_viewId1 = a_viewId1;
                 end_b_viewId2 = b_viewId2;
             end
                
       end
   end

   if isempty(maxConnection)
         % If unexpected situation occurs, exit the loop. 
         % Indicates that there are no connections available to estimate the camera extrinsics of the required image.
         % 出现意外情况直接跳出循环，表示不存在可用于求解所需图像外参的connections，放弃求解
        disp(['No connections available to estimate the camera extrinsics for image ', ...
            num2str(missingviewId') ,' insufficient association with other images']);
        break;
    end

    if end_a_viewId1 
        % Need to estimate the extrinsics for viewId2
        % 需要求解viewId2的外参
        viewId1 = maxConnection.ViewId1;
        viewId2 = maxConnection.ViewId2;
        worldviewId = [worldviewId viewId2];
        disp(['Reconstruction of view ',num2str(viewId2),' is complete. There are ', ...
            num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);

    else 
        % Need to estimate the extrinsics for viewId1
        % 需要求解viewId1的外参
        viewId2 = maxConnection.ViewId1;
        viewId1 = maxConnection.ViewId2;
        worldviewId = [worldviewId viewId2];
        disp(['Reconstruction of view ',num2str(viewId2),' is complete. There are ', ...
            num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);
        
    end
    
    % Get the data for the corresponding point pairs
    % 获取对应点对的数据
    PreFeaturePointset = vSet.Views.Points{viewId1,1}.Location;
    CurrFeaturePointset = vSet.Views.Points{viewId2,1}.Location;

    indexPairs = maxConnection.Matches{1,1};
    
    % Find the specific matched points based on the indices
    % 根据索引找到具体的匹配点
    if end_a_viewId1
        matchedPoints1 = PreFeaturePointset(indexPairs(:, 1),:);
        matchedPoints2 = CurrFeaturePointset(indexPairs(:, 2),:);
    else
        matchedPoints1 = PreFeaturePointset(indexPairs(:, 2),:);
        matchedPoints2 = CurrFeaturePointset(indexPairs(:, 1),:);
    end

    % Estimate the camera pose of current view relative to the previous view.
    % The pose is computed up to scale, meaning that the distance between
    % the cameras in the previous view and the current view is set to 1.
    % This will be corrected by the bundle adjustment.
    % 本质矩阵求解相机外参
    [relPose, inlierIdx] = helperEstimateRelativePose(...
            matchedPoints1, matchedPoints2, intrinsics);

    
    % Get the table containing the previous camera pose
    % 获取包含前一个相机姿态的表格
    prevPose = poses(vSet, maxConnection.ViewId1).AbsolutePose;
        
    % Compute the current camera pose in the global coordinate system 
    % relative to the first view.
    % 计算当前相机姿态在全局坐标系中相对于第一个视图的姿态
    currPose = rigidtform3d(prevPose.A*relPose.A);
    
    % Update the pose of the view
    % 更新视图的姿态
    vSet = updateView(vSet, viewId2, currPose);

    % Update the vSet_first view set
    % 更新视图集vSet_first
    vSet_first = addView(vSet_first, viewId2, 'Features', vSet.Views.Features{viewId2,1}, 'Points', vSet.Views.Points{viewId2,1});
    if end_a_viewId1
        vSet_first = addConnection(vSet_first,viewId1,viewId2,'Matches',maxConnection.Matches{1,1});
        % Update the pose of the view
        % 更新视图的姿态
        vSet_first = updateView(vSet_first, viewId2, currPose);
        disp(['Reconstruction of view ',num2str(viewId2),' is complete. There are ' ...
        , num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);
        % Delete the used connection
        % 删除已使用的边
        vSet = deleteConnection(vSet,viewId1,viewId2);  
    else
        vSet_first = addConnection(vSet_first,viewId2,viewId1,'Matches',maxConnection.Matches{1,1});
        % Update the pose of the view
        % 更新视图的姿态
        vSet_first = updateView(vSet_first, viewId2, currPose);
        disp(['Reconstruction of view ',num2str(viewId2),' is complete. There are ' ...
        , num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);
        % Delete the used connection
        % 删除现在所使用的边
        vSet = deleteConnection(vSet,viewId2,viewId1);
    end

    % Get the table containing camera poses for all views.
    camPoses = poses(vSet_first);

    % updates are now used to re-triangulate Tracks
    % 更新现在用来重新三角化的Tracks
    matchingTracks = findTracks(vSet_first);
    

    % triangulate the new Tracks
    % 三角化Tracks
    [xyzPoints, errors,validvalue] = triangulateMultiview(matchingTracks, camPoses, intrinsics);

    % Filter noise Points
    % 过滤噪点
    xyzPoints = xyzPoints(errors<100,:); % Errors can be adjusted
    matchingTracks = matchingTracks(errors<100);
    validvalue = validvalue(errors<100);
    realxyzPoints = [xyzPoints(validvalue == 1,:)];
    WorldTracks = [matchingTracks(validvalue == 1)];
    
    % BA Adjustment
    % 进行BA调整
    [realxyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
        realxyzPoints, WorldTracks, camPoses, intrinsics, FixedViewId=VieId_first, ...
        PointsUndistorted=true);

    % Exclude noisy 3-D world points.
    % 去除3-D的世界点
    goodIdx = (reprojectionErrors < 10);% reprojectionErrors
    
    realxyzPoints = realxyzPoints(goodIdx, :);
    WorldTracks = WorldTracks(1,goodIdx);

    % save the new cameraposes
    % 保留新的相机坐标
    vSet = updateView(vSet, camPoses);
    vSet_first = updateView(vSet_first, camPoses);
    
end
% End of loop2
% 循环2结束
disp('Congratulations！！！All views have been reconstructed！！！ ')
%% Plot Colored Sparse Point Cloud
%  绘制稀疏彩色点云

rgbMatrix = zeros(length(WorldTracks), 3);
for x = 1:length(WorldTracks)
    imageID = WorldTracks(1, x).ViewIds(1); % Find the ID of the first image in all indices
    % 找到所有索引第一张图片ID
    xyPoint = round(WorldTracks(1, x).Points(1, :)); % Extract the coordinates of the point in the first image corresponding to the track
    % 提取出track对应的第一张图片中点的坐标
    Photo = image{imageID};
    RGBinfo = Photo(xyPoint(2), xyPoint(1), :); % Extract the RGB information of the image
    % 提取出图片的RGB信息
    rgbMatrix(x, :) = squeeze(RGBinfo)'; % Return the RGB values of the point as a 1x3 row vector and save it in the first row of rgbMatrix
    % 将点的RGB三个值返回为1x3的行向量，保存在rgbMatrix的第一行
end
rgbMatrix = uint8(rgbMatrix);
ptCloud = pointCloud(realxyzPoints, 'Color', rgbMatrix);

% Display the refined camera poses.
% 显示优化后的相机位姿
figure;
plotCamera(camPoses, 'Size', 0.2);
hold on

% Display the Sparse 3-D world points.
% 显示稀疏的三维世界点云
pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', 'MarkerSize', 45);
grid on
hold off

% Specify the viewing volume.
% 指定视图范围
xlabel('X');
ylabel('Y');
zlabel('Z');
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-40, loc1(1)+40]);
ylim([loc1(2)-20, loc1(2)+20]);
zlim([loc1(3)-40, loc1(3)+40]);
camorbit(0, -30);
title('Sparse Colored Point Cloud')

%% Using the obtained camera extrinsics, reconstruct the world coordinates
% (including the residual tracks that were not utilized during reconstruction).
% 使用已求相机外参进行重建世界坐标（包括重建时未利用的剩余track）

% Get the table containing camera poses for all views.
% 获取包含所有视图中相机位姿的列表

camPoses = poses(vSet);

% 三角化新的Tracks
% triangulate the new Tracks
[xyzPoints, errors,validvalue] = triangulateMultiview(tracks, camPoses, intrinsics);


% Fliter the noise Points
% 过滤掉validvalue为0以及errors特别大的噪点值
xyzPoints = xyzPoints(errors<100,:);%errors parameters can be adjusted
tracks = tracks(errors<100);
validvalue = validvalue(errors<100);
realxyzPoints_2 = [xyzPoints(validvalue == 1,:)];
WorldTracks_2 = [tracks(validvalue == 1)];

% 进行BA调整
% BA adjustment
[realxyzPoints_2, ~, reprojectionErrors] = bundleAdjustment(...
    realxyzPoints_2, WorldTracks_2, camPoses, intrinsics, FixedViewId=VieId_first, ...
    PointsUndistorted=true);

%找到那部分真正有用的Tracks以及有用的世界坐标,并添加到最终的reakxyzPoints和WorldTracks
% Exclude noisy 3-D world points.
goodIdx = (reprojectionErrors < 10);%reprojectionErrors can be adjusted

realxyzPoints_2 = realxyzPoints_2(goodIdx, :);
WorldTracks_2 = WorldTracks_2(1,goodIdx);

% Plotting the colored images
% 绘制彩色图
rgbMatrix = zeros(length(WorldTracks_2), 3);
for x=1:length(WorldTracks_2)
    imageID=WorldTracks_2(1, x).ViewIds(1);
    xyPoint=round(WorldTracks_2(1, x).Points(1,:));
    Photo=image{imageID};
    RGBinfo=Photo(xyPoint(2),xyPoint(1),:);
    rgbMatrix(x, :) = squeeze(RGBinfo)';
end
rgbMatrix=uint8(rgbMatrix);
ptCloud_2 = pointCloud(realxyzPoints_2, 'Color', rgbMatrix);



% Replotting Colored Image
% 再次绘制彩色图
camPoses = poses(vSet);

% Display the refined camera poses.
% 展示优化的相机姿态
figure;
plotCamera(camPoses, Size=0.2);
hold on

% Display the dense 3-D world points.
% 展示密集3-D世界点
pcshow(ptCloud_2, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);
grid on
hold off

% Specify the viewing volume.
% 规定视图范围
xlabel('X');
ylabel('Y');
zlabel('Z');
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-40, loc1(1)+40]);
ylim([loc1(2)-20, loc1(2)+20]);
zlim([loc1(3)-40, loc1(3)+40]);
camorbit(0, -30);
title('Sparse Point Cloud');

%% Dense point clouds with low similarity and big angle changes
% 相似度大、视角变化小的密集点云
if direct_B == 0

    % Get the table containing camera poses for all views.

    camPoses = poses(vSet);

    track_test = findTracks(vSet_test);

    % triangulate the Track_test for the 3D world Points
    [xyzPoints, errors,validvalue] = triangulateMultiview(track_test, camPoses, intrinsics);

    
    xyzPoints = xyzPoints(errors<1000,:);
    track_test = track_test(errors<1000);
    validvalue = validvalue(errors<1000);
    realxyzPoints_3 = [xyzPoints(validvalue == 1,:)];
    WorldTracks_3 = [track_test(validvalue == 1)];

    % BA
    [realxyzPoints_3, camPoses, reprojectionErrors] = bundleAdjustment(...
        realxyzPoints_3, WorldTracks_3, camPoses, intrinsics, FixedViewId=VieId_first, ...
        PointsUndistorted=true);

    
    % Exclude noisy 3-D world points.
    goodIdx = (reprojectionErrors < 10);

    realxyzPoints_3 = realxyzPoints_3(goodIdx, :);
    WorldTracks_3 = WorldTracks_3(1,goodIdx);

    

    rgbMatrix = zeros(length(WorldTracks_3), 3);
    for x=1:length(WorldTracks_3)
        imageID=WorldTracks_3(1, x).ViewIds(1);% Find the ID of the first image in all indices
        xyPoint=round(WorldTracks_3(1, x).Points(1,:));% Extract the coordinates of the point in the first image corresponding to the track
        Photo=image{imageID};
        RGBinfo=Photo(xyPoint(2),xyPoint(1),:);% Extract the RGB information of the image
        rgbMatrix(x, :) = squeeze(RGBinfo)';% Return the RGB values of the point as a 1x3 row vector and save it in the first row of rgbMatrix
    end
    rgbMatrix=uint8(rgbMatrix);
    ptCloud_3 = pointCloud(realxyzPoints_3, 'Color', rgbMatrix);



    % get the pose of camera
    camPoses = poses(vSet);

    % Display the refined camera poses.
    figure;
    plotCamera(camPoses, Size=0.2);
    hold on


    % Display the dense 3-D world points.
    pcshow(ptCloud_3, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);
    grid on
    hold off

    % Specify the viewing volume.
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    loc1 = camPoses.AbsolutePose(1).Translation;
    xlim([loc1(1)-40, loc1(1)+40]);
    ylim([loc1(2)-20, loc1(2)+20]);
    zlim([loc1(3)-40, loc1(3)+40]);
    camorbit(0, -30);
    title('dense Point Cloud');

end

%% Dense point clouds with high similarity and low angle changes
% 相似度大、视角变化小的密集点云
if direct_B == 1
    grayImage1 = rgb2gray(image{VieId_first});

    % Detect corners in the first image.
    prevPoints = detectMinEigenFeatures(grayImage1, MinQuality=0.001);

    % Create the point tracker object to track the points across views.
    tracker = vision.PointTracker(MaxBidirectionalError=1, NumPyramidLevels=6);

    % Initialize the point tracker.
    prevPoints = prevPoints.Location;
    initialize(tracker, prevPoints, grayImage1);

    % Store the dense points in the view set.

    vSet_first = updateConnection(vSet_first, worldviewId(1), worldviewId(2), Matches=zeros(0, 2));
    vSet_first = updateView(vSet_first, 1, Points=prevPoints);

    % Track the points across all views.
    for i = 2:numel(image)
        grayImage = rgb2gray(image{i});
    
        % Track the points.
        [currPoints, validIdx] = step(tracker, grayImage);
    
        % Clear the old matches between the points.
        if i < 7
            vSet_first = updateConnection(vSet_first, i, i+1, Matches=zeros(0, 2));
        end
        vSet_first = updateView(vSet_first, i, Points=currPoints);
    
        % Store the point matches in the view set.
        matches = repmat((1:size(prevPoints, 1))', [1, 2]);
        matches = matches(validIdx, :);        
        vSet_first = updateConnection(vSet_first, i-1, i, Matches=matches);
    end

    % Find point tracks across all views.
    tracks_single = findTracks(vSet_first);

    % Find point tracks across all views.
    camPoses = poses(vSet_first);

    % triangulation
    [xyzPoints, errors,validvalue] = triangulateMultiview(tracks_single, camPoses, intrinsics);

   
    xyzPoints = xyzPoints(errors<15,:);
    tracks_single = tracks_single(errors<15);
    validvalue = validvalue(errors<15);
    realxyzPoints_3 = [xyzPoints(validvalue == 1,:)];
    WorldTracks_3 = [tracks_single(validvalue == 1)];


    [realxyzPoints_3, ~, reprojectionErrors] = bundleAdjustment(...
        realxyzPoints_3, WorldTracks_3, camPoses, intrinsics, FixedViewId=VieId_first, ...
        PointsUndistorted=true);

    
    % Exclude noisy 3-D world points.
    goodIdx = (reprojectionErrors < 5);%reprojectionErrors
    realxyzPoints_3 = realxyzPoints_3(goodIdx, :);
    WorldTracks_3 = WorldTracks_3(1,goodIdx);

    % plot the colored Image
    rgbMatrix = zeros(length(WorldTracks_3), 3);
    for x=1:length(WorldTracks_3)
        imageID=WorldTracks_3(1, x).ViewIds(1);
        xyPoint=round(WorldTracks_3(1, x).Points(1,:));
        Photo=image{imageID};
        RGBinfo=Photo(xyPoint(2),xyPoint(1),:);
        rgbMatrix(x, :) = squeeze(RGBinfo)';
    end
    rgbMatrix=uint8(rgbMatrix);
    ptCloud_3 = pointCloud(realxyzPoints_3, 'Color', rgbMatrix);



    % replotting the colored 
    camPoses = poses(vSet);

    % Display the refined camera poses.
    figure;
    plotCamera(camPoses, Size=0.2);
    hold on


    % Display the dense 3-D world points.
    pcshow(ptCloud_3, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);
    grid on
    hold off

    % Specify the viewing volume.
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    loc1 = camPoses.AbsolutePose(1).Translation;
    xlim([loc1(1)-40, loc1(1)+40]);
    ylim([loc1(2)-20, loc1(2)+20]);
    zlim([loc1(3)-40, loc1(3)+40]);
    camorbit(0, -30);
    title('dense point cloud');

end


%% Point Cloud Filtering
% 点云滤波
% the first round of denoising.
% 执行第一次去噪
ptCloudOut_1 = pcdenoise(ptCloud_3,'NumNeighbors',200,'Threshold',0.0000000001);
% the second round of denoising.
% 执行第二次去噪
[ptCloudOut_2,inlierIndices,outlierIndices] = pcdenoise(ptCloudOut_1,'NumNeighbors',200,'Threshold',0.0000000001);

cloud_inlier = select(ptCloudOut_1,inlierIndices); 


% the inlier point cloud after the second round of denoising.
% 第二次去噪后的内点
cloud_outlier = select(ptCloudOut_1,outlierIndices);  


% 提取外点点云并可视化
% Extract the outlier point cloud after the second round of denoising.
% 第二次去噪后的外点
camPoses = poses(vSet);

% Display the refined camera poses.
% 展示优化后相机位姿
figure;
plotCamera(camPoses, Size=0.2);
hold on

pcshow(cloud_inlier, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);
xlabel('X');
ylabel('Y');
zlabel('Z');
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-40, loc1(1)+40]);
ylim([loc1(2)-20, loc1(2)+20]);
zlim([loc1(3)-40, loc1(3)+40]);
camorbit(0, -30);
title('Filtered Point Cloud');

%% Perform Euclidean Clustering on Point Cloud
% 进行欧几里得聚类分割

% Parameter Settings
% 参数设置

minDistance = 0.45;  % Minimum Euclidean distance between two different clusters of points.
% 两个不同聚类团点之间的最小欧氏距离。
minPoints = 100;     % Minimum number of points in each cluster.
% 每个集群中的最小点数。
maxPoints = 100000; % Maximum number of points in each cluster.
% 每个集群中的最大点数

% Euclidean Clustering 
% 欧几里得聚类

[labels, numClusters] = pcsegdist(cloud_inlier, minDistance, ...
    'NumClusterPoints', [minPoints, maxPoints], ...
    'ParallelNeighborSearch', true);
% Save Segmentation Results
% 保存分割结果
for num=1:numClusters

    idxPoints = find(labels==num);        
    segmented = select(cloud_inlier,idxPoints);
    filename = strcat('European_cluster_',num2str(num),'.pcd');
    pcwrite(segmented,filename,'Encoding','binary'); 
end
% Color Rendering for Different Clusters%
% 对不同的类进行颜色渲染
idxValidPoints = find(labels);
labelColorIndex = labels(idxValidPoints);
segmentedPtCloud = select(cloud_inlier,idxValidPoints);
% Visualize Clustering Results%
% 可视化聚类结果

figure;
plotCamera(camPoses, 'Size', 0.2);
hold on
colormap(hsv(numClusters))
pcshow(segmentedPtCloud.Location,labelColorIndex,VerticalAxis='y', VerticalAxisDir='down');
xlabel('X');
ylabel('Y');
zlabel('Z');
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-40, loc1(1)+40]);
ylim([loc1(2)-20, loc1(2)+20]);
zlim([loc1(3)-40, loc1(3)+40]);
camorbit(0, -30);
title('Euclidean Clustering on Point Cloud')


%% OBB-like Bounding Box
% 类OBB包围盒
pc = {};
pos = {};
rotationMatrix={};
cornMatrix={};
Boxvolume=[];
Boxcoordinate=[];
for i = 1:numClusters
    pc{i} = pcread("European_cluster_" +string(i)+ ".pcd");

    % Get the maximum and minimum values of the point cloud in the XYZ directions
    % 获取点云在XYZ方向上的最大与最小值
    xMax=pc{i}.XLimits(2);
    yMax=pc{i}.YLimits(2);
    zMax=pc{i}.ZLimits(2);
    xMin=pc{i}.XLimits(1);
    yMin=pc{i}.YLimits(1);
    zMin=pc{i}.ZLimits(1);
    % Calculate the range of the point cloud in the XYZ directions
    % 计算点云xyz坐标的区间范围
    dx = xMax - xMin;
    dy = yMax - yMin;
    dz = zMax - zMin;
    % Visualization of Point Cloud and Bounding Boxes
    % 可视化点云和包围框
    
    % 定义一个长方体，并以不透明度为0.5的绿色显示它
    % Define a cuboid and display it in green with opacity 0.5.
    pos{i} = [xMin+dx/2, yMin+dy/2, zMin+dz/2,dx, dy, dz, 0 0 0];
    
    % 将点云的所有点投影到XZ平面
    % Project all points of the point cloud onto the XZ plane.
    projectedPoints = [pc{i}.Location(:,1), zeros(size(pc{i}.Location,1),1), pc{i}.Location(:,3)];

    % Normalize the main direction vector.
    % 归一化主方向向量
    centroid = mean(projectedPoints);


    centeredPoints = projectedPoints - centroid;
    
    % Compute the covariance matrix. 
    % 计算协方差矩阵
    covMatrix = cov(centeredPoints);
    
    % Compute the eigenvectors and eigenvalues of the covariance matrix.
    % 计算协方差矩阵的特征值和特征向量
    [eigenVectors, eigenValues] = eig(covMatrix);

    % Find the eigenvector corresponding to the largest eigenvalue as the main direction vector.
    % 找到最大特征值对应的特征向量作为主方向向量
    [~, maxIndex] = max(diag(eigenValues));
    mainDirection = eigenVectors(:, maxIndex);

    % Normalize the main direction vector.
    % 归一化主方向向量
    mainDirection = mainDirection / norm(mainDirection);
    
    cos_thetaz = dot(mainDirection, [0; 0;1]) / norm(mainDirection);
    if cos_thetaz < 0 
        cos_thetaz = -cos_thetaz;
    end
    thetaz = acos(cos_thetaz);

    cos_thetax = dot(mainDirection, [1; 0;0]) / norm(mainDirection);
    if cos_thetax < 0 
        cos_thetax = -cos_thetax;
    end
    thetax = acos(cos_thetax);
    if thetaz < thetax
        theta=thetaz;
    else
        theta=thetax;
    end
    
    rotationMatrix= [cos(theta)  0  sin(theta);
      0         1     0;
     -sin(theta) 0  cos(theta)];
    rt=[0 0 0 0;0 0 0 0;0 0 0 0;0 0 0 1];
    rt(1:3,1:3)=rotationMatrix;
    rotationMatrix=rt;

    pos{i} = [xMin+dx/2, yMin+dy/2, zMin+dz/2,dx, dy, dz, 0 0 0];
    % Compute the coordinates of the eight corners of the bounding box.
    % 得出包围盒八个角点坐标
    centerCoord = pos{i}(1:3);
    dx = pos{i}(4);
    dy = pos{i}(5);
    dz = pos{i}(6);
    % compute the center location and volume of the box
    Boxvolume = [Boxvolume;dx * dy * dz];
    Boxcoordinate = [Boxcoordinate;[dx dy dz]];
 
    cornMatrix0=[xMin+dx/2 yMin+dy/2 zMin+dz/2];
    cornMatrix{i} = repmat(cornMatrix0, 8, 1);
   
    corners = [
    centerCoord(1)-dx/2, centerCoord(2)-dy/2, centerCoord(3)-dz/2;  % 左下后
    % Bottom left back
    centerCoord(1)-dx/2, centerCoord(2)-dy/2, centerCoord(3)+dz/2;  % 左下前
    % Bottom left front
    centerCoord(1)-dx/2, centerCoord(2)+dy/2, centerCoord(3)+dz/2;  % 左上前
    % Top left front
    centerCoord(1)-dx/2, centerCoord(2)+dy/2, centerCoord(3)-dz/2;  % 左上后
    % Top left back
    centerCoord(1)+dx/2, centerCoord(2)-dy/2, centerCoord(3)-dz/2;  % 右下后
    % Bottom right back
    centerCoord(1)+dx/2, centerCoord(2)-dy/2, centerCoord(3)+dz/2;  % 右下前
    % Bottom right front
    centerCoord(1)+dx/2, centerCoord(2)+dy/2, centerCoord(3)+dz/2;  % 右上前
    % Top right front
    centerCoord(1)+dx/2, centerCoord(2)+dy/2, centerCoord(3)-dz/2   % 右上后
    % Top right back
    ];
    corners=corners-cornMatrix{i};
    pccloudcorners = pointCloud(corners);
   
    
    tform = affine3d(rotationMatrix);
    invcorners{i} = pctransform(pccloudcorners, tform);
    
end
 
Boxinfo = [Boxcoordinate,Boxvolume];
% 定义包围盒的六个面，每个面由四个顶点构成
% Define the faces of the bounding box, each face consisting of four vertices.
faces = [
    1, 2, 3, 4;    % 左侧面
    % Left face
    5, 6, 7, 8;    % 右侧面
    % Right face
    1, 2, 6, 5;    % 底面
    % Bottom face
    2, 3, 7, 6;    % 前面
    % Front face
    3, 4, 8, 7;    % 顶面
    % Top face
    4, 1, 5, 8     % 后面
    % Back face
];

% Display the bounding boxes.
% 显示出包围盒
figure;
hold on;
for i = 1:numClusters
    
    
    patch('Vertices', invcorners{i}.Location+cornMatrix{i}, 'Faces', faces, 'FaceColor', 'g', 'FaceAlpha', 0.2);
    text(cornMatrix{i}(1,1), cornMatrix{i}(1,2), cornMatrix{i}(1,3), num2str(i), 'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold');
end

figure;
plotCamera(camPoses, 'Size', 0.2);
hold on
colormap(hsv(numClusters))
pcshow(segmentedPtCloud.Location,labelColorIndex,VerticalAxis='y', VerticalAxisDir='down');

xlabel('X');
ylabel('Y');
zlabel('Z');
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-40, loc1(1)+40]);
ylim([loc1(2)-20, loc1(2)+20]);
zlim([loc1(3)-40, loc1(3)+40]);
camorbit(0, -30);
hold off;
title('OBB bounding box');

%% Calculate the distance between the centroids of different bounding boxes
% 计算不同包围盒之间质心的距离

Boxcoornum = size(Boxcoordinate, 1);  % number of Points

distances = zeros(Boxcoornum);  % initial distance matrix

for i = 1:Boxcoornum
    for j = 1:Boxcoornum
        Box2Boxdis(i, j) = norm(Boxcoordinate(i, :) - Boxcoordinate(j, :));  % compute the Euclidean distance

    end
end


%% 点云文件存为.ply

pcwrite(ptCloud,'delivery_area_sparse.ply','Encoding','ascii');
pcwrite(ptCloud_2,'delivery_area_sparse_render.ply','Encoding','ascii');
pcwrite(ptCloud_3,'delivery_area_dense.ply','Encoding','ascii');
pcwrite(ptCloudOut_2,'delivery_area_sparse_fliter.ply','Encoding','ascii');





