function varargout = CV_GUI(varargin)
% CV_GUI MATLAB code for CV_GUI.fig
%      CV_GUI, by itself, creates a new CV_GUI or raises the existing
%      singleton*.
%
%      H = CV_GUI returns the handle to a new CV_GUI or the handle to
%      the existing singleton*.
%
%      CV_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CV_GUI.M with the given input arguments.
%
%      CV_GUI('Property','Value',...) creates a new CV_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CV_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CV_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CV_GUI

% Last Modified by GUIDE v2.5 11-Jul-2023 15:39:43

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CV_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @CV_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CV_GUI is made visible.
function CV_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CV_GUI (see VARARGIN)

% Choose default command line output for CV_GUI
handles.output = hObject;

handles.progressBar = uicontrol('Style','text','Position',[20,20,450,20],'BackgroundColor',[0.94, 0.94, 0.94]);

set(handles.displayText, 'String', 'Hello World:) Please select the pictures and camera parameters.');


    % Initialization: By default, choose big scene
set(handles.radiobutton_small, 'Value', 0);
set(handles.radiobutton_big, 'Value', 1);
handles.direct_B = 0; % correspondingly, direct_B is 0



guidata(hObject, handles);
% Update handles structure
guidata(hObject, handles);
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);
% UIWAIT makes CV_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CV_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filenames, path] = uigetfile({'*.jpg;*.png'},'Select Image Files','MultiSelect','on');

if isequal(filenames,0)
    disp('User selected Cancel');
else
    % Create cell array of full file paths
    if iscell(filenames)
        fullpaths = cellfun(@(name) fullfile(path, name), filenames, 'UniformOutput', false);
    else
        fullpaths = {fullfile(path, filenames)};
    end
    % Store full file paths in handles structure
    handles.selectedImages = fullpaths;
    guidata(hObject, handles); % Save handles structure
end


handles = guidata(hObject);
selectedImages = handles.selectedImages;

numImages = numel(selectedImages);
numCols = 4;  % adjust as needed
numRows = ceil(numImages / numCols);


% Calculate the width and height of each subplot
subplotWidth = 1 / numCols;  % 100% of the total width, adjust as needed
subplotHeight = 1 / numRows;  % 100% of the total height, adjust as needed

% Load and display each image
for k = 1:numImages
    % Calculate the position of the subplot
    row = floor((k-1) / numCols);
    col = mod(k-1, numCols);
    left = col * subplotWidth+0.001; 
    bottom = 1 - (row+1) * subplotHeight+0.001; 
    position = [left bottom subplotWidth subplotHeight];

    % Load the image
    img = imread(selectedImages{k});

    % Create the subplot
    subplot('Position', position, 'Parent', handles.panel1);

    % Display the image
    imshow(img);
end

set(handles.displayText, 'String', 'Picture selected');



% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, ~, ~)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles = guidata(hObject);
imds = imageDatastore(handles.selectedImages);

set(handles.displayText, 'String', 'Running......');
fx = str2double(get(handles.edit1, 'String'));
fy = str2double(get(handles.edit2, 'String'));
cx = str2double(get(handles.edit3, 'String'));
cy = str2double(get(handles.edit4, 'String'));
img_width = str2double(get(handles.edit5, 'String'));
img_height = str2double(get(handles.edit6, 'String'));


IntrinMat = [fx, 0, cx;
             0, fy, cy;
             0, 0, 1];

% height and weight
imageSize = [img_width, img_height];  

% creat cameraParameters object
cameraParams = cameraParameters('ImageSize', imageSize, 'K', IntrinMat);
intrinsics = cameraParams.Intrinsics;


image = cell(1, numel(imds.Files));

for i = 1:numel(imds.Files)
    image{i} = readimage(imds, i);
end



   
%%
set(handles.progressBar,'Position',[20,20,37.5,20],'BackgroundColor',[0 1 0]);
    
    % run GUI
    drawnow;

%%
% Get intrinsic parameters of the camera
vSet = imageviewset;


featuresPrev = image;
all_views_list = 1:numel(image);

I = im2gray(image{1});
% Undistort the first image.
%I = undistortImage(I, intrinsics); 

% Use SIFT feature detector to extract feature points
points = detectSIFTFeatures(I);
[features, points] = extractFeatures(I, points,"FeatureSize",64);
vSet = addView(vSet, 1, 'Features', features, 'Points', points);

% Iterate through the remaining images
% 遍历剩余的图像
featuresPrev{1} = features; 

count = 0;

for i = 2:numel(image)
    I = im2gray(image{i});
    % Undistort the first image.
    %I = undistortImage(I, intrinsics); 

    points = detectSIFTFeatures(I);
    [~, points] = extractFeatures(I, points,"FeatureSize",64);
    features = extractFeatures(I, points, Upright=true);
    vSet = addView(vSet, i, 'Features', features, 'Points', points);
    for j = 1:i-1
        % Perform global matching and establish connections
        % 进行全局匹配并建立连接
        pairsIdx = matchFeatures(featuresPrev{j},features,MaxRatio=0.6,Unique=true);

        set(handles.displayText, 'String',  ['view',' ',num2str(i),' and  view',' ',num2str(j), ...
            ' matching is completed. The total number of matched point pairs is:',num2str(size(pairsIdx,1))]);
       
        % counter
        if numel(pairsIdx) >= 3500
            count = count +1;
        end

        % Optimize the connectivity graph by removing views with insufficient matching points
        % 通过删除匹配点不足的视图来优化连接图
        if numel(pairsIdx) >= 200 %%确保每条边是大于等于100的，该参数可调
            vSet = addConnection(vSet,j,i,'Matches',pairsIdx);
        end
    end
    featuresPrev{i} = features; % 更新这一张图像的特征
end
select_view_list = unique([vSet.Connections.ViewId1(:),vSet.Connections.ViewId2(:)]);
% Determine if any images are completely removed from the connectivity graph during the optimization of connections.
diff_element = all_views_list(~ismember(all_views_list,select_view_list));  
if numel(diff_element)
    set(handles.displayText,'String',"views"+' '+num2str(diff_element)+' '+ "was Removed due to insufficient matching points")
   
else
    set(handles.displayText,'String', "The optimization is complete. No views were deleted.")    
end

ob_gut = 2*count/(numel(image)*(numel(image)-1));

direct_B = 0;

% 判断所有图的重合程度
% degree of correspondence
direct_B = handles.direct_B;


% use for dense Point Cloud
% 用于建稠密点云使用
vSet_test = vSet;



%%
set(handles.progressBar,'Position',[20,20,75,20],'BackgroundColor',[0 1 0]);
    
    
    drawnow;


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


%%
set(handles.progressBar,'Position',[20,20,112.5,20],'BackgroundColor',[0 1 0]);
    
    
    drawnow;

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


%%
set(handles.progressBar,'Position',[20,20,150,20],'BackgroundColor',[0 1 0]);
    
    
    drawnow;
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


%%
set(handles.progressBar,'Position',[20,20,187.5,20],'BackgroundColor',[0 1 0]);
    
    
    drawnow;
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
xyzPoints = xyzPoints(errors<100,:);
tracks_first = tracks_first(errors<100);
validvalue = validvalue(errors<100);
realxyzPoints = xyzPoints(validvalue == 1,:);
WorldTracks = tracks_first(validvalue == 1);

%% Delete the used connection
% 删除已使用的连接
% Delete the connection corresponding to the maximum number of matches
% 删除与最大匹配数对应的连接
vSet = deleteConnection(vSet,viewId1,viewId2);
set(handles.displayText,'String', ['Completed the selection of the connection with ' ...
    'the highest correlation and reconstructed the associated points in the world coordinate system.'])
set(handles.displayText,'String', 'Initiating the reconstruction of the remaining connections.')
set(handles.displayText,'String', 'Reconstruction in progress...')

%%
set(handles.progressBar,'Position',[20,20,225,20],'BackgroundColor',[0 1 0]);
    
    
    drawnow;

%% Entering the main loop
% 主循环（第一部分）
while numel(worldviewId)<(numel(image))
    % Mark for jump out of the loop
    if direct_B == 1
        break;
    end
    
    connections = vSet.Connections;
    % Initialize variables for finding the next edge
    % 获取刚刚使用的两张图的视图标识符 
    maxConnection = [];
    maxtracksnum = 0;
    maxnum_points = 0;
    connections_index = 0;
    
  
    for numi = 1 : size(connections,1)
        % Get the view identifiers for the two views used previously
        % 获取刚刚使用的两张图的视图标识符
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
            matchingTr = findTracks(vSet,[viewId1,viewId2]);
    
            % Initialize count variables
            % 初始化计数变量
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
        
        set( handles.displayText, 'String', ['The remaining images do not share common world coordinate points with the existing images, ' ...
             'Now, use the essential matrix to solve for 3D coordinates. ']);
        break;
    end
    
    
    
    %%
    set(handles.progressBar,'Position',[20,20,262.5,20],'BackgroundColor',[0 1 0]);
        
        % update GUI
        drawnow;
    %%
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
        
        [worldPose,~ ,~ ] = estworldpose(tempo_imagePoints(:,1:2), tempo_realxyzPoints, intrinsics,MaxNumTrials=2000,Confidence=85,MaxReprojectionError=10);
        
        % Update the pose of the view
        % 更新视图的位姿
        vSet = updateView(vSet, viewId1, worldPose);
    
        % Update the views in vSet_first
        % 在 vSet_first 中更新视图
        vSet_first = addView(vSet_first, viewId1, 'Features', vSet.Views.Features{viewId1,1}, 'Points', vSet.Views.Points{viewId1,1});
        vSet_first = addConnection(vSet_first,viewId1,viewId2,'Matches',maxConnection.Matches{1,1});
        
        
        vSet_first = updateView(vSet_first, maxConnection.ViewId1, worldPose);
        set(handles.displayText,'String', ['Reconstruction of view ',num2str(viewId2),' is complete. There are ' ...
                , num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.'])
       
    
    end
    %
    
    %%
    set(handles.progressBar,'Position',[20,20,300,20],'BackgroundColor',[0 1 0]);
        
        % update GUI
        drawnow;
    
    % Compute camera pose for viewId2
    % 计算视图 viewId2 的相机位姿
    if ~ismember(viewId2, worldviewId)
    
        
        worldviewId = [worldviewId viewId2];
        
        [worldPose,~,~] = estworldpose(tempo_imagePoints(:,1:2), tempo_realxyzPoints, intrinsics,MaxNumTrials=2000,Confidence=85,MaxReprojectionError=10);
        
        % Update the pose of the view
        % 更新视图的位姿
        vSet = updateView(vSet, viewId2, worldPose);
    
        % Update the views in vSet_first
        % 在 vSet_first 中更新视图
        vSet_first = addView(vSet_first, viewId2, 'Features', vSet.Views.Features{viewId2,1}, 'Points', vSet.Views.Points{viewId2,1});
        vSet_first = addConnection(vSet_first,viewId1,viewId2,'Matches',maxConnection.Matches{1,1});
        % Get the table containing camera poses for all views.
        % 获取包含所有视图相机位姿的表格
        vSet_first = updateView(vSet_first, maxConnection.ViewId2, worldPose);
        set(handles.displayText,'String', ['Reconstruction of view ',num2str(viewId2),' is complete. There are ' ...
                , num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.'])
    
    
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
    xyzPoints = xyzPoints(errors<1000,:);
    matchingTracks = matchingTracks(errors<1000);
    validvalue = validvalue(errors<1000);
    realxyzPoints = [xyzPoints(validvalue == 1,:)];
    WorldTracks = [matchingTracks(validvalue == 1)];
    
    % BA Adjustment
    % 进行BA调整
    [realxyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
        realxyzPoints, WorldTracks, camPoses, intrinsics, FixedViewId=VieId_first, ...
        PointsUndistorted=true);
    
    % Exclude noisy 3-D world points.
    % 去除3-D的世界点
    goodIdx = (reprojectionErrors < 200);%reprojectionErrors参数可调整
    
    realxyzPoints = realxyzPoints(goodIdx, :);
    WorldTracks = WorldTracks(1,goodIdx);
    
    % save the new cameraposes
    % 保留新的相机坐标
    vSet = updateView(vSet, camPoses);
    vSet_first = updateView(vSet_first, camPoses);
    vSet = deleteConnection(vSet,viewId1,viewId2);
    
    % End of loop2
    % 循环2结束

end

%%
set(handles.progressBar,'Position',[20,20,337.5,20],'BackgroundColor',[0 1 0]);
    
    
    drawnow;


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
    maxMatches = 0;
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
             % viewId2 corresponds to the ID of the image to be estimated
             % viewId2对应需要求的图像的ID
            count = size(connections(numi,:).Matches{1,1},1);
            if count > maxMatches
                maxMatches = count;
                maxConnection = connections(numi,:);
                end_a_viewId1 = a_viewId1;
            end
                
        end
    end

    if isempty(maxConnection) 
         % If unexpected situation occurs, exit the loop. 
         % Indicates that there are no connections available to estimate the camera extrinsics of the required image.
         % 出现意外情况直接跳出循环，表示不存在可用于求解所需图像外参的connections，放弃求解
        disp(['此时不存在对应可以用来求图' num2str(missingviewId) '外参的connections,他们跟其他图的关联性不够好']);
        message = ['此时不存在对应可以用来求图' num2str(missingviewId) '外参的connections,他们跟其他图的关联性不够好'];
        set(handles.displayText, 'String', message);
        break;
    end

    if end_a_viewId1 
        % Need to estimate the extrinsics for viewId2
        % 需要求解viewId2的外参
        viewId1 = maxConnection.ViewId1;
        viewId2 = maxConnection.ViewId2;
        worldviewId = [worldviewId viewId2];
        set(handles.displayText, 'String',['Reconstruction of view ',num2str(viewId2),' is complete. There are ', ...
            num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);

     
    else         
        % Need to estimate the extrinsics for viewId1
        % 需要求解viewId1的外参
        viewId2 = maxConnection.ViewId1;
        viewId1 = maxConnection.ViewId2;
        worldviewId = [worldviewId viewId2];
        set(handles.displayText, 'String', ['Reconstruction of view ',num2str(viewId2),' is complete. There are ', ...
            num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);
    end


 %%
set(handles.progressBar,'Position',[20,20,375,20],'BackgroundColor',[0 1 0]);
    
    % apdate GUI
    drawnow;



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

    [relPose, ~] = helperEstimateRelativePose(...
            matchedPoints1, matchedPoints2, intrinsics);

    % Get the table containing the previous camera pose
    % 获取包含前一个相机姿态的表格
    prevPose = poses(vSet, viewId1).AbsolutePose;
        
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
        set(handles.displayText, 'String', ['Reconstruction of view ',num2str(viewId2),' is complete. There are ' ...
        , num2str((numel(image) - numel(diff_element))-numel(worldviewId)) ,' remaining views to be reconstructed.']);
        % Delete the used connection
        % 删除已使用的边
        vSet = deleteConnection(vSet,viewId1,viewId2);  
    else
        vSet_first = addConnection(vSet_first,viewId2,viewId1,'Matches',maxConnection.Matches{1,1});
        % Update the pose of the view
        % 更新视图的姿态
        vSet_first = updateView(vSet_first, viewId2, currPose);
        set(handles.displayText, 'String', ['Reconstruction of view ',num2str(viewId2),' is complete. There are ' ...
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
    xyzPoints = xyzPoints(errors<1000,:);% Errors can be adjusted
    matchingTracks = matchingTracks(errors<1000);
    validvalue = validvalue(errors<1000);
    realxyzPoints = [xyzPoints(validvalue == 1,:)];
    WorldTracks = [matchingTracks(validvalue == 1)];

    % BA Adjustment
    % 进行BA调整
    [realxyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
        realxyzPoints, WorldTracks, camPoses, intrinsics, FixedViewId=VieId_first, ...
        PointsUndistorted=true);

    % Exclude noisy 3-D world points.
    % 去除3-D的世界点
    goodIdx = (reprojectionErrors < 1000);%reprojectionErrors参数可调整

    realxyzPoints = realxyzPoints(goodIdx, :);
    WorldTracks = WorldTracks(1,goodIdx);

    % save the new cameraposes
    % 保留新的相机坐标
    vSet = updateView(vSet, camPoses);
    vSet_first = updateView(vSet_first, camPoses);
    
end
% End of loop2
% 循环2结束
set(handles.displayText, 'String', 'Congratulations！！！All views have been reconstructed！！！ ');

%%
set(handles.progressBar,'Position',[20,20,412.5,20],'BackgroundColor',[0 1 0]);
    

    drawnow;

set(handles.displayText, 'String', 'start drawing......');
%% Plot Colored Sparse Point Cloud
%  绘制稀疏彩色点云

rgbMatrix = zeros(length(WorldTracks), 3);
for x=1:length(WorldTracks)
    imageID=WorldTracks(1, x).ViewIds(1);% Find the ID of the first image in all indices
    % 找到所有索引第一张图片ID
    xyPoint=round(WorldTracks(1, x).Points(1,:));% Extract the coordinates of the point in the first image corresponding to the track
    % 提取出track对应的第一张图片中点的坐标
    Photo=image{imageID};
    RGBinfo=Photo(xyPoint(2),xyPoint(1),:);% Extract the RGB information of the image
    % 提取出图片的RGB信息
    rgbMatrix(x, :) = squeeze(RGBinfo)';% Return the RGB values of the point as a 1x3 row vector and save it in the first row of rgbMatrix
    % 将点的RGB三个值返回为1x3的行向量，保存在rgbMatrix的第一行
end
rgbMatrix=uint8(rgbMatrix);
ptCloud = pointCloud(realxyzPoints, 'Color', rgbMatrix);


set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);
camPoses = poses(vSet);
axes(handles.axes2);
% Display the refined camera poses.
% 显示优化后的相机位姿

plotCamera(camPoses, Size=0.2);
hold on


% Display the Sparse 3-D world points.
% 显示稀疏的三维世界点云
pcshow(ptCloud, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);
grid on
hold off

% Specify the viewing volume.
% 指定视图范围
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-40, loc1(1)+40]);
ylim([loc1(2)-40, loc1(2)+40]);
zlim([loc1(3)-30, loc1(3)+30]);
camorbit(0, -30);
title('Sparse Colored Point Cloud');
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);
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
xyzPoints = xyzPoints(errors<1000,:);%errors parameters can be adjusted
tracks = tracks(errors<1000);
validvalue = validvalue(errors<1000);
realxyzPoints_2 = [xyzPoints(validvalue == 1,:)];
WorldTracks_2 = [tracks(validvalue == 1)];

% 进行BA调整
% BA adjustment
[realxyzPoints_2, ~, reprojectionErrors] = bundleAdjustment(...
    realxyzPoints_2, WorldTracks_2, camPoses, intrinsics, FixedViewId=VieId_first, ...
    PointsUndistorted=true);

%找到那部分真正有用的Tracks以及有用的世界坐标,并添加到最终的reakxyzPoints和WorldTracks
% Exclude noisy 3-D world points.
goodIdx = (reprojectionErrors < 200);%reprojectionErrors can be adjusted

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
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);

% Replotting Colored Image
% 再次绘制彩色图
camPoses = poses(vSet);
axes(handles.axes3);

% Display the refined camera poses.
% 展示优化的相机姿态
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
ylim([loc1(2)-40, loc1(2)+40]);
zlim([loc1(3)-30, loc1(3)+30]);
camorbit(0, -30);
title('Sparse Colored Point Cloud');
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);
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
    [realxyzPoints_3, ~, reprojectionErrors] = bundleAdjustment(...
    realxyzPoints_3, WorldTracks_3, camPoses, intrinsics, FixedViewId=VieId_first, ...
    PointsUndistorted=true);

    % Exclude noisy 3-D world points.
    goodIdx = (reprojectionErrors < 200);%reprojectionErrors参数可调整

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


    set(handles.figure1, 'Color', [1, 1, 1]);
    camPoses = poses(vSet);
    axes(handles.axes4);
    % Display the refined camera poses.

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
    ylim([loc1(2)-40, loc1(2)+40]);
    zlim([loc1(3)-30, loc1(3)+30]);
    camorbit(0, -30);
    title('dense Point Cloud');
    set(handles.figure1, 'Color', [1, 1, 1]);
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
        if i < numel(image)
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

    
    [xyzPoints, errors,validvalue] = triangulateMultiview(tracks_single, camPoses, intrinsics);

    
    xyzPoints = xyzPoints(errors<1000,:);%reprojectionErrors
    tracks_single = tracks_single(errors<1000);
    validvalue = validvalue(errors<1000);
    realxyzPoints_3 = [xyzPoints(validvalue == 1,:)];
    WorldTracks_3 = [tracks_single(validvalue == 1)];

    
    [realxyzPoints_3, ~, reprojectionErrors] = bundleAdjustment(...
        realxyzPoints_3, WorldTracks_3, camPoses, intrinsics, FixedViewId=VieId_first, ...
        PointsUndistorted=true);

    %找到那部分真正有用的Tracks以及有用的世界坐标,并添加到最终的reakxyzPoints和WorldTracks
    % Exclude noisy 3-D world points.
    goodIdx = (reprojectionErrors < 100);%reprojectionErrors参数可调整

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


    set(handles.figure1, 'Color', [1, 1, 1]);
    % replotting the colored 
    camPoses = poses(vSet);
    axes(handles.axes4);
    % Display the refined camera poses.

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
    ylim([loc1(2)-40, loc1(2)+40]);
    zlim([loc1(3)-30, loc1(3)+30]);
    camorbit(0, -30);
    title('dense point cloud');
    set(handles.figure1, 'Color', [1, 1, 1]);

end


%% Point Cloud Filtering
% 点云滤波
% the first round of denoising.
% 执行第一次去噪
ptCloudOut_1 = pcdenoise(ptCloud_3,'NumNeighbors',200,'Threshold',0.0000000001);
% the second round of denoising.
% 执行第二次去噪
[~,inlierIndices,outlierIndices] = pcdenoise(ptCloudOut_1,'NumNeighbors',200,'Threshold',0.0000000001);

cloud_inlier = select(ptCloudOut_1,inlierIndices); % the inlier point cloud after the second round of denoising.
% 第二次去噪后的内点




% 提取外点点云并可视化
% Extract the outlier point cloud after the second round of denoising.

cloud_outlier = select(ptCloudOut_1,outlierIndices);  
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);
camPoses = poses(vSet);
% Display the refined camera poses.
% 展示优化后相机位姿
axes(handles.axes5);
plotCamera(camPoses, Size=0.2);
hold on

pcshow(cloud_inlier, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);
loc1 = camPoses.AbsolutePose(1).Translation;
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([loc1(1)-40, loc1(1)+40]);
ylim([loc1(2)-40, loc1(2)+40]);
zlim([loc1(3)-30, loc1(3)+30]);
camorbit(0, -30);
title('Filtered Point Cloud');
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);
%% Perform Euclidean Clustering on Point Cloud
% 进行欧几里得聚类分割

% Parameter Settings
% 参数设置
minDistance = 1.5;  % Minimum Euclidean distance between two different clusters of points.
% 两个不同聚类团点之间的最小欧氏距离。
minPoints = 100;     % Minimum number of points in each cluster.
% 每个集群中的最小点数。
maxPoints = 100000; % Maximum number of points in each cluster.
% Euclidean Clustering 
% 欧几里得聚类
[labels,numClusters] = pcsegdist(cloud_inlier,minDistance, ...
    'NumClusterPoints',[minPoints,maxPoints], ...
    'ParallelNeighborSearch',true);
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
axes(handles.axes6)



colormap(hsv(numClusters))
plotCamera(camPoses, Size=0.2);
hold on
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
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);
%% OBB-like Bounding Box
% 类OBB包围盒
pc = {};
pos = {};

cornMatrix={};
Boxvolume=[];
Boxcoordinate=[];
for i = 1:numClusters
    pc{i} = pcread("European_cluster_" +string(i)+ ".pcd");% + string(i)

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

handles.Boxinfo = Boxinfo;
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
guidata(hObject, handles);
axes(handles.axes7)
plotCamera(camPoses, Size=0.2);
hold on
for i = 1:numClusters
    
    patch('Vertices', invcorners{i}.Location+cornMatrix{i}, 'Faces', faces, 'FaceColor', 'g', 'FaceAlpha', 0.2);
    text(cornMatrix{i}(1,1), cornMatrix{i}(1,2), cornMatrix{i}(1,3), num2str(i), 'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold');
end
colormap(hsv(numClusters))
pcshow(segmentedPtCloud.Location, labelColorIndex, VerticalAxis='y', VerticalAxisDir='down');

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
set(handles.figure1, 'Color', [0.94, 0.94, 0.94]);

%% Calculate the distance between the centroids of different bounding boxes
% 计算不同包围盒之间质心的距离

Boxcoornum = size(Boxcoordinate, 1);  % number of Points

distances = zeros(Boxcoornum);  % initial distance matrix

for i = 1:Boxcoornum
    for j = 1:Boxcoornum
        Box2Boxdis(i, j) = norm(Boxcoordinate(i, :) - Boxcoordinate(j, :));  % compute the Euclidean distance

    end
end

% Store the distance matrix in handles
handles.Box2Boxdis = Box2Boxdis;
% Update handles structure
guidata(hObject, handles);



%%
set(handles.progressBar,'Position',[20,20,450,20],'BackgroundColor',[0 1 0]);
    
    % 强制GUI更新
    drawnow;

set(handles.displayText, 'String', 'Your program ran successfully! Give your computer a pat for its hard work:)');







% --- Executes during object creation, after setting all properties.
function panel1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to panel1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2


% --- Executes during object creation, after setting all properties.
function axes3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes3



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btnReadOrInput.
function btnReadOrInput_Callback(hObject, eventdata, handles)
% hObject    handle to btnReadOrInput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Ask the user to choose the method
% Ask the user to choose the method
choice = questdlg('Would you like to read the camera parameters from a file or input them manually?', ...
    'Choose Method', ...
    'Read from file', 'Input manually', 'Cancel', 'Cancel');

    switch choice
        case 'Read from file'
            % Let the user select the .txt file
            [filename, path] = uigetfile('*.txt', 'Select the cameras.txt file');
            
            % When the user cancels file selection, filename will be '0'
            if isequal(filename, '0')
                disp('User cancelled file selection.');
            else
                direction = fullfile(path, filename);  % Concatenate the full path of the file
                camInfo = cameraReader(direction);
            
            % Set the camera parameters to the edit boxes
            set(handles.edit5, 'String', num2str(camInfo(1)));
            set(handles.edit6, 'String', num2str(camInfo(2)));
            set(handles.edit1, 'String', num2str(camInfo(3)));
            set(handles.edit2, 'String', num2str(camInfo(4)));
            set(handles.edit3, 'String', num2str(camInfo(5)));
            set(handles.edit4, 'String', num2str(camInfo(6)));
%% 定义相机参数类
% 定义相机内参矩阵
            IntrinMat = [camInfo(3), 0, camInfo(5);
                         0, camInfo(4), camInfo(6);
                         0, 0, 1];

            % 定义图像尺寸
            imageSize = [camInfo(1), camInfo(2)];  % 图像宽度和高度

            % 创建 cameraParameters对象
            cameraParams = cameraParameters('ImageSize', imageSize, 'K', IntrinMat);
            intrinsics = cameraParams.Intrinsics;
              
            end

case 'Input manually'
    prompt = {'Enter the width:', 'Enter the height:', 'Enter fx:', 'Enter fy:', 'Enter cx:', 'Enter cy:'};
    dlgtitle = 'Input';
    dims = [1 35];
    definput = {'', '', '', '', '', ''};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if ~isempty(answer)  % if input is given
        camInfo = cellfun(@str2num, answer, 'UniformOutput', false);
        camInfo = cell2mat(camInfo);
        
        % Set the camera parameters to the edit boxes
        set(handles.edit5, 'String', num2str(camInfo(1)));
        set(handles.edit6, 'String', num2str(camInfo(2)));
        set(handles.edit1, 'String', num2str(camInfo(3)));
        set(handles.edit2, 'String', num2str(camInfo(4)));
        set(handles.edit3, 'String', num2str(camInfo(5)));
        set(handles.edit4, 'String', num2str(camInfo(6)));
        
        %% 定义相机参数类
        % 定义相机内参矩阵
        IntrinMat = [camInfo(3), 0, camInfo(5);
                    0, camInfo(4), camInfo(6);
                    0, 0, 1];

        % 定义图像尺寸
        imageSize = [camInfo(1), camInfo(2)];  % 图像宽度和高度

        % 创建 cameraParameters对象
        cameraParams = cameraParameters('ImageSize', imageSize, 'K', IntrinMat);
        intrinsics = cameraParams.Intrinsics;
    end
    end

    set(handles.displayText, 'String', 'Camera parameters selected');


% --- Executes during object creation, after setting all properties.
function axes4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes4


% --- Executes during object creation, after setting all properties.
function axes5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes5


% --- Executes during object creation, after setting all properties.
function axes6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes6


% --- Executes during object creation, after setting all properties.
function axes7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes7


% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on slider movement.
function sliderProgress_Callback(hObject, eventdata, handles)
% hObject    handle to sliderProgress (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function displayText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to displayText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in radiobutton_small.


% --- Executes on button press in radiobutton_small.
function radiobutton_small_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_small (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_small
    set(handles.radiobutton_big, 'Value', 0);
    % set direct_B to 1
    handles.direct_B = 1;
    guidata(hObject, handles);

% --- Executes on button press in radiobutton_big.
function radiobutton_big_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_big (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_big
    % When big scene is chosen, deselect small scene
    set(handles.radiobutton_small, 'Value', 0);
    % set direct_B to 0
    handles.direct_B = 0;
    guidata(hObject, handles);


    



function edit_boundingBoxID_Callback(hObject, eventdata, handles)
% hObject    handle to edit_boundingBoxID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_boundingBoxID as text
%        str2double(get(hObject,'String')) returns contents of edit_boundingBoxID as a double
% Your bounding box matrix
% Your bounding box matrix here

% Get the bounding box ID from the user
boundingBoxID = str2double(get(hObject, 'String')); % hObject here is edit_boundingBoxID

Boxinfo = handles.Boxinfo;
% Check if the ID is within the matrix bounds
if boundingBoxID > size(Boxinfo, 1) || boundingBoxID < 1
    % If it is not, display an error message
    set(handles.displayText, 'String', 'Error: Invalid bounding box ID.');
else
    % Get the bounding box data
    boundingBoxData = Boxinfo(boundingBoxID, :);

    % Create a string to display
    displayString = sprintf('dx: %f, dy: %f, dz: %f, Volume: %f', boundingBoxData(1), boundingBoxData(2), boundingBoxData(3), boundingBoxData(4));
    % Display the string in the text box
    set(handles.displayText, 'String', displayString);
end



% --- Executes during object creation, after setting all properties.
function edit_boundingBoxID_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_boundingBoxID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_bbox1_Callback(hObject, eventdata, handles)
% hObject    handle to edit_bbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_bbox1 as text
%        str2double(get(hObject,'String')) returns contents of edit_bbox1 as a double

% Get the user input from editbox1
box_num1 = str2double(get(hObject,'String'));


% If the user has also entered the second box number
if isfield(handles, 'box_num2')
    % Get the distance between the two boxes from the distance matrix
    distance = handles.Box2Boxdis(box_num1, handles.box_num2);

    % Update the text component with the distance
    set(handles.displayText, 'String', sprintf('Distance: %.2f', distance));
else
    % Update the text component to indicate that the user needs to enter the second box number
    set(handles.displayText, 'String', 'Please enter the second Bounding Box number');
end

% Store the first box number in the handles structure
handles.box_num1 = box_num1;
guidata(hObject, handles);









% --- Executes during object creation, after setting all properties.
function edit_bbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_bbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_bbox2_Callback(hObject, eventdata, handles)
% hObject    handle to edit_bbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


box_num2 = str2double(get(hObject,'String'));

% If the user has also entered the first box number
if isfield(handles, 'box_num1')
    % Get the distance between the two boxes from the distance matrix
    distance = handles.Box2Boxdis(handles.box_num1, box_num2);

    % Update the text component with the distance
    set(handles.displayText, 'String', sprintf('Distance: %.2f', distance));
else
    % Update the text component to indicate that the user needs to enter the first box number
    set(handles.displayText, 'String', 'Please enter the first Bounding Box number');
end

% Store the second box number in the handles structure
handles.box_num2 = box_num2;
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function edit_bbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_bbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
