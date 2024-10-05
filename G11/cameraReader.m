function camInfo = cameraReader(direction)
%CAMERA PROPERTY READER 读取cameras.txt文件，返回数组
%   读取camera.txt文件（需要GUI进行文件的选取.txt）,返回相机高，宽，内参的array
    % Open the camera file
    fid = fopen(direction,"r");

    % Read the lines
    lines = textscan(fid,'%s','Delimiter','\n');
    lines = lines{1};

    fclose(fid);

    % Init property array
    camInfo = [];

    for i = 1:numel(lines)
        line = lines{i};

        % skip description start with "#"
        if startsWith(line,'#')
            continue;
        end

        % Split the line into separate properties
        property = strsplit(line);
        camInfo = str2double(property(3:end));
        break;
    end
    
end


