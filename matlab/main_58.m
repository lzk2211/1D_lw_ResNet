close all;

strs = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001"];
floder = "data_set_1024_1024_5.8";

% str = "0001";
distance = "D00";
T = 1024*1024;
N = 1024;
fs = 1e8;
h = hanning(N);
noverlap = N/2;

if exist(sprintf('F:\\%s\\%s', distance, floder))==0 %%判断文件夹是否存在
    mkdir(sprintf('F:\\%s\\%s', distance, floder));  %%不存在时候，创建文件夹
end

for k = 1:length(strs)
    str = strs(k);
    filename = sprintf('F:\\%s\\mat\\T%s\\*.mat', distance, str);
    if exist(sprintf('F:\\%s\\%s\\T%s', distance, floder, str))==0 %%判断文件夹是否存在
        mkdir(sprintf('F:\\%s\\%s\\T%s', distance, floder, str));  %%不存在时候，创建文件夹
    end
    
    Files = dir(fullfile(filename));
    LengthFiles = length(Files);
    
    index = 0;
    
    for j=1:LengthFiles
        name=Files(j).name;           %读取struct变量的格式
        folder=Files(j).folder;
        disp(['loading... ' folder,'\',name]);
        load([folder,'\',name]);    %导入文件
    
        signal = RF1_I + 1i*RF1_Q;% RF1 是5.8GHz
        
        disp('packing pic...');
        for i = 0:(size(signal)/T)-1
            signal_data = signal((i*T+1):((i+1)*T));
            [S, f, t] = spectrogram(signal_data, h, noverlap, N, fs);
            S = fftshift(S);
            
            A = 20*log10(abs(S));
    
            pic_name = sprintf('F:\\%s\\%s\\T%s\\%d.png', distance, floder, str, index + i);
            
            % 归一化矩阵
            A_min = min(A(:)); % 矩阵的最小值
            A_max = max(A(:)); % 矩阵的最大值
            
            % 计算最大值和最小值之间的差
            A_range = A_max - A_min;
            
            % 归一化
            A_normalized = (A - A_min) / A_range;
    
            GRAY_image = cat(3, A_normalized, A_normalized, A_normalized);
    
            % 保存图像
%             save(mat_name, "A");
            
            GRAY_image = imresize(GRAY_image, [1024,1024]);
            imwrite(GRAY_image, pic_name);

%             save(pic_name, 'A_normalized')
            fprintf('%d ', index + i);
        end
        
        fprintf('\n');
        index = index + i + 1;
        
    end

end