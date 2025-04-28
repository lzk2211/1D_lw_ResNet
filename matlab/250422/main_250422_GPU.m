close all;

% strs = ["0000","0001","0010","0011","0100","0101","0110","0111"];
strs_mult = ["0010","0011","0101","0110","0111","1001"];
strs_58 = ["1010","1100","1110","1111","10010"];
strs_24 = ["10000","10001","10011","10100","10101","10110","10111","11000"];

strs_mult_12 = ["1011","1101"];

% strs = ["1010","1011","1100","1101","1110","1111"];
% strs = ["10000","10001","10010","10011","10100","10101","10110","10111","11000"];
strs = ["11000"];

% str = "0001";
distance = "D00";
pro = "data_set_5e6";
T = 5e6;
N = 2048;
fs = 1e8;
h = gpuArray(hanning(N)); % 将汉宁窗转移到GPU
noverlap = N/2;

% 检查GPU是否可用
if gpuDeviceCount == 0
    error('No GPU device found. Please check your GPU configuration.');
else
    gpu = gpuDevice();
    fprintf('Using GPU: %s\n', gpu.Name);
end

for k = 1:length(strs)
    str = strs(k);
    filename = sprintf('E:\\%s\\mat\\T%s\\*.mat', distance, str);
    if exist(sprintf('E:\\%s\\%s\\T%s', distance, pro, str))==0 %%判断文件夹是否存在
        mkdir(sprintf('E:\\%s\\%s\\T%s', distance, pro, str));  %%不存在时候，创建文件夹
    end
    
    Files = dir(fullfile(filename));
    LengthFiles = length(Files);
    
    index = 0;
    
    for j=1:LengthFiles
        name=Files(j).name;           %读取struct变量的格式
        folder=Files(j).folder;
        disp(['loading... ' folder,'\\',name]);
        load([folder,'\\',name]);    %导入文件
    
        if any(strcmp(str, strs_mult))
            if name(end-7)=='0'%前8
                signal = RF0_I + 1i*RF0_Q;
            else%后8
                signal = RF1_I + 1i*RF1_Q;
            end
        elseif any(strcmp(str, strs_58))
            signal = RF1_I + 1i*RF1_Q;
        elseif any(strcmp(str, strs_24))
            signal = RF0_I + 1i*RF0_Q;
        elseif any(strcmp(str, strs_mult_12))
            if str == "1101"
                if bin2dec(name(end-7:end-4)) <= 5
                    signal = RF1_I + 1i*RF1_Q;
                else
                    signal = RF0_I + 1i*RF0_Q;
                end
            end
            if str == "1011"
                if bin2dec(name(end-7:end-4)) > 6
                    signal = RF1_I + 1i*RF1_Q;
                else
                    signal = RF0_I + 1i*RF0_Q;
                end
            end
        end
        
        % 将信号数据转移到GPU
        signal = gpuArray(signal);
        
        disp('packing pic...');
        for i = 0:(size(signal)/T)-1
            signal_data = signal((i*T+1):((i+1)*T));
            
            % 在GPU上执行STFT计算
            [S, f, t] = spectrogram(signal_data, h, noverlap, N, fs);
            S = fftshift(S);
            
            % 将结果转移回CPU
            A = gather(20*log10(abs(S)));
    
            pic_name = sprintf('E:\\%s\\%s\\T%s\\%d.png', distance, pro, str, index + i);
            
            % 归一化矩阵
            A_min = min(A(:)); % 矩阵的最小值
            A_max = max(A(:)); % 矩阵的最大值
            
            % 计算最大值和最小值之间的差
            A_range = A_max - A_min;
            
            % 归一化
            A_normalized = (A - A_min) / A_range;
    
            GRAY_image = cat(3, A_normalized, A_normalized, A_normalized);
    
            % 保存图像
            GRAY_image = imresize(GRAY_image, [512,512]);
            imwrite(GRAY_image, pic_name);

            fprintf('%d ', index + i);
        end
        
        fprintf('\n');
        index = index + i + 1;
        
        % 清除GPU内存
        clear signal signal_data S;
        wait(gpu); % 等待GPU操作完成
        
    end
end