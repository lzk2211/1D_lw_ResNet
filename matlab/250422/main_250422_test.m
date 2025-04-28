% clear;

distance = "D00";
str = "est";
T = 5e6;
N = 2048;
fs = 1e8;
h = hanning(N);
noverlap = N/2;

signal = RF1_I + 1i*RF1_Q;
% signal = RF0_I + 1i*RF0_Q;

disp('packing pic...');
for i = 0:(size(signal)/T)-1
    signal_data = signal((i*T+1):((i+1)*T));
    [S, f, t] = spectrogram(signal_data, h, noverlap, N, fs);
    S = fftshift(S);
    
    A = 20*log10(abs(S));

    pic_name = sprintf('E:\\%s\\data_set_5e6\\T%s\\%d.png', distance, str, i);
    
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
    
    GRAY_image = imresize(GRAY_image, [512,512]);
    imwrite(GRAY_image, pic_name);

%             save(pic_name, 'A_normalized')
%     fprintf('%d ', index + i);
end
