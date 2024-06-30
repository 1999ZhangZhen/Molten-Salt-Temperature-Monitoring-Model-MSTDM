%% 清空环境变量和添加变量  (工业数据集的预测)
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
currentFolder = pwd;
addpath(genpath(currentFolder))

%%  创建时间命名文件夹
currentDateTime = datestr(now, '(1107)zzyyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, currentDateTime);
mkdir(outputFolder);

%% 确定K值的样本
% 读取.xlsx文件，假设文件名为'rg_values.xlsx'，第一列为R，第二列为G
biaoding_data = xlsread('industrial_1075.xlsx');
R = biaoding_data(:, 3);
G = biaoding_data(:, 2);
ZZ_T = 1075;
TT = zeros(size(G)) + ZZ_T;

C = 0.014388;
Xg = 546.1* 1e-9;
Xr = 700* 1e-9;
b = 273.15;
T = TT+b;

% 初始化K向量
K = zeros(size(R));
for i = 1:length(R)
    K(i) = ((C * ((1/Xg) - (1/Xr))) / T(i)) - 5 * log(Xr/Xg) - log(R(i)/G(i));
end

K1 =mean(K);
fprintf('K1 = %.6f\n', K1);     %标定的K值

%% 用标定的K值来预测其他工业温度数据集
ALL_data = xlsread('industrial_1107.xlsx');

R1 = ALL_data(:, 3);
G1 = ALL_data(:, 2);
ZZ_T1 = 1107;                 % 预测样本的真实标签

TT1_real = zeros(size(G1)) + ZZ_T1;

TT1_prediction = zeros(size(R1));

for k = 1:length(R1)
    TT1_prediction(k) = ((C * ((1/Xg) - (1/Xr))) / (log(R1(k)/G1(k))+K1+5*log(Xr/Xg)))-b;
end 


%% 误差统计
%  均方根误差 RMSE
TT1_prediction = TT1_prediction';
TT1_real = TT1_real';
N = size(ALL_data, 1);
error2 = sqrt(sum((TT1_real - TT1_prediction).^2)./N);

%决定系数
R2 = 1 - norm(TT1_real - TT1_prediction)^2 / norm(TT1_real - mean(TT1_real ))^2;

%均方误差 MSE
mse2 = sum((TT1_prediction - TT1_real).^2)./N;

%RPD 剩余预测残差
SE=std(TT1_prediction - TT1_real);
RPD2=std(TT1_real)/SE;

% 平均绝对误差MAE
MAE2 = mean(abs(TT1_real - TT1_prediction));

% 平均绝对百分比误差MAPE
MAPE2 = mean(abs((TT1_real - TT1_prediction)./TT1_real));

% 打印出评价指标
disp(['-----------------------误差计算--------------------------'])
disp(['评价结果如下所示：'])
disp(['平均绝对误差MAE为：',num2str(MAE2)])
disp(['均方误差MSE为：       ',num2str(mse2)])
disp(['均方根误差RMSE为：  ',num2str(error2)])
disp(['决定系数R^2为：  ',num2str(R2)])
disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE2)])
grid

filePath = fullfile(outputFolder, 'Evaluation_indicators.txt');
fileID = fopen(filePath, 'w');
fprintf(fileID, '平均绝对误差MAE为: %.6f\n', MAE2);
fprintf(fileID, '均方误差MSE为: %.6f\n', mse2);
fprintf(fileID, '均方根误差RMSE为: %.6f\n', error2);
fprintf(fileID, '决定系数R^2为: %.6f\n', R2);
fprintf(fileID, '剩余预测残差RPD为: %.6f\n', RPD2);
fprintf(fileID, '平均绝对百分比误差MAPE为: %.6f\n', MAPE2);
fclose(fileID);

%% 绘制折线图
fps = 1; % 每帧秒数
time1 = (0:length(TT1_prediction)-1) * fps; % 生成时间序列（T1对应的时间）
time2 = (0:length(TT1_real)-1) * fps; % 生成时间序列（T2对应的时间）
figure;
hold on;
% 绘制T1折线图，橙色
plot(time1, TT1_prediction, '-', 'Color', [1, 0.5, 0], 'LineWidth', 1.0);
% 绘制T2折线图，蓝色
plot(time2, TT1_real, '-', 'Color', [0, 0, 1] , 'LineWidth', 1.5);

xlabel('{\fontname{Arial} FPS}','FontSize',13,'Interpreter','tex');
ylabel('{\fontname{Arial} Temperature/^\circC}','FontSize',13,'Interpreter','tex');
legend('T Predicted', 'T True');
title('用于标定的样本为1075℃');
grid off;


% text(0.72, 0.18, {['MAE: ', num2str(MAE2)], ['MSE: ', num2str(mse2)], ...
%     ['RMSE: ', num2str(error2)], ['R^2: ', num2str(R2)], ['RPD: ', num2str(RPD2)], ...
%     ['MAPE: ', num2str(MAPE2)]}, 'Units', 'normalized', 'FontSize', 10, 'FontName', 'Arial');
annotation('textbox', [0.68, 0.14, 0.2, 0.3], 'String', {['MAE: ', num2str(MAE2)], ...
    ['MSE: ', num2str(mse2)], ['RMSE: ', num2str(error2)], ['R^2: ', num2str(R2)], ...
    ['RPD: ', num2str(RPD2)], ['MAPE: ', num2str(MAPE2)]}, 'FontSize', 10, 'FontName', 'Arial', ...
    'EdgeColor', 'black', 'BackgroundColor', 'white', 'LineStyle', '-', 'LineWidth', 1.5);

saveas(gcf, fullfile(outputFolder, 'ALL_data_Predicted.fig'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'ALL_data_Predicted.png'))

%% 保存相关的数据
save(fullfile(outputFolder, 'K1.mat'),"K1");
save(fullfile(outputFolder, 'TT1_prediction.mat'),"TT1_prediction");
save(fullfile(outputFolder, 'biaoding_data.mat'),"biaoding_data");




