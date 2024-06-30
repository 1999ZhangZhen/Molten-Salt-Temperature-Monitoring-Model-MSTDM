%%  清空环境变量和添加变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
tic
currentFolder = pwd;
addpath(genpath(currentFolder))
% restoredefaultpath

%%  创建时间命名文件夹
currentDateTime = datestr(now, '(ALL)zzyyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, currentDateTime);
mkdir(outputFolder);

%% 导入数据
res = xlsread('C:\Users\zhangzhen\Documents\MATLAB\wendu\ZZ_Experiment_Seven\Laboratory dataset.xlsx');
res_T = res(:, end-1);      % 真实的温度T
res_K = res(:, end);        % 由比色法真实计算的K值
size1 = size(res,1);        % 行数 

%% 7:3划分数据集为训练集和测试集
%将所有温度数据（打乱）后提取30%的数据作为温度预测集
% P = size1*0.7; P = fix(P);
% size2=size(res,2);
% temp = randperm(size1);

%将所有温度数据的最后30%的数据作为温度预测集
% temp = 1:size1;   %不随机的话注释上一条，打开这一条

%在所有温度数据的中间40%~70%之间提取30%的数据作为温度预测集
% start_index = ceil(0.4 * size1);
% end_index = floor(0.7 * size1);
% temp = 1:size1;
% moved_elements = temp(start_index:end_index);
% temp(start_index:end_index) = [];
% insert_index = ceil(0.7 * size1);
% temp = [temp(1:insert_index-1) moved_elements
% temp(insert_index:end)];

%将1050-1070的温度数据作为温度预测集
P = size1*(1-(375/1478)); P = fix(P);
size2=size(res,2);
seq1 = 1:559;
seq2 = 937:1478;
seq3 = 560:936;
temp = [seq1 seq2 seq3];

P_train = res(temp(1:P),1:size2-2)';
T_train = res_T(temp(1:P),1)';
K_train = res_K(temp(1:P),1)';
M = size(P_train, 2);

P_test = res(temp(P:end),1:size2-2)';
T_test = res_T(temp(P:end),1)';
K_test = res_K(temp(P:end),1)';
N = size(P_test, 2);

num_dim = size(res, 2) - 2;               % 特征维度
train_data = [P_train', T_train', K_train']; % 将P_train和T_train拼接成一个矩阵
test_data = [P_test', T_test', K_test']; % 将P_train和T_train拼接成一个矩阵
save(fullfile(outputFolder, 'train_data.mat'),"train_data");
save(fullfile(outputFolder, 'test_data.mat'),"test_data");

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

save(fullfile(outputFolder, 'ps_input.mat'),"ps_input");
save(fullfile(outputFolder, 'ps_output.mat'),"ps_output");

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  创建模型
c = 4.0;    % 惩罚因子
g = 0.4;    % 径向基函数参数
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
model = svmtrain(t_train, p_train, cmd);

save(fullfile(outputFolder, 'model.mat'),"model");

%%  仿真预测
[t_sim1, error_1] = svmpredict(t_train, p_train, model);
[t_sim2, error_2] = svmpredict(t_test , p_test , model);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output)';
T_sim2 = mapminmax('reverse', t_sim2, ps_output)';

save(fullfile(outputFolder, 'T_sim1.mat'),"T_sim1");
save(fullfile(outputFolder, 'T_sim2.mat'),"T_sim2");

%%  相关回归指标的计算
%均方根误差 RMSE  (可直接看出模型对温度的偏差)
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%决定系数  R^2 (衡量模型回归效果的好坏，越接近1越好)
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%均方误差 MSE (可大概看出模型对温度的偏差)
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;

%RPD 剩余预测残差 (预测模型的可靠程度)
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;
SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;

%平均绝对误差 MAE (可直接看出模型对温度的偏差)
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));

%平均绝对百分比误差 MAPE (越小越好，是个百分比评价指标)
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));

%% 测试集上的回归效果和误差统计图分析结果
figure;
plotregression(T_test,T_sim2,['回归图']);
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '回归图.png'))

figure;
ploterrhist(T_test-T_sim2,['误差直方图']);
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '误差直方图.png'))

%%  训练集绘图
figure
%plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1)
plot(1:M,T_train,'r-',1:M,T_sim1,'b-','LineWidth',1.5)
legend('真实值','CNN-BILSTM-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '训练集预测结果对比.png'))

%% 预测集绘图
figure
plot(1:N,T_test,'r-',1:N,T_sim2,'b-','LineWidth',1.5)
legend('真实值','CNN-BILSTM-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '测试集预测结果对比.png'))

%% 测试集误差图
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('测试集样本编号')
ylabel('预测误差')
title('测试集预测误差')
grid on;
legend('预测输出误差')
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '测试集预测输出误差.png'))

%% 绘制线性拟合图（训练集拟合效果图）
figure
plot(T_train,T_sim1,'*r');
xlabel('真实值')
ylabel('预测值')
string = {'训练集效果图';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '训练集效果图.png'))

%% 绘制线性拟合图（预测集拟合效果图）
figure
plot(T_test,T_sim2,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'测试集效果图';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '测试集效果图.png'))

%% 总数据线性预测拟合图（指标求平均）
R3=(R1+R2)./2;
error3=(error1+error2)./2;
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'所有样本拟合预测图';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '所有样本拟合预测图.png'))

%% 打印出评价指标(并保存)
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
fprintf(fileID, '平均绝对误差MAE为：%.6f\n', MAE2);
fprintf(fileID, '均方误差MSE为：%.6f\n', mse2);
fprintf(fileID, '均方根误差RMSE为： %.6f\n', error2);
fprintf(fileID, '决定系数R^2为： %.6f\n', R2);
fprintf(fileID, '剩余预测残差RPD为： %.6f\n', RPD2);
fprintf(fileID, '平均绝对百分比误差MAPE为： %.6f\n', MAPE2);
fclose(fileID);

%% ZZ想要的图(针对整体预测效果图)
ALL_data = res;
ALL_T = res_T;
ALL_feature = ALL_data(:,1:size2-2)';
ALL_feature_1 = mapminmax('apply', ALL_feature, ps_input)';

[ALL_sim, error_ALL] = svmpredict(ALL_T, ALL_feature_1, model);

ALL_sim_1 = mapminmax('reverse', ALL_sim', ps_output);
ALL_sim_2 = double(ALL_sim_1)';
save(fullfile(outputFolder, 'ALL_sim_2.mat'),"ALL_sim_2");

% 完整数据预测的测评结果
N10 = size(ALL_T, 1);
error10 = sqrt(sum((ALL_T - ALL_sim_2).^2)./N10);
R10 = 1 - norm(ALL_T - ALL_sim_2)^2 / norm(ALL_T - mean(ALL_T ))^2;
mse10 = sum((ALL_sim_2 - ALL_T).^2)./N10;
SE10=std(ALL_sim_2 - ALL_T);
RPD10=std(ALL_T)/SE10;
MAE10 = mean(abs(ALL_T - ALL_sim_2));
MAPE10 = mean(abs((ALL_T - ALL_sim_2)./ALL_T));

fps = 1; % 每帧秒数
time = (0:length(ALL_sim_2)-1) * fps; % 生成时间序列（T对应的时间）
figure;
hold on;
% 绘制T1折线图，橙色
plot(time, ALL_sim_2, '-', 'Color', [1, 0.5, 0], 'LineWidth', 1.9);
% 绘制T2折线图，蓝色
plot(time, ALL_T, '-', 'Color', [0, 0, 1] , 'LineWidth', 1.5);
xlabel('{\fontname{Arial} FPS}','FontSize',13,'Interpreter','tex');
ylabel('{\fontname{Arial} Temperature/^\circC}','FontSize',13,'Interpreter','tex');
legend('T Predicted', 'T True');
title(['整体温度预测结果']);
grid off;
text(0.72, 0.18, {['MAE: ', num2str(MAE10)], ['MSE: ', num2str(mse10)], ...
    ['RMSE: ', num2str(error10)], ['R^2: ', num2str(R10)], ['RPD: ', num2str(RPD10)], ...
    ['MAPE: ', num2str(MAPE10)]}, 'Units', 'normalized', 'FontSize', 10, 'FontName', 'Arial');
saveas(gcf, fullfile(outputFolder, '整体温度按顺序预测结果.fig'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '整体温度按顺序预测结果.png'))

