%%  ��ջ�����������ӱ���
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������
tic
currentFolder = pwd;
addpath(genpath(currentFolder))
% restoredefaultpath

%%  ����ʱ�������ļ���
currentDateTime = datestr(now, '(ALL)zzyyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, currentDateTime);
mkdir(outputFolder);

%% ��������
res = xlsread('C:\Users\zhangzhen\Documents\MATLAB\wendu\ZZ_Experiment_Seven\Laboratory dataset.xlsx');
res_T = res(:, end-1);      % ��ʵ���¶�T
res_K = res(:, end);        % �ɱ�ɫ����ʵ�����Kֵ
size1 = size(res,1);        % ���� 

%% 7:3�������ݼ�Ϊѵ�����Ͳ��Լ�
%�������¶����ݣ����ң�����ȡ30%��������Ϊ�¶�Ԥ�⼯
% P = size1*0.7; P = fix(P);
% size2=size(res,2);
% temp = randperm(size1);

%�������¶����ݵ����30%��������Ϊ�¶�Ԥ�⼯
% temp = 1:size1;   %������Ļ�ע����һ��������һ��

%�������¶����ݵ��м�40%~70%֮����ȡ30%��������Ϊ�¶�Ԥ�⼯
% start_index = ceil(0.4 * size1);
% end_index = floor(0.7 * size1);
% temp = 1:size1;
% moved_elements = temp(start_index:end_index);
% temp(start_index:end_index) = [];
% insert_index = ceil(0.7 * size1);
% temp = [temp(1:insert_index-1) moved_elements
% temp(insert_index:end)];

%��1050-1070���¶�������Ϊ�¶�Ԥ�⼯
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

num_dim = size(res, 2) - 2;               % ����ά��
train_data = [P_train', T_train', K_train']; % ��P_train��T_trainƴ�ӳ�һ������
test_data = [P_test', T_test', K_test']; % ��P_train��T_trainƴ�ӳ�һ������
save(fullfile(outputFolder, 'train_data.mat'),"train_data");
save(fullfile(outputFolder, 'test_data.mat'),"test_data");

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

save(fullfile(outputFolder, 'ps_input.mat'),"ps_input");
save(fullfile(outputFolder, 'ps_output.mat'),"ps_output");

%%  ת������Ӧģ��
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  ����ģ��
c = 4.0;    % �ͷ�����
g = 0.4;    % �������������
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
model = svmtrain(t_train, p_train, cmd);

save(fullfile(outputFolder, 'model.mat'),"model");

%%  ����Ԥ��
[t_sim1, error_1] = svmpredict(t_train, p_train, model);
[t_sim2, error_2] = svmpredict(t_test , p_test , model);

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output)';
T_sim2 = mapminmax('reverse', t_sim2, ps_output)';

save(fullfile(outputFolder, 'T_sim1.mat'),"T_sim1");
save(fullfile(outputFolder, 'T_sim2.mat'),"T_sim2");

%%  ��ػع�ָ��ļ���
%��������� RMSE  (��ֱ�ӿ���ģ�Ͷ��¶ȵ�ƫ��)
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%����ϵ��  R^2 (����ģ�ͻع�Ч���ĺû���Խ�ӽ�1Խ��)
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%������� MSE (�ɴ�ſ���ģ�Ͷ��¶ȵ�ƫ��)
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;

%RPD ʣ��Ԥ��в� (Ԥ��ģ�͵Ŀɿ��̶�)
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;
SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;

%ƽ��������� MAE (��ֱ�ӿ���ģ�Ͷ��¶ȵ�ƫ��)
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));

%ƽ�����԰ٷֱ���� MAPE (ԽСԽ�ã��Ǹ��ٷֱ�����ָ��)
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));

%% ���Լ��ϵĻع�Ч�������ͳ��ͼ�������
figure;
plotregression(T_test,T_sim2,['�ع�ͼ']);
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '�ع�ͼ.png'))

figure;
ploterrhist(T_test-T_sim2,['���ֱ��ͼ']);
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '���ֱ��ͼ.png'))

%%  ѵ������ͼ
figure
%plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1)
plot(1:M,T_train,'r-',1:M,T_sim1,'b-','LineWidth',1.5)
legend('��ʵֵ','CNN-BILSTM-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'ѵ����Ԥ�����Ա�';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'ѵ����Ԥ�����Ա�.png'))

%% Ԥ�⼯��ͼ
figure
plot(1:N,T_test,'r-',1:N,T_sim2,'b-','LineWidth',1.5)
legend('��ʵֵ','CNN-BILSTM-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'���Լ�Ԥ�����Ա�';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '���Լ�Ԥ�����Ա�.png'))

%% ���Լ����ͼ
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('���Լ��������')
ylabel('Ԥ�����')
title('���Լ�Ԥ�����')
grid on;
legend('Ԥ��������')
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '���Լ�Ԥ��������.png'))

%% �����������ͼ��ѵ�������Ч��ͼ��
figure
plot(T_train,T_sim1,'*r');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string = {'ѵ����Ч��ͼ';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'ѵ����Ч��ͼ.png'))

%% �����������ͼ��Ԥ�⼯���Ч��ͼ��
figure
plot(T_test,T_sim2,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'���Լ�Ч��ͼ';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '���Լ�Ч��ͼ.png'))

%% ����������Ԥ�����ͼ��ָ����ƽ����
R3=(R1+R2)./2;
error3=(error1+error2)./2;
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'�����������Ԥ��ͼ';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '�����������Ԥ��ͼ.png'))

%% ��ӡ������ָ��(������)
disp(['-----------------------������--------------------------'])
disp(['���۽��������ʾ��'])
disp(['ƽ���������MAEΪ��',num2str(MAE2)])
disp(['�������MSEΪ��       ',num2str(mse2)])
disp(['���������RMSEΪ��  ',num2str(error2)])
disp(['����ϵ��R^2Ϊ��  ',num2str(R2)])
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD2)])
disp(['ƽ�����԰ٷֱ����MAPEΪ��  ',num2str(MAPE2)])
grid

filePath = fullfile(outputFolder, 'Evaluation_indicators.txt');
fileID = fopen(filePath, 'w');
fprintf(fileID, 'ƽ���������MAEΪ��%.6f\n', MAE2);
fprintf(fileID, '�������MSEΪ��%.6f\n', mse2);
fprintf(fileID, '���������RMSEΪ�� %.6f\n', error2);
fprintf(fileID, '����ϵ��R^2Ϊ�� %.6f\n', R2);
fprintf(fileID, 'ʣ��Ԥ��в�RPDΪ�� %.6f\n', RPD2);
fprintf(fileID, 'ƽ�����԰ٷֱ����MAPEΪ�� %.6f\n', MAPE2);
fclose(fileID);

%% ZZ��Ҫ��ͼ(�������Ԥ��Ч��ͼ)
ALL_data = res;
ALL_T = res_T;
ALL_feature = ALL_data(:,1:size2-2)';
ALL_feature_1 = mapminmax('apply', ALL_feature, ps_input)';

[ALL_sim, error_ALL] = svmpredict(ALL_T, ALL_feature_1, model);

ALL_sim_1 = mapminmax('reverse', ALL_sim', ps_output);
ALL_sim_2 = double(ALL_sim_1)';
save(fullfile(outputFolder, 'ALL_sim_2.mat'),"ALL_sim_2");

% ��������Ԥ��Ĳ������
N10 = size(ALL_T, 1);
error10 = sqrt(sum((ALL_T - ALL_sim_2).^2)./N10);
R10 = 1 - norm(ALL_T - ALL_sim_2)^2 / norm(ALL_T - mean(ALL_T ))^2;
mse10 = sum((ALL_sim_2 - ALL_T).^2)./N10;
SE10=std(ALL_sim_2 - ALL_T);
RPD10=std(ALL_T)/SE10;
MAE10 = mean(abs(ALL_T - ALL_sim_2));
MAPE10 = mean(abs((ALL_T - ALL_sim_2)./ALL_T));

fps = 1; % ÿ֡����
time = (0:length(ALL_sim_2)-1) * fps; % ����ʱ�����У�T��Ӧ��ʱ�䣩
figure;
hold on;
% ����T1����ͼ����ɫ
plot(time, ALL_sim_2, '-', 'Color', [1, 0.5, 0], 'LineWidth', 1.9);
% ����T2����ͼ����ɫ
plot(time, ALL_T, '-', 'Color', [0, 0, 1] , 'LineWidth', 1.5);
xlabel('{\fontname{Arial} FPS}','FontSize',13,'Interpreter','tex');
ylabel('{\fontname{Arial} Temperature/^\circC}','FontSize',13,'Interpreter','tex');
legend('T Predicted', 'T True');
title(['�����¶�Ԥ����']);
grid off;
text(0.72, 0.18, {['MAE: ', num2str(MAE10)], ['MSE: ', num2str(mse10)], ...
    ['RMSE: ', num2str(error10)], ['R^2: ', num2str(R10)], ['RPD: ', num2str(RPD10)], ...
    ['MAPE: ', num2str(MAPE10)]}, 'Units', 'normalized', 'FontSize', 10, 'FontName', 'Arial');
saveas(gcf, fullfile(outputFolder, '�����¶Ȱ�˳��Ԥ����.fig'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, '�����¶Ȱ�˳��Ԥ����.png'))

