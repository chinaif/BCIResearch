% Run_All_Experiments.m
clear; clc; close all;

% ================= 配置区域 =================
% 1. 路径设置
addpath('utils');
addpath('datasets');
data_path = 'D:\if\Research\Code\Multi-view-datasets-master'; % 您的数据路径
addpath(data_path);

% 2. 结果保存文件夹 (自动创建)
result_dir = 'Experiment_Results';
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% 3. 定义要跑的数据集列表 (您可以根据需要增减)
dataset_list = {
    % --- 快速验证组 ---
    '3Sources'; 'BBCSport'; 'Yale'; 
    
    % --- 稳定测试组 ---
    'UCI'; 'BBC4view'; 'MSRC-v5'; 'Prokaryotic';
    
    % --- 压力测试组 ---
    'Handwritten'; 'Reuters-1500'; 'OutdoorScene'
};

% 4. 定义要跑的算法模式
mode_list = {'A', 'B', 'C'};

% ================= 参数设置 =================
% 这里保持您原有的参数设置
K_list = [2];         
mu = 10.^[1];         
lambda = 2^10;        
beta = 10.^[-3];      
alpha_param = 10;     

% ================= 主循环 =================
fprintf('开始批量实验...\n');
fprintf('提示：如果您想安全停止程序，请在当前文件夹新建一个名为 STOP.txt 的空文件。\n\n');

total_tasks = length(dataset_list) * length(mode_list);
current_task = 0;

for d_idx = 1:length(dataset_list)
    dataset_name = dataset_list{d_idx};
    
    for m_idx = 1:length(mode_list)
        ALGO_MODE = mode_list{m_idx};
        current_task = current_task + 1;
        
        % --- 0. 检查软停止信号 ---
        if exist('STOP.txt', 'file')
            fprintf('\n检测到 STOP.txt，程序正在安全停止...\n');
            return; % 退出整个脚本
        end
        
        % --- 1. 生成结果文件名 ---
        % 文件名格式: Result_数据集_模式.mat
        save_name = sprintf('Res_%s_Mode%s.mat', dataset_name, ALGO_MODE);
        save_path = fullfile(result_dir, save_name);
        
        % --- 2. 断点续传检查 ---
        if exist(save_path, 'file')
            fprintf('[%d/%d] 跳过: %s (模式 %s) - 结果已存在\n', ...
                current_task, total_tasks, dataset_name, ALGO_MODE);
            continue; 
        end
        
        fprintf('==================================================\n');
        fprintf('[%d/%d] 正在运行: %s | 模式: %s\n', ...
            current_task, total_tasks, dataset_name, ALGO_MODE);
        
        % --- 3. 核心逻辑 (包裹在 try-catch 中以防单个数据报错卡死) ---
        try
            % 3.1 加载数据
            data_file = fullfile(data_path, [dataset_name, '.mat']);
            if ~exist(data_file, 'file')
                warning('数据文件不存在: %s', dataset_name);
                continue;
            end
            
            % 清理旧变量，防止污染
            clear X Y y truth gnd
            load(data_file);
            
            % 统一标签变量名
            if exist('y', 'var') && ~exist('Y', 'var'), Y = y; end
            if exist('truth', 'var') && ~exist('Y', 'var'), Y = truth; end
            if exist('gnd', 'var') && ~exist('Y', 'var'), Y = gnd; end
            
            % 3.2 数据预处理
            num = size(X{1}, 1);
            V_num = length(X);
            c = length(unique(Y));
            for i = 1 : V_num
                X{i} = full((X{i} - mean(X{i}, 2)) ./ repmat(std(X{i}, [], 2), 1, size(X{i}, 2)));
            end
            
            % 3.3 算法实现 (您的核心逻辑)
            results = []; % 存储当前数据集当前模式下的所有参数组合结果
            
            for k_idx = 1:length(K_list)
                K_curr = K_list(k_idx);
                
                % [步骤 1 & 2] 构图逻辑分支
                switch ALGO_MODE
                    case 'A' 
                        % 算法 A: 仅优化初始视图
                        A_base = construct_Graph(X, 5, 1); 
                        A_input = cell(V_num, K_curr);
                        for v = 1:V_num
                            W_opt = optimize_Z(A_base{v, 1}, alpha_param);
                            W_opt = (W_opt + W_opt') / 2;
                            A_input{v, 1} = W_opt;
                            for k = 2:K_curr
                                A_input{v, k} = A_input{v, k-1} * W_opt;
                            end
                        end
                        
                    case 'B'
                        % 算法 B: 全阶优化
                        A_temp = construct_Graph(X, 5, K_curr);
                        A_input = A_temp;
                        for v = 1:V_num
                            for k = 1:K_curr
                                A_input{v, k} = optimize_Z(A_temp{v, k}, alpha_param);
                                A_input{v, k} = (A_input{v, k} + A_input{v, k}') / 2;
                            end
                        end
                        
                    case 'C'
                        % 算法 C: 后置优化 (输入不处理)
                        A_input = construct_Graph(X, 5, K_curr);
                end
                
                % [步骤 3] Solver 循环
                for m_loop = 1:length(mu)
                    for b_loop = 1:length(beta)
                        
                        % 计时开始
                        tic;
                        [S_mo, obj, H] = solver(X, num, V_num, mu(m_loop), lambda, beta(b_loop), K_curr, c, A_input);
                        run_time = toc;
                        
                        S_mo(S_mo < 1e-5) = 0;
                        S_mo = (S_mo + S_mo') / 2;
                        
                        S_final = S_mo;
                        if strcmp(ALGO_MODE, 'C')
                            S_final = optimize_Z(S_mo, alpha_param);
                            S_final = (S_final + S_final') / 2;
                        end
                        
                        % 聚类
                        G_temp = graph(S_final);
                        bins = conncomp(G_temp);
                        final_labels = bins';
                        
                        if length(unique(final_labels)) ~= c
                             % warning('启用 K-means 修正...'); % 减少刷屏，可注释
                             L_final = diag(sum(S_final, 2)) - S_final;
                             [eigvec, ~] = eigs(L_final, c, 'smallestabs'); 
                             final_labels = kmeans(eigvec, c, 'Replicates', 10);
                        end
                        
                        metrics = ClusteringMeasure_new(Y, final_labels);
                        
                        % 记录单次结果
                        res_struct = struct();
                        res_struct.Dataset = dataset_name;
                        res_struct.Mode = ALGO_MODE;
                        res_struct.Params = [K_curr, mu(m_loop), beta(b_loop), alpha_param];
                        res_struct.Metrics = metrics;
                        res_struct.Time = run_time;
                        results = [results, res_struct];
                        
                        fprintf('  -> 完成: ACC=%.4f, NMI=%.4f (耗时 %.2fs)\n', ...
                            metrics.ACC, metrics.NMI, run_time);
                    end
                end
            end
            
            % --- 4. 只有当顺利跑完所有参数后，才保存文件 ---
            save(save_path, 'results');
            fprintf('  >> 已保存结果到: %s\n', save_name);
            
        catch ME
            % 如果出错，记录错误日志，不中断主循环
            err_msg = sprintf('Error in %s Mode %s: %s', dataset_name, ALGO_MODE, ME.message);
            fprintf(2, '%s\n', err_msg);
            
            fid = fopen(fullfile(result_dir, 'error_log.txt'), 'a');
            fprintf(fid, '%s - %s\n', datestr(now), err_msg);
            fclose(fid);
        end
    end
end

fprintf('\n所有任务执行完毕。\n');