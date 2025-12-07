clear; clc; close all;
addpath('utils');
addpath('datasets');
data_path = 'D:\if\Research\Code\Multi-view-datasets-master';
addpath(data_path);

% ==========================================
%           数据集列表 (备选库)
% ==========================================
% 请取消注释您想要使用的一行:

% --- Text / Document (文本类) ---
% dataset_name = '3Sources';
% dataset_name = 'ACM';
% dataset_name = 'BBCSport';
% dataset_name = 'BBC4view';
% dataset_name = 'CiteSeer';
% dataset_name = 'Cora';
% dataset_name = 'Movies';
% dataset_name = 'Reuters';
% dataset_name = 'Reuters-1200';
% dataset_name = 'Reuters-1500';
% dataset_name = 'WebKB';
% dataset_name = 'Wikipedia';
% dataset_name = 'Wikipedia-test';

% --- Image / Object (图像物体类) ---
% dataset_name = '100Leaves';
% dataset_name = 'ALOI';
% dataset_name = 'ALOI-1k';
% dataset_name = 'Animal';
dataset_name = 'Caltech101-7';
% dataset_name = 'Caltech101-20';
% dataset_name = 'Caltech101-all';
% dataset_name = 'COIL20';
% dataset_name = 'MSRC-v5';
% dataset_name = 'NUS-WIDE';
% dataset_name = 'NUS-WIDE-OBJ';
% dataset_name = 'OutdoorScene';

% --- Handwritten / Face (手写体与人脸) ---
% dataset_name = 'Handwritten';
% dataset_name = 'MNIST-4';
% dataset_name = 'MNIST-10k';
% dataset_name = 'UCI';
% dataset_name = 'ORL';
% dataset_name = 'Yale';

% --- Biology / Other (生物信息与其他) ---
% dataset_name = 'Prokaryotic';
% dataset_name = 'ProteinFold';

% ==========================================

load([dataset_name, '.mat']);

if exist('y', 'var') && ~exist('Y', 'var')
    Y = y;
end
if exist('truth', 'var') && ~exist('Y', 'var') % 有些数据集可能叫 truth
    Y = truth;
end
if exist('gnd', 'var') && ~exist('Y', 'var')   % 有些数据集可能叫 gnd
    Y = gnd;
end

% 选择算法模式: 'A', 'B', 'C'
% 'A': Pre-Optimization (构图后立即优化输入图)
% 'B': Standard/Independent (标准求解，或特定中间态)
% 'C': Post-Optimization (求解后对结果图进行优化)
ALGO_MODE = 'A'; 

% 通用参数
K_list = [2];         % 阶数
mu = 10.^[1];         % 稀疏性参数
lambda = 2^10;        % 谱约束参数
beta = 10.^[-3];      % 分布一致性参数
alpha_param = 10;     % 流形优化参数 (用于 A 和 C)

% --- 数据预处理 ---
% 数据标准化 (Z-score)
for i = 1 : length(X)
    X{i} = full((X{i} - mean(X{i}, 2)) ./ repmat(std(X{i}, [], 2), 1, size(X{i}, 2)));
end
num = size(X{1}, 1);
V_num = length(X);
c = length(unique(Y));

results = [];


fprintf('当前运行数据集: %s\n', dataset_name);
fprintf('当前运行模式: 算法 %s\n', ALGO_MODE);

for k_idx = 1:length(K_list)
    K_curr = K_list(k_idx);
    
    % [步骤 1] 基础构图 (所有算法通用)
    % 生成原始的多视图多阶图 A_base
    fprintf('正在构建基础图 (K=%d)...\n', K_curr);
    A_input = construct_Graph(X, 5, K_curr); 
    
    % [步骤 2] 算法分支处理 (严格对应您的三种构思)
    switch ALGO_MODE
        case 'A' 
            % --- 算法 A: 源头优化 (Source Optimization) ---
            % 逻辑：只优化初始的 1 阶图。高阶图由优化后的 1 阶图自然传播生成，传播过程不干预。
            fprintf('正在执行算法 A: 仅优化初始视图，再进行传播...\n');
            
            % 1. 先构建基础的 1 阶图 (W)
            % 注意：这里我们将 k 设为 1，仅获取基础图
            A_base = construct_Graph(X, 5, 1); 
            
            A_input = cell(V_num, K_curr);
            for v = 1:V_num
                % 2. 核心：对 1 阶图进行流形拓扑优化
                W_opt = optimize_Z(A_base{v, 1}, alpha_param);
                W_opt = (W_opt + W_opt') / 2; % 对称化
                
                A_input{v, 1} = W_opt;
                
                % 3. 传播：基于优化后的 W_opt 生成高阶图 (不再进行额外优化)
                % A^k = A^(k-1) * A^1_opt
                for k = 2:K_curr
                    A_input{v, k} = A_input{v, k-1} * W_opt;
                    % 注意：这里不再调用 optimize_Z，否则就变成算法 B 了
                end
            end
            
        case 'B'
            % --- 算法 B: 全阶优化 (Full-Order Optimization) ---
            % 逻辑：计算出每一阶图后，立即对其进行流形优化。
            fprintf('正在执行算法 B: 对每一阶视图都进行优化...\n');
            
            % 1. 先构建所有阶的基础图 (含传播)
            A_temp = construct_Graph(X, 5, K_curr);
            A_input = A_temp;
            
            for v = 1:V_num
                for k = 1:K_curr
                    % 2. 核心：对每一个 k 阶图都强行进行流形优化
                    A_input{v, k} = optimize_Z(A_temp{v, k}, alpha_param);
                    A_input{v, k} = (A_input{v, k} + A_input{v, k}') / 2;
                end
            end
            
        case 'C'
            % --- 算法 C: 后置优化 (Post-Optimization) ---
            % 逻辑：输入图不做任何流形处理，保留原汁原味的 KNN 和传播结构。
            % 优化操作留到 Solver 之后对 S 进行。
            fprintf('正在执行算法 C: 保留原始输入，等待后置优化...\n');
            
            % 直接使用标准构图
            A_input = construct_Graph(X, 5, K_curr);
    end
    
    % [步骤 3] 联合优化求解 (Solver)
    for m_idx = 1:length(mu)
        for b_idx = 1:length(beta)
            
            % 调用通用 Solver
            % 注意：如果算法 A 已经优化过 A_input，Solver 用的就是优化后的图
            [S_mo, obj, H] = solver(X, num, V_num, mu(m_idx), lambda, beta(b_idx), K_curr, c, A_input);
            
            % 清理数值噪声
            S_mo(S_mo < 1e-5) = 0;
            S_mo = (S_mo + S_mo') / 2;
            
            % [步骤 4] 结果处理分支
            S_final = S_mo; % 默认情况
            
            if strcmp(ALGO_MODE, 'C')
                % --- 算法 C 特有: 后置优化 ---
                fprintf('正在执行后置流形优化 (Alpha=%f)...\n', alpha_param);
                S_final = optimize_Z(S_mo, alpha_param);
                S_final = (S_final + S_final') / 2;
            end
            
            % [步骤 5] 聚类与评估
            % 使用 conncomp 或 spectral clustering
            G_temp = graph(S_final);
            bins = conncomp(G_temp);
            final_labels = bins';
            
            % 如果连通分量个数不对，使用 K-means 修正
            if length(unique(final_labels)) ~= c
                 warning('连通分量(%d) != 类别数(%d)，启用 K-means...', length(unique(final_labels)), c);
                 L_final = diag(sum(S_final, 2)) - S_final;
                 % 使用前 c 个最小特征向量
                 [eigvec, ~] = eigs(L_final, c, 'smallestabs'); 
                 final_labels = kmeans(eigvec, c, 'Replicates', 10);
            end
            
            % 计算指标
            metrics = ClusteringMeasure_new(Y, final_labels);
            
            % 打印结果
            fprintf('K=%d, mu=%.1f, beta=%.3f | ACC: %.4f, NMI: %.4f\n', ...
                K_curr, mu(m_idx), beta(b_idx), metrics.ACC, metrics.NMI);
            
            % 保存
            res_struct.Mode = ALGO_MODE;
            res_struct.K = K_curr;
            res_struct.Params = [mu(m_idx), beta(b_idx), alpha_param];
            res_struct.Metrics = metrics;
            results = [results, res_struct];
        end
    end
end