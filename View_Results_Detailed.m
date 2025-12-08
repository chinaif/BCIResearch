clear; clc;

% 设置结果文件夹路径
result_dir = 'Experiment_Results';
files = dir(fullfile(result_dir, 'Res_*.mat'));

% 初始化容器
summary_data = {};

fprintf('正在扫描 %d 个结果文件...\n', length(files));

for i = 1:length(files)
    % 1. 加载数据
    filepath = fullfile(files(i).folder, files(i).name);
    try
        load(filepath); % 载入 'results' 变量
    catch
        warning('文件损坏无法读取: %s', files(i).name);
        continue;
    end
    
    if isempty(results)
        continue;
    end

    % 2. 寻找最佳结果 (默认以 ACC 最高为基准)
    % 注意：results 是一个结构体数组，包含该数据集下所有参数组合的结果
    all_acc = arrayfun(@(x) x.Metrics.ACC, results);
    [max_acc, best_idx] = max(all_acc);
    best_res = results(best_idx);
    
    % 3. 提取基本信息
    d_name = best_res.Dataset;
    mode_name = best_res.Mode;
    
    % 4. 提取参数 (根据 Run_All_Experiments 里的保存顺序)
    % Params = [K_curr, mu, beta, alpha]
    param_K = best_res.Params(1);
    param_mu = best_res.Params(2);
    param_beta = best_res.Params(3);
    
    % 5. 提取运行时间
    run_time = best_res.Time;
    
    % 6. 智能提取所有 Metrics (防止不同工具包字段名不一样)
    metrics = best_res.Metrics;
    
    % 必选指标
    val_acc = metrics.ACC;
    val_nmi = metrics.NMI;
    
    % 可选指标 (尝试获取，如果没有则填 NaN)
    if isfield(metrics, 'Purity'), val_purity = metrics.Purity; else, val_purity = NaN; end
    if isfield(metrics, 'Fscore'), val_fscore = metrics.Fscore; else, val_fscore = NaN; end
    if isfield(metrics, 'AR'), val_ar = metrics.AR; else, val_ar = NaN; end
    
    % 7. 存入表格行
    % 格式: Dataset | Mode | ACC | NMI | Purity | F-score | AR | Time | Best_Mu | Best_Beta
    summary_data(end+1, :) = { ...
        d_name, ...
        mode_name, ...
        val_acc, ...
        val_nmi, ...
        val_purity, ...
        val_fscore, ...
        val_ar, ...
        run_time, ...
        param_mu, ...
        param_beta ...
    };
end

% 8. 转换为 Table 并显示
VarNames = {'Dataset', 'Mode', 'ACC', 'NMI', 'Purity', 'Fscore', 'AR', 'Time_s', 'Best_Mu', 'Best_Beta'};
T = cell2table(summary_data, 'VariableNames', VarNames);

% 排序：先按数据集名排，再按模式排
T = sortrows(T, {'Dataset', 'Mode'});

% 显示表格
disp(T);

% 9. (可选) 导出为 Excel 方便写论文
% writetable(T, 'Full_Experiment_Report.xlsx');
% fprintf('已导出完整报表到 Excel。\n');