function result = ClusteringMeasure_new(Y, predY)
% Y: 真实标签 (Ground Truth)
% predY: 预测标签 (Predicted Labels)

    if size(Y, 1) == 1; Y = Y'; end
    if size(predY, 1) == 1; predY = predY'; end

    n = length(Y);

    % 1. ACC (需要 bestMap 函数，通常您的 utils 里应该有)
    % 如果没有 bestMap，ACC 会报错。鉴于您之前能跑出 ACC，说明您有这个环境。
    try
        new_predY = bestMap(Y, predY);
        result.ACC = sum(Y == new_predY) / n;
    catch
        % 备用方案（如果没有 bestMap）
        result.ACC = 0; 
        warning('未找到 bestMap 函数，跳过 ACC 计算');
    end

    % 2. NMI (Normalized Mutual Information)
    result.NMI = compute_nmi(Y, predY);

    % 3. Purity
    result.Purity = compute_purity(Y, predY);

    % 4. AR (Adjusted Rand Index) & 5. F-score
    [AR, RI, MI, HI, Fscore] = compute_fscore_ar(Y, predY);
    result.AR = AR;
    result.Fscore = Fscore;
    % result.RI = RI; % 如果需要 Rand Index 可取消注释
end

% ------------------------------------------------
% 内部子函数 (直接拷进去，不需要额外文件)
% ------------------------------------------------

function [AR,RI,MI,HI,Fscore]=compute_fscore_ar(Y, predY)
    % 基于列联表计算 AR 和 F-score
    % 这种方法比两两比较快得多
    
    Classes = unique(Y);
    Clusters = unique(predY);
    n = length(Y);
    
    % 构建混淆矩阵 (Contingency Table)
    % C(i,j) 表示属于真实类 i 且被聚类到 j 的样本数
    nClass = length(Classes);
    nClus = length(Clusters);
    
    % 快速构建矩阵
    % C = crosstab(Y, predY); % MATLAB 自带函数，但为了兼容性手写如下:
    C = zeros(nClass, nClus);
    % 映射标签到 1..k
    [~, ~, mapY] = unique(Y);
    [~, ~, mapP] = unique(predY);
    for i = 1:n
        C(mapY(i), mapP(i)) = C(mapY(i), mapP(i)) + 1;
    end
    
    % 计算基本统计量
    n_dot_j = sum(C, 1); % 列和
    n_i_dot = sum(C, 2); % 行和
    
    term1 = sum(sum(C.^2));
    term2 = sum(n_dot_j.^2);
    term3 = sum(n_i_dot.^2);
    
    % --- 计算 AR (Adjusted Rand Index) ---
    numer = 2 * (n*(n-1)/2 - (sum(n_i_dot.^2) + sum(n_dot_j.^2))/2 + sum(sum(C.^2)));
    % 下面是标准 AR 公式
    a = (term1 - n)/2; % TP
    b = (term3 - term1)/2; % FN
    c = (term2 - term1)/2; % FP
    d = (n*(n-1)/2) - a - b - c; % TN
    
    RI = (a + d) / (a + b + c + d); 
    
    % AR 公式
    Index = sum(sum(nchoosek2(C, 2)));
    ExpectedIndex = sum(nchoosek2(n_i_dot, 2)) * sum(nchoosek2(n_dot_j, 2)) / nchoosek2(n, 2);
    MaxIndex = 0.5 * (sum(nchoosek2(n_i_dot, 2)) + sum(nchoosek2(n_dot_j, 2)));
    AR = (Index - ExpectedIndex) / (MaxIndex - ExpectedIndex);
    
    % --- 计算 F-score ---
    % Precision = TP / (TP + FP) = a / (a+c)
    % Recall = TP / (TP + FN) = a / (a+b)
    P = a / (a + c);
    R = a / (a + b);
    if (P+R) == 0
        Fscore = 0;
    else
        Fscore = 2 * P * R / (P + R);
    end
    
    MI = 0; HI = 0; % 占位
end

function val = nchoosek2(x, k)
    % 快速计算 nchoosek(x, 2)，支持向量输入
    % x*(x-1)/2
    if k ~= 2
        error('Only support k=2 for fast calculation');
    end
    val = x .* (x-1) / 2;
end

function score = compute_nmi(A, B)
    % 标准 NMI 计算
    total = length(A);
    A_ids = unique(A);
    B_ids = unique(B);
    MI = 0;
    for idA = A_ids'
        for idB = B_ids'
            idAOccur = find(A == idA);
            idBOccur = find(B == idB);
            idABOccur = intersect(idAOccur,idBOccur);
            px = length(idAOccur)/total;
            py = length(idBOccur)/total;
            pxy = length(idABOccur)/total;
            if pxy > 0
                MI = MI + pxy*log2(pxy/(px*py));
            end
        end
    end
    Hx = 0; for idA = A_ids'; idAOccurCount = length(find(A == idA)); Hx = Hx - (idAOccurCount/total) * log2(idAOccurCount/total); end
    Hy = 0; for idB = B_ids'; idBOccurCount = length(find(B == idB)); Hy = Hy - (idBOccurCount/total) * log2(idBOccurCount/total); end
    score = 2 * MI / (Hx+Hy);
end

function purity = compute_purity(Y, predY)
    % 计算纯度
    labels = unique(Y);
    pred_labels = unique(predY);
    sum_max = 0;
    for i = 1:length(pred_labels)
        cluster_mask = (predY == pred_labels(i));
        cluster_true_labels = Y(cluster_mask);
        % 找出该簇中出现最多的真实标签
        [u, ~, j] = unique(cluster_true_labels);
        counts = accumarray(j, 1);
        if ~isempty(counts)
            sum_max = sum_max + max(counts);
        end
    end
    purity = sum_max / length(Y);
end