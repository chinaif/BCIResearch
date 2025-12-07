function A = construct_Graph(X, k, d)
% CONSTRUCT_GRAPH 构建多视图多阶图的基础函数
% X: 数据 cell 数组
% k: 近邻数
% d: 阶数 (Order)

n = size(X{1}, 1);
V = length(X);
A = cell(V, d);

for v = 1 : V
    % 计算距离矩阵
    D_mat = L2_distance_1(X{v}', X{v}');
    [dumb, idx] = sort(D_mat, 2); 
    
    W = zeros(n,n);
    for i = 1:n
        di = dumb(i, 2:k+2);
        id = idx(i, 2:k+2);
        % 自适应高斯核 (PKN 变体)
        W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
    
    % 1阶图
    P = (W+W')/2;
    A{v, 1} = P;
    
    % 高阶图传播 (A^k = A^(k-1) * A^1)
    for i = 2 : d
         A{v,i} = A{v, i-1} * P;
    end
end
end