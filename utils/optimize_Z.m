function Z = optimize_Z(G, alpha)
% 流形优化核心函数
% min Tr(Z' * L * Z) + alpha * ||Z - I||^2

    num = size(G, 1);
    D_vec = sum(G, 2);
    D_inv_sqrt = diag(1 ./ sqrt(D_vec + eps));
    S_norm = D_inv_sqrt * G * D_inv_sqrt;
    L = eye(num) - S_norm;
    
    H_qp = 2 * (L + alpha * eye(num));
    H_qp = (H_qp + H_qp') / 2;
    
    options = optimset('Algorithm', 'interior-point-convex', 'Display', 'off');
    Aeq = ones(1, num);
    beq = 1;
    lb = zeros(num, 1);
    
    Z = zeros(num, num);
    % 建议: 如果数据量大，将 for 改为 parfor
    for i = 1:num
        e_i = zeros(num, 1);
        e_i(i) = 1;
        f = -2 * alpha * e_i;
        [z_i, ~] = quadprog(H_qp, f, [], [], Aeq, beq, lb, [], [], options);
        Z(i, :) = z_i';
    end
end