function v = fisherLinearDiscriminant(X1, X2)
    m1 = size(X1, 1);
    m2 = size(X2, 1);

    mu1 = mean(X1);
    mu2 = mean(X2);
     
    S1 = (X1 - mu1)' * (X1 - mu1) / m1; % Calculate scatter matrix of class 1
    S2 = (X2 - mu2)' * (X2 - mu2) / m2; % Calculate scatter matrix of class 2
    
    Sw = S1 + S2; % Within class scatter matrix

    v = pinv(Sw) * (mu1 - mu2)'; % Compute optimal direction for maximum class separation

    v = v / norm(v); % Normalize v to have unit norm
end