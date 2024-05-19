function v = fisherLinearDiscriminant(X1, X2)
    % fisherLinearDiscriminant Computes the Fisher Linear Discriminant vector
    % for two classes given by X1 and X2.
    %
    % This function calculates the optimal linear direction that maximizes
    % the separation between the two classes. The direction vector 'v' is
    % computed to maximize the ratio of the between-class variance to the
    % within-class variance.

    % Inputs:
    %   X1 - A matrix where each row represents an observation of class 1.
    %   X2 - A matrix where each row represents an observation of class 2.
    %
    % Output:
    %   v  - The optimal direction vector for maximum class separation.

    % Extracting the size of each class
    m1 = size(X1, 1);
    m2 = size(X2, 1);

    % Calculating the mean of each class
    mu1 = mean(X1);
    mu2 = mean(X2);
    
    % Calculating the scatter matrix for each class
    S1 = (X1 - mu1)' * (X1 - mu1) / m1;
    S2 = (X2 - mu2)' * (X2 - mu2) / m2;
    
    % Calculating Within class scatter matrix
    Sw = S1 + S2;

    % Calculating the optimal direction for maximum class separation
    v = inv(Sw) * (mu1 - mu2)';

    % Normilizeing the optimal direction vector
    v = v / norm(v);
end