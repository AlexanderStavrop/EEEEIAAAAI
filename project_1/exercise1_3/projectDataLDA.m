function [Z] = projectDataLDA(X, v)

% Initialize Z to store the projected data
Z = zeros(size(X, 1), size(v, 2));

% Project the data onto the LDA space
Z = X * v;

end

