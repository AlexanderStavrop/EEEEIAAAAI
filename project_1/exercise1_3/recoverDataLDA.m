function [X_rec] = recoverDataLDA(Z, v)

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), length(v));

X_rec = Z .* v(:, 1)';

end