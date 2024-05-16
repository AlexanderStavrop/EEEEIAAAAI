function [U, S] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = principalComponentAnalysis(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S

% Useful values
[m, n] = size(X);

% Make sure each feature from the data is zero mean
X_centered = X - mean(X);

% Compute the covariance matrix
Sigma = (1 / m) * (X_centered' * X_centered);

% Compute eigenvectors and eigenvalues
[U, S] = eig(Sigma);

% Sort eigenvalues in descending order
[S, order] = sort(diag(S), 'descend');
U = U(:, order);

end
