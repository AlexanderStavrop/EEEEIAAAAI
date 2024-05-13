function [eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [eigenval, eigenvec, order] = myPCA(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order

% Useful values
[m, n] = size(X);

% Make sure each feature from the data is zero mean
X_centered = X - mean(X);

% Compute the covariance matrix
Sigma = (1 / m) * (X_centered' * X_centered);

% Compute eigenvectors and eigenvalues
[eigenvec, eigenval] = eig(Sigma);

% Sort eigenvalues in descending order
[eigenval, order] = sort(diag(eigenval), 'descend');
eigenvec = eigenvec(:, order);

end