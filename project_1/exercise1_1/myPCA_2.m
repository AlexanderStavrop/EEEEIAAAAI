function [ eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [ eigenval, eigenvec, order] = mypca(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order 
%

% Useful values
[m, n] = size(X);

% Make sure each feature from the data is zero mean
%X_centered = X - mean(X);

% Get the number of columns of X
num_columns = size(X, 2);
mu = mean(X);

% Normalize each feature
X_centered = zeros(size(X)); % Initialize X_norm with the same size as X

for column_no = 1:num_columns
    X_centered(:,column_no) = (X(:,column_no) - mu(column_no));
end

% Calculate the covariance matrix
Sigma = 1/m * X_centered' * X_centered);

% Compute eigenvectors and eigenvalues
[eigenvec, eigenval] = eig(Sigma);

% Convert eigenvalues from diagonal matrix to vector
eigenval = diag(eigenval);

% Sort eigenvalues and eigenvectors in descending order
[eigenval, order] = sort(eigenval, 'descend');
eigenvec = eigenvec(:, order);
