function [X_rec] = recoverDataLDA(Z, v)
%   recoverDataLDA Recovers the original data from the LDA projection.
%   X_rec = recoverDataLDA(Z, v) recovers the original data matrix from the
%   LDA (Linear Discriminant Analysis) projection Z using the projection
%   vector v.
%
% Inputs:
%   Z - The projected data matrix obtained from LDA. Each row represents a
%       data point, and each column represents a feature.
%   v - The projection vector obtained from LDA. It represents the direction
%       in which the data is projected.
%
% Output:
%   X_rec - The recovered data matrix, which is the original data
%           reconstructed from the LDA projection. It has the same dimensions
%           as the input matrix Z.

% Initializing the recovered data matrix.
X_rec = zeros(size(Z, 1), length(v));

% Reconstructing the original data by element-wise multiplication of the
% projected data Z with the projection vector v.
X_rec = Z .* v(:, 1)';

end
