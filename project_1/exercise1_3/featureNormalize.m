function [X_norm, mu, sigma] = featureNormalize(X)
%   featureNormalize normalizes the features in X.
%   [X_norm, mu, sigma] = FEATURENORMALIZE(X) returns a normalized version 
%   of X where the mean value of each feature is 0 and the standard
%   deviation is 1. It also returns the mean and standard deviation
%   calculated from the original data.
%
%   X: Input matrix where each row represents a data point and each
%      column represents a feature.
%
%   X_norm - Normalized version of X.
%   mu     - Mean of each feature in X.
%   sigma  - Standard deviation of each feature in X.

% Calculate mean and standard deviation of each feature
mu = mean(X); % Calculate mean of each feature
sigma = std(X); % Calculate standard deviation of each feature

% Normalize the features
X_norm = (X - mu) ./ sigma; % Normalize each feature using mean and standard deviation

end
