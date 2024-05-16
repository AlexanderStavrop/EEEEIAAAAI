function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standarddeviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Calculate the mean and standard deviation of each feature
mu = mean(X);
sigma = std(X);

% Normalize the features
X_norm = (X - mu) ./ sigma;

end
