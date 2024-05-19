function [Z] = projectDataLDA(X, v)
    % projectDataLDA Projects the data X onto the LDA direction vector v
    %
    % This function projects the input data matrix X onto the linear discriminant
    % direction vector v obtained from LDA. The result is a new data matrix Z
    % where the data is represented in the lower-dimensional LDA space.
    %
    % Inputs:
    %   X - A matrix where each row represents an observation and each column
    %       represents a feature.
    %   v - The LDA direction vector computed using fisherLinearDiscriminant.
    %
    % Output:
    %   Z - The matrix of projected data.

    % Initialize Z to store the projected data
    Z = zeros(size(X, 1), size(v, 2));

    % Project the data onto the LDA space
    Z = X * v;
end

