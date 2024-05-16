function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
    
    [NumSamples NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes
    
    A=zeros(NumFeatures,NewDim);
    %For each class i
    %Find the necessary statistics

    % Initialize variables for within-class scatter matrix
    Sw = zeros(NumFeatures, NumFeatures);
    Sb = zeros(NumFeatures, NumFeatures);

    % Extracting the unique numbers (Classes)    
    for i = 1:NumClasses
        % Extracting how many of each class are in
        freq = sum(Labels == Classes(i));
        
        % Calculate the Class Prior Probability
        P(i) = freq/length(Labels);

        % Calculate the Class Mean
        mu(i, :) = mean(Samples(Labels == Classes(i), :));

        % Calculate the Global Mean
        m0= mean(Samples);

        % Extract samples of class i
        SamplesOfClass = Samples(Labels == Classes(i), :);
        
        % Calculate the class mean
        mu(i, :) = mean(SamplesOfClass);
        
        % Update within-class scatter matrix
        Sw = Sw + (SamplesOfClass - mu(i, :))' * (SamplesOfClass - mu(i, :));

        Sb = Sb + P(i) * (mu(i, :) - m0)' * (mu(i, :) - m0);
    end

    % Calculate the eigenvectors and eigenvalues of Sw^(-1) * Sb
    [V, D] = eig(inv(Sw) * Sb);

    % Sort eigenvectors according to eigenvalues
    [~, ind] = sort(diag(D), 'descend');
    A = V(:, ind(1:NewDim));




    % %Calculate the Between Class Scatter Matrix
    % Sb= 
    % 
    % %Eigen matrix EigMat=inv(Sw)*Sb
    % EigMat = inv(Sw)*Sb;
    % 
    % %Perform Eigendecomposition
    % 
    % 
    % %Select the NewDim eigenvectors corresponding to the top NewDim
    % %eigenvalues (Assuming they are NewDim<=NumClasses-1)
    % %% You need to return the following variable correctly.
    % A=zeros(NumFeatures,NewDim);  % Return the LDA projection vectors
