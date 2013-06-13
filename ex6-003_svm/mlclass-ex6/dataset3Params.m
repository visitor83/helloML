function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

candidateParam = [0.01 0.03 0.1 0.3 1 3 10 30];


K = 0;
K_error = 0.0; K_C = 0.0; K_sigma = 0.0; K_error_old = 0.0;

size(candidateParam, 2)

for i = 1: size(candidateParam, 2)
    C = candidateParam(i); 

    for j = 1: size(candidateParam, 2)
        sigma = candidateParam(j);

        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict (model, Xval);

        K_error = mean(double(predictions ~= yval));
        % First Time Update K_error_old
        if K == 0
          K_error_old = K_error;
          K = K + 1;
          K_C = C; K_sigma = sigma;
        end

        fprintf("C=%0.4f, sigma=%0.4f, K_error=%0.4f, K_error_old=%0.4f\n", C, sigma, K_error, K_error_old);
        if K_error < K_error_old
          K_C = C; K_sigma = sigma;
          K_error_old = K_error;
          fprintf("Update C and sigma\n");
        end
    end 


    %error_cv (K, 1) = mean(double(predictions ~= yval));
    %error_cv (K, 2) = C;
    %error_cv (K, 3) = sigma;
    %K = K + 1;

end 

% Find the minimum  mean 
K_error_old
C = K_C 
sigma = K_sigma

%error_cv(min(error_cv(i, 1));



% =========================================================================

end
