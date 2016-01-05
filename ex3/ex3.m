%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 1024;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('cifar-10-batches-mat/data_batch_1.mat');
X = double(data);
y = labels + 1;
m = size(X, 1);
Z = zeros(m,1024);
i = 1;
while i<m+1
    j = 1;
    while j < 1024
        grayscale_pixel_density = X(i,(j+(0)))*0.2989 + X(i,(j+(1024)))*0.5870 + X(i,(j+(2048)))*0.1140;
        Z(i,j) = grayscale_pixel_density;
        %disp(grayscale_pixel_density);
        j = j+1;
    end
    i = i+1;
end

save data_gray_final.mat Z;
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
X = Z;
%displayData(Z(sel, :));
%data = load ('output_removedLectureHalls.csv');
%y = data;
m = size(X, 1)
k = size (y,1)

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.5;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
save final_params_logistic_2000_0.5.mat all_theta;
fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...

%H = load('prd.csv');
pred = predictOneVsAll(all_theta, X);

disp(pred);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%disp(all_theta);
