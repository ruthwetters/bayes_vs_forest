% Feature Selection for Naive Bayes using Minimum-Redundancy Maximum Relevance 
% Method based on Mutual Information
% Comparison between mRMR and chi-squared feature ranking methods

clear all
data = readtable('/Users/ruthwetters/Downloads/dfdummy.csv');

%remove Var1 which is the index: as dataset is ordered by class, including
%Var1 means the algorithm can just separate on index
data.Var1=[];
% Data Shape
[m n] = size(data);

featureNames = ['Class','age','menopause'];

% Feature Selection and Ranking using Chi-Squared
[idx,scores] = fscchi2(data,'Class');

% Feature Selection and Ranking using mRMR
[idx2,scores2] = fscmrmr(data,'Class');

%Visualisation of Chi-Squared
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
xticklabels(strrep(data.Properties.VariableNames(idx),'_','\_'))
xtickangle(45)
title('Chi-Squared Feature Ranking')