%Clear workspace
clear all

load nb.mat

%import data
data = readtable('/Users/ruthwetters/Downloads/ML Submission/Data/dfnamed1.csv');
[m n] = size(data);

%Split into test + train
P = 0.70;
rng('default'); %for reproducability
idx=randperm(m);
Trainingset=data(idx(1:round(P*m)),:);
Testingset=data(idx(round(P*m)+1:end),:);

Xtrain = Trainingset(:,3:end);
Ytrain = Trainingset(:,2);
Xtest = Testingset(:,3:end);
Ytest = Testingset(:,2);

%calculate model testing time part 1
tic

%predict result
yhat = predict(nb,Xtest);

cvmodelnb = crossval(nb);
L = kfoldLoss(cvmodelnb);
trainError = resubLoss(nb);
trainAccuracy = 1-trainError

%formatting for confusion matrix
Ytestcell = table2array(Ytest);
con = confusionmat(yhat,Ytestcell)

%plot area under curve
[X,Y,T,AUC,OPTROCPT] = perfcurve(Ytestcell,yhat,'1');
plot(X,Y)
xlabel('X')
ylabel('Y')
AUC
title('AUC')

%calculate model testing time part 2
timeElapsed = toc