%Clear workspace
clear all

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

%calculate model training time part 1
tic

%train model
prior = [0.8 0.2];
nb = fitcnb(Xtrain,Ytrain,'Prior',prior,'Distribution','mvmn');

%calculate model training time part 2
timeElapsed = toc

yhat = resubPredict(nb);

cvmodelnb = crossval(nb);
L = kfoldLoss(cvmodelnb);
trainError = resubLoss(nb);
trainAccuracy = 1-trainError

%formatting for confusion matrix
Ytraincell = table2array(Ytrain);
con = confusionmat(yhat,Ytraincell);

%plot area under curve
[X,Y,T,AUC,OPTROCPT] = perfcurve(Ytraincell,yhat,'1');
plot(X,Y)
xlabel('X')
ylabel('Y')
title('AUC')

%save model to be run on test set
save('nb')