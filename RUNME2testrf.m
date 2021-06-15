%Clear workspace
clear all

load rf.mat

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

%test model
yhat = predict(rf,Xtrain);

%calculate error + accuracy
treeerror = oobError(rf,'Mode','Ensemble')
treeaccuracy = 1-treeerror

%changing cell types for compatibility
Ytraincell = table2array(Ytrain);
yhatmat = str2double(yhat);

%plot confusion matrix
con = confusionmat(yhatmat,Ytraincell)

[X,Y,T,AUC,OPTROCPT] = perfcurve(Ytraincell,yhatmat,'1')
plot(X,Y)
save('rf')

%calculate model testing time part 2
timeElapsed = toc