clear all 
close all
clc

T = readtable('data.csv');
X=T{:,3:end};
y=T{:,2};
[a,b]=size(y);
[A,B]=size(X);
Y=zeros(a,1);
for i=1:a
   if (y{i}=='B')
       Y(i)=1;
   
   end
end
X=X';
Y=Y';


numOfActivationNodes = B;       
optimizationFnc      = 'traingd'; 
net                  = patternnet(numOfActivationNodes, optimizationFnc);
 net.divideParam.trainRatio = 0.7; 
net.divideParam.testRatio  = 0.2; 
net.divideParam.valRatio   = 0.1; 
net.trainParam.epochs      = 5;
net.performFcn             = 'crossentropy';

[trainedNet, trainingRecord] = train(net, X, Y);

ypred = trainedNet(X);


error = perform(trainedNet, Y, ypred);
figure;
plotperf(trainingRecord);

figure;
plotconfusion(ypred, Y);
title("Shallow neural network results");
xlabel("True"); ylabel("Prediction");
xticklabels(["Benign", "Malignant"]);
yticklabels(["Benign", "Malignant"]);