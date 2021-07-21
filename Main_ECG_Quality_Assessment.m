%  Author:      Guoyang Liu
%  @Software:   MATLAB 2020a/b
%  Please cite: G. LIU, X. HAN, L. TIAN, W. ZHOU, and H. LIU, "ECG Quality Assessment Based on Hand-Crafted Statistics and Deep-Learned S-Transform Spectrogram Features," Computer Methods and Programs in Biomedicine, p. 106269, 2021.

clear;

addpath(genpath('D:\GraduateFiles\STUDY\ECG\'));

% load FormatData_setAB_2000.mat X Y;
load FormatData_setA_998.mat X Y;

%% Generate Fold Index
% nFold = 10;
% classNames = unique(Y);
% cvIndex = cell(1, numel(classNames));
% for iClass = 1:numel(classNames)
%     cvIndex{iClass} = crossvalind('Kfold',sum(Y==classNames(iClass)),nFold);
% end
% save cvIndex_10Fold_combine.mat cvIndex classNames nFold;

%% Load Fold Index
% load cvIndex_10Fold_combine.mat cvIndex classNames nFold;
load cvIndex_10Fold.mat cvIndex classNames nFold;

%% Cross Validation
YPred_All = [];
YTest_All = [];
for iFold = 1:nFold
    %% Depart Fold Data
    XTrain = [];
    YTrain = [];
    XTest = [];
    YTest = [];
    for iClass = 1:numel(classNames)
        testIndex = cvIndex{iClass} == iFold;
        trainIndex = ~testIndex;
        X_curClass = X(:,:,Y==classNames(iClass));
        Y_curClass = Y(Y==classNames(iClass));
        XTrain = cat(3, XTrain, X_curClass(:,:,trainIndex));
        YTrain = cat(1,YTrain, Y_curClass(trainIndex));
        XTest = cat(3, XTest, X_curClass(:,:,testIndex));
        YTest = cat(1,YTest, Y_curClass(testIndex));
    end
    
    %% Format HE Data
    disp(['Hand Extraction...']);
    tic;
    parfor iSample = 1:size(XTrain,3)
        XTrain_HE_raw(:,iSample) = ExtractHEFeatures(XTrain(:,:,iSample));
    end
    parfor iSample = 1:size(XTest,3)
        XTest_HE_raw(:,iSample) = ExtractHEFeatures(XTest(:,:,iSample));
    end
    
    nCh = size(XTrain, 2);
%     XTrain_HE = squeeze(cell2mat(cellfun(@ExtractHEFeatures, num2cell(XTrain, [1 2]), 'UniformOutput', false)));
%     XTest_HE = squeeze(cell2mat(cellfun(@ExtractHEFeatures, num2cell(XTest, [1 2]), 'UniformOutput', false)));
    XTrain_HE = XTrain_HE_raw';
    XTest_HE = XTest_HE_raw';
    XTrain_HE = reshape(XTrain_HE, size(XTrain_HE,1), [], nCh);
    XTest_HE = reshape(XTest_HE, size(XTest_HE,1), [], nCh);
    XTrain_HE = permute(XTrain_HE, [4 2 3 1]);
    XTest_HE = permute(XTest_HE, [4 2 3 1]);
    clearvars XTrain_HE_raw XTest_HE_raw;
    t = toc;
    disp(['Hand Extraction...done! Time: ' num2str(t) 's']);
    
    %% Format DL Data
    XTrain_DL = permute(XTrain, [4 1 2 3]);
    XTest_DL = permute(XTest, [4 1 2 3]);
    
    %% Concatnate Data
    XTrain_Mix = cat(2,XTrain_HE,XTrain_DL);
    XTest_Mix = cat(2,XTest_HE,XTest_DL);
    
    %% Format Label
    YTrain = categorical(YTrain,[1 2],{'Acceptable' 'Unacceptable'});
    YTest = categorical(YTest,[1 2],{'Acceptable' 'Unacceptable'});
    
    %% Train DL Network
    opt.inputSize = [size(XTrain_Mix,1) size(XTrain_Mix,2) size(XTrain_Mix,3)];
    opt.filterSize = [3 3];
    opt.nDepth = 3;
    opt.nInitalFilters = 32;
    opt.classWeights = [1 sum(YTrain=='Acceptable')/sum(YTrain=='Unacceptable')];
%     load lgraph_onlyHE lgraph;
    
    lgraph = buildNetwork(opt);
    [net, trainInfo] = trainModel(XTrain_Mix,YTrain,XTest_Mix,YTest,lgraph);

    repN = 2;
    YScore = zeros(numel(YTest), 2, repN);
    for iN = 1:repN
        [~, YScore(:,:,iN)] = classify(net, XTest_Mix);
    end
    YScore = mean(YScore, 3);
    [~, YLabel] = max(YScore, [], 2);
    YPred = categorical(YLabel,[1 2],{'Acceptable' 'Unacceptable'});
    
%     YPred = classify(net, XTest_Mix);
    
    C = confusionmat(YTest,YPred,'Order',{'Acceptable' 'Unacceptable'});
    YPred_All = cat(1,YPred_All,YPred);
    YTest_All = cat(1,YTest_All,YTest);
    Correct_Fold(iFold) = sum(C.*eye(2), 'all');
    All_Fold(iFold) = sum(C, 'all');
    Acc_Fold(iFold) = Correct_Fold(iFold)./All_Fold(iFold);
    Net_Fold{iFold} = net; 
    trainInfo_Fold{iFold} = trainInfo; 
%     figure;
%     plotconfusion(YTest,YPred);
    
end

%% Summary Result
Accuracy_All = sum(Correct_Fold)./sum(All_Fold);
figure;
plotconfusion(YTest_All,YPred_All);

save result.mat


