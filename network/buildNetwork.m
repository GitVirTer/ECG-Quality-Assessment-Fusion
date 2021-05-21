%  Author:      Guoyang Liu
%  @File:       net_CNN_varDeep_1.m
%  @Software:   MATLAB 2020a/b

function lgraph = buildNetwork(opt)
lgraph = layerGraph();
inputSize = opt.inputSize;  % [1 5003 12]
filterSize = opt.filterSize;    % 3
nDepth = opt.nDepth;    % 3
nInitialFilters = opt.nInitalFilters;   % 32

tempLayers = [
    imageInputLayer(inputSize,"Name","InputLayer",'Normalization', 'none') 
    seperateFeatureLayer('SepFeaLayer') % Seperate the hand-craft statistics and raw ECGs
    
    ];
lgraph = addLayers(lgraph,tempLayers);

STLayer = mapCrop_STrand_Ap(3000,'STLayer');    % Online augmentation layer

lgraph = addLayers(lgraph,STLayer);

lgraph = connectLayers(lgraph,'SepFeaLayer/DL','STLayer');

for iDepth = 1:nDepth
    filterNumOut = nInitialFilters;   %*2^(iDepth-1);
    aLayerName = lgraph.Layers(end).Name;
    convBlock = layerBlock(filterSize, filterNumOut, iDepth);
    bLayerName = convBlock(1).Name;
    lgraph = addLayers(lgraph,convBlock);
    lgraph = connectLayers(lgraph,aLayerName,bLayerName);
end

aLayerName = lgraph.Layers(end).Name;

flatten_HE = flatten3DLayer('flatten_HE');  % Flatten the Hand-crafted statistics
lgraph = addLayers(lgraph,flatten_HE);
flatten_DL = flatten3DLayer('flatten_DL');  % Flatten the Deep-learned features
lgraph = addLayers(lgraph,flatten_DL);
depthConcatLayer = depthConcatenationLayer(2,'Name','depthConcatLayer');    % Concatenate two types of feature
lgraph = addLayers(lgraph,depthConcatLayer);
lgraph = connectLayers(lgraph,'SepFeaLayer/HE','flatten_HE');
lgraph = connectLayers(lgraph,aLayerName,'flatten_DL');
lgraph = connectLayers(lgraph,'flatten_HE','depthConcatLayer/in1');
lgraph = connectLayers(lgraph,'flatten_DL','depthConcatLayer/in2');

symBlock = [
    fullyConnectedLayer(2,"Name","fc")  % Dense layer
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
    ];
bLayerName = symBlock(1).Name;
lgraph = addLayers(lgraph,symBlock);
lgraph = connectLayers(lgraph,'depthConcatLayer',bLayerName);

end
%%
function block = layerBlock(filterSize, filterNumOut, idx)
convLayer = convolution2dLayer(filterSize, filterNumOut, 'Name', ['conv_' num2str(idx)], 'Padding', 0);
activationLayer = @reluLayer;
block = [
    convLayer
    batchNormalizationLayer("Name",['bn_' num2str(idx)])
    activationLayer("Name",['activation_' num2str(idx)])
    maxPooling2dLayer([2 2],"Name",['pool_' num2str(idx)],"Padding","same","Stride",[2 2])
    ];

end

