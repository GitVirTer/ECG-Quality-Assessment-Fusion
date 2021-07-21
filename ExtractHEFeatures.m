%  Author:      Guoyang Liu
%  @Software:   MATLAB 2020a/b
%  Please cite: G. LIU, X. HAN, L. TIAN, W. ZHOU, and H. LIU, "ECG Quality Assessment Based on Hand-Crafted Statistics and Deep-Learned S-Transform Spectrogram Features," Computer Methods and Programs in Biomedicine, p. 106269, 2021.

function HEFeatures = ExtractHEFeatures(data)
nCh = size(data,2);
% lenData = size(data,1);
% Hd = biasFilter_8order;
concatFeature = zeros(3, nCh, 'like', data);
for iCh = 1:nCh
    chData = data(:,iCh);
    
    % Feature 1
    curData = chData';
    tmpData = tabulate(curData);
    uniqueData = tmpData(:,1);
    uniqueData = uniqueData(tmpData(:,2)>1);
%     uniqueData = unique(curData);
    cntOut = [];
    for iValue = 1:numel(uniqueData)
        logicalData = curData == uniqueData(iValue);
        out = find(diff([logicalData false])==-1)-find(diff([false logicalData])==1)+1;
        cntOut = cat(2, cntOut, out);
    end
    f1 = max(cntOut);
    
    % Feature 2
    a = [1;-7.838967981032241;26.885713620195883;-52.695281240277190;64.554605916118860;-50.616003676692560;24.805811247040097;-6.947134780895171;0.851256895543203];
    b = [3.421961416593648e-15;2.737569133274919e-14;9.581491966462216e-14;1.916298393292443e-13;2.395372991615554e-13;1.916298393292443e-13;9.581491966462216e-14;2.737569133274919e-14;3.421961416593648e-15];
    tmpData = filter(b, a, chData);
    f2 = max(tmpData);
    
    % Feature 3
    f3 = sum(abs(chData)==max(chData));
    
    % Concatenate Features
    concatFeature(:,iCh) = [f1 f2 f3];
    
end
HEFeatures = concatFeature(:);

end
