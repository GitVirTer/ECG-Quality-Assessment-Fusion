%  Author:      R.G. Stockwell, Guoyang Liu
%  @File:       FeatureExtract_STrans_Ap_4D_GPU.m
%  @Software:   MATLAB 2020a/b
%  Input:       data is a 4-D tensor (1*Length*Channel*Sample)
%  Output:      NewData is also a 4-D tensor (Frequency*Time*Channel*Sample)
%  Note:        This function support GPU array
function NewData = FeatureExtract_STrans_Ap_4D_GPU(data)
%% Global Setting
Fs = 500;                               % Sampling Rate               
sig_downSamplingRate = 10;              % Downsampling Parameter
sample_freq = Fs/sig_downSamplingRate;  % Downsampling Rate
freqRange = [1 sample_freq/2];          % Frequency Range for S-transform
freqPrecision = 1;                      % Frequency Precision for S-transform
pValue = 0.3;                           % p Value for S-transform

%% Excute Procedure
data = data(:,1:sig_downSamplingRate:end,:,:);

freqSamplingRate = length(data)/sample_freq;
minFreq = freqRange(1)*freqSamplingRate;
maxFreq = freqRange(2)*freqSamplingRate;

if logical(round(freqSamplingRate/freqPrecision))
    freqInterval = round(freqSamplingRate/freqPrecision);
else
    freqInterval = 1;
end

tempNum = data(1,1,1,1);
nCh = size(data,3);
nSample = size(data,4);

synData = reshape(data, [size(data,1) size(data,2) nCh*nSample]);

data_st = ast_gpu_multiCh(synData,minFreq,maxFreq,1/sample_freq,freqInterval,pValue);
data_st_psd = (abs(data_st)).^2;

div = 1;
SegN = size(synData,2)/div;
st_psd_us = repmat(tempNum, [size(data_st,1) SegN nCh*nSample]);
for nSeg = 1:SegN
    st_psd_us(:, nSeg, :) = sum(data_st_psd(:,(nSeg-1)*div+1 : nSeg*div,:),2);
end

NewData = repmat(tempNum, [size(data_st,1) SegN nCh nSample]);
for iCh = 1:nCh
    NewData(:,:,iCh,:) = st_psd_us(:,:,iCh:nCh:end);
end

function [st,t,f] = ast_gpu_multiCh(timeseries,minfreq,maxfreq,samplingrate,freqsamplingrate,p)

% This is the S transform wrapper that holds default values for the function.
TRUE = 1;
FALSE = 0;
%%% DEFAULT PARAMETERS  [change these for your particular application]
factor = 1;
%%% END of DEFAULT PARAMETERS

% Change to column vector
timeseries_new = reshape(timeseries, [size(timeseries,2),size(timeseries,3)]);

% If you want to "hardwire" minfreq & maxfreq & samplingrate & freqsamplingrate do it here

% calculate the sampled time and frequency values from the two sampling rates
t = (0:size(timeseries_new,1)-1)*samplingrate;
spe_nelements =ceil((maxfreq - minfreq+1)/freqsamplingrate);%频段个数
f = (minfreq + [0:spe_nelements-1]*freqsamplingrate)/(samplingrate*size(timeseries_new,1));


% The actual S Transform function is here:
st = strans(timeseries_new,minfreq,maxfreq,samplingrate,freqsamplingrate,factor,p);
% this function is below, thus nicely encapsulated

%WRITE switch statement on nargout
% if 0 then plot amplitude spectrum



return

function st = strans(timeseries,minfreq,maxfreq,samplingrate,freqsamplingrate,factor,p)
% Returns the Stockwell Transform, STOutput, of the time-series
% Code by R.G. Stockwell.
% Reference is "Localization of the Complex Spectrum: The S Transform"
% from IEEE Transactions on Signal Processing, vol. 44., number 4,
% April 1996, pages 998-1001.
%
%-------Inputs Returned------------------------------------------------
%         - are all taken care of in the wrapper function above
%
%-------Outputs Returned------------------------------------------------
%
%	ST    -a complex matrix containing the Stockwell transform.
%			 The rows of STOutput are the frequencies and the
%			 columns are the time values
%
%
%-----------------------------------------------------------------------

% Compute the length of the data.
n = size(timeseries,1);
original = timeseries;
tempNum = timeseries(1,1,1,1);
nCh = size(timeseries,2);
% If vector is real, do the analytic signal

% Compute FFT's
vector_fft=fft(timeseries);
vector_fft=cat(1,vector_fft,vector_fft);


% Preallocate the STOutput matrix
st = repmat(tempNum, [ceil((maxfreq - minfreq+1)/freqsamplingrate), n, size(timeseries, 2)])+1i;
% st=complex(zeros(ceil((maxfreq - minfreq+1)/freqsamplingrate),n),zeros(ceil((maxfreq - minfreq+1)/freqsamplingrate),n));
% Compute the mean
% Compute S-transform value for 1 ... ceil(n/2+1)-1 frequency points
if minfreq == 0
    st(1,:,:) = mean(timeseries)*(1&[1:1:n]);
else
    st(1,:,:) = ifft(vector_fft(minfreq+1:minfreq+n, :).*g_window(n,minfreq,factor,p,tempNum,nCh));
end

%the actual calculation of the ST
% Start loop to increment the frequency point
for banana=freqsamplingrate:freqsamplingrate:(maxfreq-minfreq)
    st(round(banana/freqsamplingrate)+1,:,:)=ifft(vector_fft(round(minfreq+banana)+1:round(minfreq+banana)+n, :).*g_window(n,round(minfreq+banana),factor,p,tempNum,nCh));
end   % a fruit loop!   aaaaa ha ha ha ha ha ha ha ha ha ha
% End loop to increment the frequency point

%%% end strans function

%------------------------------------------------------------------------
function gauss=g_window(length,freq,factor,p,tempNum,nCh)

% Function to compute the Gaussion window for
% function Stransform. g_window is used by function
% Stransform. Programmed by Eric Tittley
%
%-----Inputs Needed--------------------------
%
%	length-the length of the Gaussian window
%
%	freq-the frequency at which to evaluate
%		  the window.
%	factor- the window-width factor
%
%-----Outputs Returned--------------------------
%
%	gauss-The Gaussian window
%
vector = repmat(tempNum, [2,length]);
vector(1,:)=0:length-1;
vector(2,:)=-length:-1;
vector=vector.^2;
vector=vector*(-factor*2*pi^2/(freq/p)^2);
% Compute the Gaussion window
gauss = repmat(sum(exp(vector))', [1 nCh]);

%-----------------------------------------------------------------------
