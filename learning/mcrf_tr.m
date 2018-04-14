function [ccobj, pp]=mcrf_tr(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid)
%[rfobject pp]=mcrf_tr(dataFiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid)
% Train a RF classifier for multiclass EEG BCI classification.
%  dataFiles - cell array of data files containing EEG data in o-format
%  predt and postdt - the detection window offsets, in seconds. Use 
%  positive values of predt to indicate the beginning of the detection 
%  window predt sec ahead of (before) the trial-onset signal, and use 
%  negative values to indicate the beginning of the detection window 
%  -predt sec behind of (after) the trial-onset signal. postdt is always 
%  positive and measures the ending of the detection window relative to 
%  the trial-onset signal
%  chid - array specifying the indexes of the eeg-channels to include in 
%  EEG BCI classification 
%  ftid - feature pre-selection specifier, see below
%  targets - set of all targets to be included in EEG BCI classification
%
% Feature pre-selection specifier can be given in one of three ways: 
% - if ftid is empty, default slow-ERP is used by selecting <=5Hz
%    real/imaginary Fourier amplitudes from the ftprep-feature vector; 
% - if ftid is an array of size {1 x n_feature}, the features specified
%    by ftid in the combined feature-vector returned by make_features are 
%    selected, as if ftid was logical or integer index array, see 
%    'help make_features' for more information; 
% - if ftid is a string of the form "[fs][MET][cs][NUM]", features are
%    selected according to the rules below;
%    [fs] is a single character indicating the type of features to select 
%    and can be one of the following:
%     't' to select time-series features,
%     's' to select FT amplitude features (re/im), 
%     'a' to select FT amplitude features (abs/angle), 
%     'p' to select PSD features in quadratic form, 
%     'd' to select log-PSD features (dB form),
%     'e' to select EEG band power features,
%     'h' to select log-EEG band power features (dB form).
%     'x' to select all types of features.
%    [MET] is a 3-character string indicating the feature ranking criterion 
%    to use and can be one of the following:
%     'FRQ' simple low-pass or band-pass frequency filtering,
%     'MUI' target-feature Mutual Information-based criterion,
%     'KLD' Kullback-Leibler divergence-based criterion (can be used for
%           two-target classification only)
%     'COR' Pearson correlation-based criterion (can be used for two-
%           target classification only)
%    [cs] is a single character specifying the type of threshold to use 
%    on the ranking criterion and can be one of the following
%     'z' to use z-score based cutoff,
%     'n' to use a fixed number of top ranked features.
%    [NUM] specifies the numerical value to use for threshold (for 'z') or 
%    the number of features to keep (for 'n'). For FRQ-type criterion NUM
%    can be either a single number (understood as low-pass filter
%    cutoff) or a pair of frequencies separated by dash (understood as
%    band-pass filter cutoff).
%    If [NUM] is not given, automatic selection of NUM using
%    cross-validation performance is performed inside the function.
%
%   Some examples of ftid:
%    'sFRQz5' - real/imaginary Fourier amplitudes low-pass filtered at 5Hz
%    'pFRQz10-15' - PSD features band-pass filtered to 10-15Hz
%    'xMUIz3' - all features ranked by MUI criterion and thresholded at
%             z=3 above the mean
%    'xMUI' - all features ranked by MUI criterion, with the number of
%             features kept automatically selected by cross-validation
%    'eCORn20' - EEG-band features ranked by individual correlation with
%             target (0 or 1 - two-target classification only), with 20 top
%             features kept.
%
% Pass optional 'ft' and 'ftmrk' pre-computed using 'ftprep' to prevent 
% the function from re-computing the features and instead use provided ft 
% and ftmrk for training the classifier (useful if calling the function
% within an external routine such as learning-curve builder). Set ft.tridx 
% field (trial-idx) of the structure 'ft', in order to use only some of 
% the trials' examples for the classifier training, specified as a logical 
% or integer index array over the list of trials in ft.
% 
% Pass optional 'testid' to use a fixed subset of trials as the 'test' set. 
% This is useful for controlling regularization-caused overfitting by an 
% hyperparameter selection algorithm. 'testid' should be a logical index 
% array over the list of trials in ft.
%
% Example usage:
%  [rfobj pp]=mcrf_tr({'nkdeney-example.mat'},0,0.85,1:21,'sfrqz5',1:3);
%
% This function uses TreeBagger Matlab's function and requires Matlab 8.5 
% or higher.
%
% Y.Mishchenko (c) 2016

%Undocumented feature, `global xvalsequential` is used to specify whether
%training/validation split should be randomized (false) or sequntial (true)

%% Validating input parameters
if nargin<5 ftid=[]; end
if nargin<6 || isempty(targets) targets=[1 2]; end
if nargin<8
  ft=[];
  ftmrk=[];
end
if nargin<9
  testid=[];
end

%% Calling training scaffold
[ccobj, pp]=gen_tr(@owntrain,@ownclassify,datafiles,predt,postdt,...
  chid,ftid,targets,ft,ftmrk,testid);

  %Train classifier looping over the hyperparameter values
  %using cross-validation to select the best hyperparameter value
  function clobj=owntrain(train_examples,train_targets,val_examples,val_targets)
    fprintf('Building category classifier...\n');
    %loop over hyperparameter here
    %range of values to check for hyperparameter
    %hyper=[50,100,150,200,300,500];
    %WE WILL NOT SCAN OVER HYPERPARAMETERS BUT USE OFTEN 
    %QUOTED SIZE OF THE FOREST OF 100
    hyper=[100];
    val_performance=zeros(size(hyper));
    val_classifier=cell(size(hyper));
    
    %fprintf commented out due to removal of hyperparameter scanning
    %fprintf(' Selecting hyperparameter...\n');
    cnt=1;
    for k=hyper
      clobj=TreeBagger(k,train_examples,train_targets);
      labels=predict(clobj,val_examples);
      
      %need to do this additionally for TreeBagger
      labels=cellfun(@str2num,labels);
      
      val_classifier{cnt}=clobj;
      val_performance(cnt)=mean(labels==val_targets);      
      cnt=cnt+1;
      
      %fprintf(' %g,',val_performance(cnt-1)); 
    end
    %fprintf('\b\n');
    
    %select best hyperparameter
    [~, best_xval_index]=max(val_performance);    
    
    %fprintf(' Selected hyperparameter %g...\n',hyper(best_xval_index));
    clobj=val_classifier{best_xval_index};
  end


  %classify using classifier model object
  function labels=ownclassify(clobj,examples)
    labels=predict(clobj,examples);
    
    %need to do this additionally for TreeBagger
    labels=cellfun(@str2num,labels);    
  end

end
