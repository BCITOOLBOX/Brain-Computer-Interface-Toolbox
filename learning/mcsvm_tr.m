function [svmObject pp]=mcsvm_tr(datafiles,predt,postdt,chidx,ftid,target,ft,ftmrk,act_flgtest)
%[svmobject pp]=mcsvm_tr(datafiles,predt,postdt,chidx,ftid,target,ft,ftmrk,flgtest)
% Train multiclass SVM classifier for EEG BCI classification.
% 'datafiles' is cell array of o-data files containing the EEG data,
% 'predt' and 'postdt' are before-stimulus and after-stimulus epoch
% lengths, in seconds. 'chidx' is the array of indexes of the eeg-data
% channels to use. 'ftid' is feature pre-selection specifier. 'target' 
% is the set of targets to be included in the classification.
%
% Feature pre-selection can be specified in one of three ways: 
% - if 'ftid' is empty, default slow-ERP will be used by selecting <=5Hz
%   real/imaginary Fourier amplitudes from ftprep features vector; 
% - 'ftid' can be an array of size 1 x n_feature specifying the features 
%    in the combined features vector from ftprep, such as produced by 
%    make_features; see help make_features for more information; 
% - 'ftid' can be a string of the form [fs][MET][cs][NUM].
%    [fs] is a single character feature-type selector, such as passed to
%    make_features, and can be one of the following
%    'tXXX' to use time-series features,
%    'sXXX' to use FT amplitude features (re/im), 
%    'aXXX' to use FT amplitude features (abs/angle), 
%    'pXXX' to use PSD features in quadratic form, 
%    'dXXX' to use PSD features in log (dB) form,
%    'eXXX' to use EEG band power features.
%    [MET] is an feature ranking method and can be one of the following
%    'xMUIxxx' to use Mutual Information-based feature ranking,
%    'xFRQxxx' to use low-pass frequency filtering,
%    'xKLDxxx' to use Kullback-Leibler divergence-based feature ranking
%     (not supported for LDA)
%    'xCORxxx' to use Pearson correlation-based feature ranking (not
%     supported  for LDA)
%    [cs] is a single character specifying the type of method to use for
%    selecting top-ranked features, can be 
%    'XXXzNUM' for z-score based threshold cutoff,
%    'XXXnNUM' to keep a fixed number of top ranked features.
%    [NUM] is the numerical threshold (if [cs]=='z') or the count of
%    features (if [cs]=='n') to be used
%
% mcsvm_tr returns a svmobject for trained multiclass SVM, and pp - an 
% array of train-validation-test performance values observed.
%
% Pass 'ft' and 'ftmrk' outputs of ftprep to prevent svm_tr recomputing
% feature vector and train and test the classifier using given ft and 
% ftmrk values. Set ft.tridx field to use only some of the examples in the
% training, ft.tridx should be an index array (int or bool) selecting
% examples from ft. Set 'flgtest' to use a fixed subset of samples as the 
% 'test' set (over already selected test and validation sets); this is
% used for controlling external regularization-caused overfitting.
% 'flgtest' should be an index array (int or bool) selecting examples 
% from ft and defines the examples to be used as the test set.
%
% Example usage:
%  [svmobj pp]=mcsvm_tr({'nkdeney-example.mat'},0,0.85,1:21,'smuiz3',1:3);
%
% See more examples in ftr_in1ch.m and ftr_out1ch.m.
%
% Y.Mishchenko (c) 2015

%ranking-cutoff runs
%smui 40(2.5std)~0.63;60(2.25std*)~0.65;80(2.1std)~0.64;100(1.9std)~0.59
%skld 40(2.5std)~0.63;60(2.25std*)~0.65;80(1.9std)~0.58;100(1.75std)~0.55
%scor 40(2.6std)~0.72; 60(2.4std)~0.68;80(2.25std)~0.68;100(2.1std)~0.68
%smui 30~0.69; skld 30~0.63; scor 30~0.72;
%slow-ERP(168 features)~0.75;

%Undocumented feature `global xvalsequential` is used to specify whether
%training/validation split should be randomized (false) or sequntial (true)

%% Parameters
xvalthr=0.70;     %train-validation split
testthr=0.1;      %train-validation--test split
nnmax=10000;      %max number of examples to draw for training
global xvalsequential 	%sequential/random train-validation split modifier
global commonmode       %common mode modifier
if isempty(xvalsequential) xvalsequential=false; end

if nargin<5 ftid=[]; end
if nargin<6 || isempty(target) target=[1 2]; end


%% Prepare features
fprintf('Preparing features...\n');

%use precomputed features (if ft or ftmrk were given) or compute features
if nargin<8 || isempty(ft) || isempty(ftmrk)
  [ft ftmrk]=ftprep(datafiles,predt,postdt,chidx,commonmode,ftid);
end

%make features
[eegsamples,ftmrk]=make_features(ft,ftmrk,ftid);

%if trial-idx are passed, constrain samples to specified trials (ft.tridx)
if isfield(ft,'tridx')
  eegsamples=eegsamples(ft.tridx,:);
  ftmrk=ftmrk(ft.tridx);
end

%% Finalize data
%restruct examples to only such within 'target'
target=sort(target);
ttidx=ismember(ftmrk,target);
eegsamples=eegsamples(ttidx,:);
mrktargets=ftmrk(ttidx);
nn=size(eegsamples,1);

fprintf('#########################\n');
fprintf('Total samples %i\n',size(eegsamples,1));
fprintf('Total features %i\n',size(eegsamples,2));
fprintf('#########################\n');

%training-validation/test-sets split
if nargin<9 || isempty(act_flgtest)
  act_flgtest=rand(1,nn)<testthr;
end

%% Prepare features
nt=length(target);
osvm=cell(nt,nt);

%training/validation-sets split
if xvalsequential
  fprintf('Sequential x-validation split\n');
  act_flgtrain=((1:nn)/nn)<xvalthr;
else
  fprintf('Random x-validation split\n');
  act_flgtrain=rand(1,nn)<xvalthr;
end

%form example sets
%training exampels set
idx=find(act_flgtrain & ~act_flgtest);
idx=idx(randperm(length(idx)));
idx=idx(1:min(length(idx),nnmax));
train_examples=eegsamples(idx,:);
train_targets=mrktargets(idx);

%validation examples set
idx=find(~act_flgtrain & ~act_flgtest);
idx=idx(randperm(length(idx)));
val_examples=eegsamples(idx,:);
val_targets=mrktargets(idx);

%test examples set
idx=find(act_flgtest);
idx=idx(randperm(length(idx)));
test_examples=eegsamples(idx,:);
test_targets=mrktargets(idx);

%% train SVM's
tic
fprintf('Train %ix%i SVM models...\n',nt,nt);
fprintf('Prepare %ix%i feature sets...\n',nt,nt);
for i=2:nt
  for j=1:i-1
    fprintf(' ranking features for pair (%i,%i)...\n',i,j);
    
    %training examples, only contain {i-j} pair
    idx=ismember(train_targets,target([i,j]));
    xexamples=train_examples(idx,:);
    xtargets=train_targets(idx);
    xtargets=(xtargets==target(i));
    
    [ftidx,ranks,nid]=parse_ftid(ftid,ft,xexamples,xtargets,[0,1]);

    %train SVM    
    fprintf(' training svm for pair (%i,%i)...\n',i,j);        
    options=optimset('MaxIter',10000);
    if nid>0
      %come back with given cutoff - just train SVM 
      warning off all
      svm2=svmtrain(xexamples(:,ftidx),xtargets,'Method','LS',...
        'QuadProg_Opts',options);
      warning on all
    else
      %come back with given cutoff - just train SVM
      dnn=25;     %initial num of features and the number increment step
      dnn_stop=4; %number of feature number increments without improvement 
                  %before stop
      dnn_cnt=0;  %counter of feature number increments without improvement
      dnn_mininc=2E-2; %minimal required increment
      xsvm2=[];   %best SVM
      xnid=[];    %best nid
      xp=-[Inf,Inf,Inf];   %previous p values
      for nid=dnn:dnn:length(ranks)
        ftidx_=ftidx(1:nid);
        warning off all
        svm2=svmtrain(xexamples(:,ftidx_),xtargets,'Method','LS',...
          'QuadProg_Opts',options);
        warning on all
        
        %check performance on training set
        xtest=svmclassify(svm2,xexamples(:,ftidx_));
        p1=mean(xtargets==xtest);
        
        %check performance on validation set
        idx=ismember(val_targets,target([i,j]));
        xxexamples=val_examples(idx,:);
        xxtargets=val_targets(idx);
        xxtargets=(xxtargets==target(i));
        
        xtest=svmclassify(svm2,xxexamples(:,ftidx_));
        p2=mean(xxtargets==xtest);
        

        %check performance on test set
        idx=ismember(test_targets,target([i,j]));
        xxexamples=test_examples(idx,:);
        xxtargets=test_targets(idx);
        xxtargets=(xxtargets==target(i));
        
        xtest=svmclassify(svm2,xxexamples(:,ftidx_));
        p3=mean(xxtargets==xtest);
        
        %evaluate stopping condition - no improvement for dnn_stop
        %increments
        if (xp(2)>p2+dnn_mininc || xp(3)>p3+dnn_mininc)
            dnn_cnt=dnn_cnt+1;            
        else
          xsvm2=svm2;
          xnid=nid;
          xp=[p1,p2,p3];     
          dnn_cnt=0;
        end
        
        fprintf(' feature number search %i: (%g,%g,%g)...\n',nid,p1,p2,p3);
            
        if dnn_cnt>dnn_stop
          break;
        end
      end
      
      %set best SVM out
      svm2=xsvm2;
      nid=xnid;
      ftidx=ftidx(1:xnid);      
      
      fprintf(' ===selected %i: (%g,%g,%g)...\n',xnid,xp);
    end
    
    %read SVM model w2*x'-b2
    b2=svm2.Bias;
    w2=svm2.SupportVectors'*svm2.Alpha;
    w2=w2.*svm2.ScaleData.scaleFactor';
    b2=b2+svm2.ScaleData.shift*w2;
    w2=-w2;
    
    
    o=[];
    o.ftidx=ftidx;      
    o.w2=w2;
    o.b2=b2;
    o.bbef2=0;
    
    osvm{i,j}=o;
  end
end
toc


%check overall performance on training set
xtest=ownclassify(osvm,train_examples);
p1=mean(train_targets==xtest);
fprintf('Training %g\n',p1);

%% X-validation
%obtain mnSVM values
vals1=ownclassify(osvm,val_examples);
vals=val_targets;

%check performance on validation set
p2=sum(vals==vals1)/length(vals);
fprintf('X-validation %g\n',p2);


%% Test
%obtain SVM values
vals1=ownclassify(osvm,test_examples);
vals=test_targets;

%check performance on test set
p3=sum(vals==vals1)/length(vals);
fprintf('Test %g\n',p3);

%% Form SVM object
svmObject=[];
svmObject.target=target;
svmObject.o=osvm;
svmObject.predt=predt;
svmObject.postdt=postdt;
svmObject.ftid=ftid;

%output train-validation-test errors
pp=[p1 p2 p3];


  %classify using SVM model object
  function labels=ownclassify(osvm,xeegsamples)
    nn=size(osvm,1);    %number of 1-1 classifiers
    nt=size(xeegsamples,1);
    votes=zeros(nt,nn);  %1-1 classifiers' votes
    
    for ii=2:nn
      for jj=1:ii-1
        w2=osvm{ii,jj}.w2;
        b2=osvm{ii,jj}.b2;
        bbef2=osvm{ii,jj}.bbef2;
        ftidx_=osvm{ii,jj}.ftidx;
        
        %obtain SVM values
        valsd=-b2+xeegsamples(:,ftidx_)*w2;
        vals1=(sign(valsd-bbef2)+1)/2;
        
        votes(vals1>0,ii)=votes(vals1>0,ii)+1;
        votes(vals1==0,jj)=votes(vals1==00,jj)+1;
      end
    end
    
    [g labels]=max(votes,[],2);
    labels=reshape(target(labels),[],1);
  end

end
