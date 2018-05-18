classdef ClassificationSupportVectorMachine <handle & SuperClass
    
    properties
        eegsamples
    end
    
    methods(Access=public)
        
        function object = ClassificationSupportVectorMachine(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid)
            
            if nargin<5 || isempty(ftid)
                ftid=[];
            end
            
            if nargin<6 || isempty(targets)
                targets=[1 2];
            end
            
            if nargin<7 || isempty(ft)
                ft=[];
            end
            
            if nargin<8 ||isempty(ftmrk)
                ftmrk=[];
            end
            
            if nargin<9 ||isempty(testid)
                testid=[];
            end
            
            object@SuperClass(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid);
            
        end
        
        
        
        function [ccobj, pp]=callTrainingScaffold(this)
            
            [ccobj, pp]=GeneralTraining(@owntrain,@ownclassify,this.datafiles,this.predt,this.postdt,...
                this.chid,this.ftid,this.targets,this.ft,this.ftmrk,this.testid);
            
            
            
            function [ccobj, pp]=GeneralTraining(funcTrain,funcClassify,datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid)
                
                %% Parameters
                xvalThr=0.70;     %train-validation split
                testThr=0.1;      %{train-validation}-test split
                nnmax=10000;      %max number of examples to use in training
                
                global xvalsequential 	%sequential/random train-validation split modifier
                global commonmode       %subtraction mode modifier
                
                if isempty(xvalsequential)
                    xvalsequential=false;
                end
                if nargin<5
                    ftid=[];
                end
                if nargin<8 || isempty(targets)
                    targets=[1 2];
                end
                
                %% Prepare features
                fprintf('Preparing features...\n');
                
                %use precomputed features (if ft or ftmrk were given) or compute features
                if nargin<8 || isempty(ft) || isempty(ftmrk)
                    [ft,ftmrk]=ftprep(datafiles,predt,postdt,chid,commonmode,ftid);
                end
                
                %make features
                [trialExamples,trialLabels]=make_features(ft,ftmrk,ftid);
                
                %if trial-idx are passed, constrain samples to specified trials (ft.tridx)
                if isfield(ft,'tridx')
                    trialExamples=trialExamples(ft.tridx,:);
                    trialLabels=trialLabels(ft.tridx);
                end
                
                %% Finalize data
                %restruct examples to only such within 'target'
                targets=sort(targets);
                ttidx=ismember(trialLabels,targets);
                trialExamples=trialExamples(ttidx,:);
                mrktargets=trialLabels(ttidx);
                nn=size(trialExamples,1);
                
                fprintf('#########################\n');
                fprintf('Total samples %i\n',size(trialExamples,1));
                fprintf('Total features %i\n',size(trialExamples,2));
                fprintf('#########################\n');
                
                %training-validation/test-sets split
                if nargin<9 || isempty(testid)
                    testid=rand(1,nn)<testThr;
                end
                
                %% Prepare features
                nt=length(targets);
                osvm=cell(nt,nt);
                
                %training/validation-sets split
                if xvalsequential
                    fprintf('Sequential x-validation split\n');
                    act_flgtrain=((1:nn)/nn)<xvalThr;
                else
                    fprintf('Random x-validation split\n');
                    act_flgtrain=rand(1,nn)<xvalThr;
                end
                
                %form example sets
                %training exampels set
                idx=find(act_flgtrain & ~testid);
                idx=idx(randperm(length(idx)));
                idx=idx(1:min(length(idx),nnmax));
                train_examples=trialExamples(idx,:);
                train_targets=mrktargets(idx);
                
                %validation examples set
                idx=find(~act_flgtrain & ~testid);
                idx=idx(randperm(length(idx)));
                val_examples=trialExamples(idx,:);
                val_targets=mrktargets(idx);
                
                %test examples set
                idx=find(testid);
                idx=idx(randperm(length(idx)));
                test_examples=trialExamples(idx,:);
                test_targets=mrktargets(idx);
                
                %% train SVM's
                tic
                fprintf('Train %ix%i SVM models...\n',nt,nt);
                fprintf('Prepare %ix%i feature sets...\n',nt,nt);
                for i=2:nt
                    for j=1:i-1
                        fprintf(' ranking features for pair (%i,%i)...\n',i,j);
                        
                        %training examples, only contain {i-j} pair
                        idx=ismember(train_targets,targets([i,j]));
                        xexamples=train_examples(idx,:);
                        xtargets=train_targets(idx);
                        xtargets=(xtargets==targets(i));
                        
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
                                svm2=svmtrain(xexamples(:,ftidx_),xtargets,'Method','LS','QuadProg_Opts',options);
                                warning on all
                                
                                %check performance on training set
                                xtest=svmclassify(svm2,xexamples(:,ftidx_));
                                p1=mean(xtargets==xtest);
                                
                                %check performance on validation set
                                idx=ismember(val_targets,targets([i,j]));
                                xxexamples=val_examples(idx,:);
                                xxtargets=val_targets(idx);
                                xxtargets=(xxtargets==targets(i));
                                
                                xtest=svmclassify(svm2,xxexamples(:,ftidx_));
                                p2=mean(xxtargets==xtest);
                                
                                
                                %check performance on test set
                                idx=ismember(test_targets,targets([i,j]));
                                xxexamples=test_examples(idx,:);
                                xxtargets=test_targets(idx);
                                xxtargets=(xxtargets==targets(i));
                                
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
                ccobj=[];
                ccobj.targets=targets;
                ccobj.o=osvm;
                ccobj.predt=predt;
                ccobj.postdt=postdt;
                ccobj.ftid=ftid;
                
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
                    labels=reshape(targets(labels),[],1);
                end
                
                
                function [ftdata,mrkdata,ftidx]=make_features(ft,ftmrk,ftid,verb)
                    %[ftdata mrkdata ftidx]=make_features(ft,ftmrk[,ftid,verb])
                    % Companion function for ftprep, convert the output of ftprep to
                    % n_trials x n_feature matrix of features ('ftdata') and corresponding
                    % n_trials x 1 vector of labels ('mrkdata').
                    %
                    % 'ftid' specifies which type of features is to be produced from ftprep.
                    % 'ftid' can be given as a string such as
                    % - "all" - returns all features from ftprep,
                    % - 'xXXX' - returns all features from ftprep (with subsequent
                    %            feature pre-selection such as MUI,KLD or COR [no FRQ!])
                    % - 'tXXX' - returns time-series features,
                    % - 'sXXX' - returns FT amplitude features as re/im,
                    % - 'aXXX' - returns FT amplitude features as abs/angle,
                    % - 'pXXX' - returns PSD features as amplitude-square,
                    % - 'dXXX' - returns PSD features as log10 (dB),
                    % - 'eXXX' - returns EEG band power features as amplitude square,
                    % - 'hXXX' - returns EEG band power features as log10 (dB),
                    %
                    % Skip or specify 'ftid' as [] for default selector, which is
                    % the slow (freq<=5Hz) FT amplitudes in re/im form.
                    %
                    % Specify 'ftid' as logical 1 x n_feature vector to directly select the
                    % features from the array of all ftprep-features. The order of the features
                    % in the array of all ftprep-features is [ft.tseries,ft.eegpow,
                    % ft.(db)eegpow,ft.pow,ft.(db)pow,ft.real,ft.imag,ft.ampl,ft.angle].
                    %
                    % 'verb' specifies the level of verbocity.
                    %
                    % Upon completion returns the corresponding features, labels, and the
                    % logical 1 x n_features feature selector-vector in 'ftdata','mrkdata' and
                    % 'ftidx', respectively.
                    %
                    % Example usage:
                    %  [ftdata, mrkdata, ftidx]=ftget(ft,ftmrk,[])
                    %
                    %Y. Mishchenko (c) 2016
                    
                    if nargin<4 || isempty(verb)
                        verb=0; end
                    
                    if isempty(ftid)
                        ftid='sFRQz5.01';
                    end
                    
                    nults=false(size(ft.tsampleid));
                    onets=true(size(ft.tsampleid));
                    nulft=false(size(ft.freqid));
                    oneft=true(size(ft.freqid));
                    nulpsd=false(size(ft.freqid));
                    onepsd=true(size(ft.freqid));
                    nuleeg=false(size(ft.eegfreqid));
                    oneeeg=true(size(ft.eegfreqid));
                    
                    if verb>0
                        fprintf('ftprep-feature vector''s structure:\n{\n');
                        fprintf(' - Time-series features %i\n',length(nults));
                        fprintf(' - EEG band-power features %i\n',length(nuleeg));
                        fprintf(' - EEG band-power (dB) features %i\n',length(nuleeg));
                        fprintf(' - PSD features %i\n',length(nulpsd));
                        fprintf(' - PSD (dB) features %i\n',length(nulpsd));
                        fprintf(' - FT amplitude-real features %i\n',length(nulft));
                        fprintf(' - FT amplitude-imaginary features %i\n',length(nulft));
                        fprintf(' - FT amplitude-abs features %i\n',length(nulft));
                        fprintf(' - FT amplitude-angle features %i\n}\n',length(nulft));
                    end
                    
                    e=1E-6;
                    if nargin<3 || isempty(ftid)
                        %default selector, FT re/im + freqid<=5 (Hz)
                        ftdata=[real(ft.ft(:,ft.freqid<=5)),imag(ft.ft(:,ft.freqid<=5))];
                        ftidx=[nults,nuleeg,nuleeg,nulpsd,nulpsd,ft.freqid<=5,ft.freqid<=5,nulft,nulft];
                    elseif ischar(ftid)
                        if(strcmpi(ftid,'all') || ftid(1)=='x')
                            ftdata=cat(2,ft.tseries,ft.eegpow,log10(ft.eegpow+e),...
                                ft.pow,log10(ft.pow+e),real(ft.ft),imag(ft.ft),abs(ft.ft),angle(ft.ft));
                            ftidx=true(1,size(ftdata,2));
                        elseif(ftid(1)=='t')
                            ftdata=ft.tseries;
                            ftidx=[onets,nuleeg,nuleeg,nulpsd,nulpsd,nulft,nulft,nulft,nulft];
                        elseif(ftid(1)=='s')
                            ftdata=[real(ft.ft),imag(ft.ft)];
                            ftidx=[nults,nuleeg,nuleeg,nulpsd,nulpsd,oneft,oneft,nulft,nulft];
                        elseif(ftid(1)=='a')
                            ftdata=[abs(ft.ft),angle(ft.ft)];
                            ftidx=[nults,nuleeg,nuleeg,nulpsd,nulpsd,nulft,nulft,oneft,oneft];
                        elseif(ftid(1)=='e')
                            ftdata=ft.eegpow;
                            ftidx=[nults,oneeeg,nuleeg,nulpsd,nulpsd,nulft,nulft,nulft,nulft];
                        elseif(ftid(1)=='h')
                            ftdata=log10(ft.eegpow+e);
                            ftidx=[nults,nuleeg,oneeeg,nulpsd,nulpsd,nulft,nulft,nulft,nulft];
                        elseif(ftid(1)=='p')
                            ftdata=ft.pow;
                            ftidx=[nults,nuleeg,nuleeg,onepsd,nulpsd,nulft,nulft,nulft,nulft];
                        elseif(ftid(1)=='d')
                            ftdata=log10(ft.pow+e);
                            ftidx=[nults,nuleeg,nuleeg,nulpsd,onepsd,nulft,nulft,nulft,nulft];
                        else
                            fprintf(' Warning (make_features): unrecognized ftid string;\n');
                            fprintf(' defaulting to freqid<=5Hz\n');
                            ftdata=[real(ft.ft(:,ft.freqid<=5)),imag(ft.ft(:,ft.freqid<=5))];
                            ftidx=[nults,nuleeg,nuleeg,nulpsd,nulpsd,ft.freqid<=5,ft.freqid<=5,nulft,nulft];
                        end
                    else
                        ftdata=cat(2,ft.tseries,ft.eegpow,log10(ft.eegpow+e),...
                            ft.pow,log10(ft.pow+e),real(ft.ft),imag(ft.ft),abs(ft.ft),angle(ft.ft));
                        ftdata=ftdata(:,ftid);
                        ftidx=ftid;
                    end
                    
                    fprintf('Total %i values\n',sum(ftidx));
                    
                    %prepare labels array
                    if nargin>1 && nargout>1
                        mrkdata=ftmrk;
                    end
                    
                end
                
                function [ftidx,ranks,nid]=parse_ftid(ftid,ft,xeegsamples,xmrktargets,target)
                    %[ftidx,ranks,nid]=parse_ftid(ftid,ft,xeegsamples,xmrktargets,target)
                    %Utility function performing parsing the feature pre-selection identifier
                    %'ftid' and performing necessary feature ranking and selection based on
                    %'ftid' for training-learning functions in this section.
                    %
                    % Feature pre-selector can be specified in one of following ways:
                    % - if 'ftid' is empty, default slow-ERP selector is implied (handled by
                    %   make-features) and nothing is done;
                    % - if 'ftid' is an array of size 1 x n_feature, direct feature selection
                    %    is implied (handled by make_features) and nothing is done; In these
                    %    cases parse_ftid returns 'rankids' and 'nids' corresponding to the
                    %    full received featureset;
                    % - if 'ftid' is a string of the form "[fs][MET][cs][NUM]";
                    %    [fs] is a single character identifying type of features to be used,
                    %    (also used by make_features) and can be one of the following
                    %    'tXXX' to use time-series features,
                    %    'sXXX' to use FT amplitude features (re/im),
                    %    'aXXX' to use FT amplitude features (abs/angle),
                    %    'pXXX' to use PSD features in quadratic form,
                    %    'dXXX' to use PSD features in log (dB) form,
                    %    'eXXX' to use EEG band power features.
                    %    [MET] is a alphabetic specifier identifying the feature ranking to be
                    %    used for ranking of features, can be one of the following
                    %    'xMUIxxx' to use Mutual Information-based ranking of features,
                    %    'xFRQxxx' to use low-pass frequency selector,
                    %    'xFRQxxx-xxx' to use band-pass frequency selector,
                    %    'xKLDxxx' to use Kullback-Leibler divergence ranking of features
                    %     (not supported for #targets>2)
                    %    'xCORxxx' to use correlation-based ranking of features
                    %    [cs] is a single character specifying the type of method to use for
                    %    selecting the features by their rank, can be
                    %    'XXXzNUM' to select top features based on a z-score-type threshold, in
                    %     which case features with rank-scores NUM times STD above rank-score
                    %     average are selected;
                    %    'XXXnNUM' to select a fixed number of features starting from highest
                    %     rank-scores in descending order;
                    %    [NUM] is the numerical threshold (if [cs]=='z') or the number of
                    %    features (if [cs]=='n') to be selected.
                    %
                    % Y.Mishchenko (c) 2017
                    
                    
                    %enable default selector
                    if isempty(ftid)
                        ftid='sFRQz5.01';
                    end
                    
                    %adjust ftid if char
                    if ischar(ftid)
                        feature_type=ftid(1);
                        
                        if length(ftid)>2
                            selector_type=ftid(2:4);
                        else
                            selector_type='';
                        end
                        
                        if length(ftid)>4
                            cutoff_type=ftid(5);
                        else
                            cutoff_type='';
                        end
                        
                        if length(ftid)>5
                            cutoff_value=ftid(6:end);
                        else
                            cutoff_value='';
                        end
                    end
                    
                    
                    %rank features if requested
                    if ~ischar(ftid)
                        fprintf('''ftid'' is not alphanumeric...\n');
                        ranks=[];
                        ftidx=1:sum(ftid);
                        nid=length(ftidx);
                        return
                    else
                        fprintf('Ordering features ''%s''...\n',ftid);
                        
                        if strcmpi(selector_type,'MUI')
                            %mui based selector, can be multi-target
                            ranks=xftr_mui(xeegsamples,xmrktargets,target);
                            [ranks,ranksid]=sort(ranks,'descend');
                            m=mean(ranks); s=std(ranks);
                            num1=str2double(cutoff_value); num2=Inf;
                        elseif strcmpi(selector_type,'KLD')
                            %KLD based selector, do not use for multi-target
                            ranks=xftr_kld(xeegsamples,xmrktargets);
                            [ranks,ranksid]=sort(ranks,'descend');
                            m=mean(ranks); s=std(ranks);
                            num1=str2double(cutoff_value); num2=Inf;
                        elseif strcmpi(selector_type,'COR')
                            %COR based selector
                            ranks=xftr_r2(xeegsamples,xmrktargets);
                            [ranks,ranksid]=sort(ranks,'descend');
                            m=nanmean(ranks); s=nanstd(ranks);
                            num1=str2double(cutoff_value); num2=Inf;
                        elseif strcmpi(selector_type,'FRQ')
                            %FRQ based selector
                            if feature_type=='e' || feature_type=='h'
                                ranks=ft.eegfreqid;
                            elseif feature_type=='p' || feature_type=='d'
                                ranks=ft.freqid;
                            elseif feature_type=='s' || feature_type=='a'
                                ranks=[ft.freqid,ft.freqid];
                            elseif feature_type=='t'
                                ranks=zeros(size(xeegsamples,2),1);
                            else
                                error('parse_ftid: Cannot use FRQ with this feature-type');
                            end
                            
                            m=0; s=1;
                            
                            dash_position=strfind(cutoff_value,'-');
                            if ~isempty(dash_position)
                                %if cutoff is in the form 'minNUM-maxNUM', parse directly
                                num1=str2double(cutoff_value(1:dash_position-1));
                                num2=str2double(cutoff_value(dash_position+1:end));
                            else
                                %if cutoff is in the form 'NUM', assume '0-NUM'
                                num1=0;
                                num2=str2double(cutoff_value);
                            end
                            
                            [ranks,ranksid]=sort(ranks,'ascend');
                        else  %no valid selector chosen, pass all
                            fprintf('Empty or invalid selector, returning all\n');
                            ftidx=1:size(xeegsamples,2);
                            nid=length(ftidx);
                            ranks=zeros(size(ftidx));
                            return
                        end
                        
                        if feature_type=='t' && strcmpi(selector_type,'FRQ')
                            ftidx=1:size(xeegsamples,2);
                            nid=length(ftidx);
                        elseif strcmpi(cutoff_type,'z')
                            %Z-type cutoff
                            ftidx=ranksid((ranks-m)>=num1*s & (ranks-m)<=num2*s);
                            nid=length(ftidx);
                        elseif strcmpi(cutoff_type,'n')
                            %number of features-type cutoff
                            nid=min(length(ranks),str2double(cutoff_value));
                            ftidx=ranksid(1:nid);
                        else
                            %no cutoff specified (will be selected downstream)
                            ftidx=ranksid;
                            nid=0;
                        end
                        
                        fprintf('Pre-selected features %i\n',nid);
                    end
                    
                end
                
                
                
            end
            
            
            
            
        end
        
        
        
    end
    
end

