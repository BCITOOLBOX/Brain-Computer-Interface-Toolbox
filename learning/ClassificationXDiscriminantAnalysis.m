classdef ClassificationXDiscriminantAnalysis <handle & SuperClass
    
    properties(Access = public)
        ctype
    end
    
    methods(Access=public)
        
        function object = ClassificationXDiscriminantAnalysis(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid,ctype)
            
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
            
            if nargin<10 || isempty(ctype)
                object.ctype='linear';
            end
        end
        
        
        function [ccobj, pp]=callTrainingScaffold(this)
            %% Calling training scaffold
            %function to be passed to GeneralTraining scaffold should only have the signature
            % function classifierObject=funcTrain(trainExamples,trainTargets,...
            %  validationExamples,validationTargets)
            func=@(train_examples,train_targets,val_examples,val_targets) ...
                owntrain(train_examples,train_targets,val_examples,val_targets,this.ctype);
            
            [ccobj, pp]=GeneralTraining(func,@ownclassify,this.datafiles,this.predt,this.postdt,...
                this.chid,this.ftid,this.targets,this.ft,this.ftmrk,this.testid);
            
            function clobj=owntrain(train_examples,train_targets,val_examples,val_targets,cftype)
                clobj=[];
                clobj.cfexamples=train_examples;
                clobj.cftargets=train_targets;
                clobj.cftype=cftype;
            end
            
            
            %classify using SVM model object
            function labels=ownclassify(clobj,xeegsamples)
                labels=classify(xeegsamples,clobj.cfexamples,clobj.cftargets,clobj.cftype);
            end
            
            function [outclass, err, posterior, logp, coeffs] = classify(sample, training, group, type, prior)
                %CLASSIFY Discriminant analysis (override).
                %   CLASS = CLASSIFY(SAMPLE,TRAINING,GROUP) classifies each row of the data
                %   in SAMPLE into one of the groups in TRAINING.  SAMPLE and TRAINING must
                %   be matrices with the same number of columns.  GROUP is a grouping
                %   variable for TRAINING.  Its unique values define groups, and each
                %   element defines which group the corresponding row of TRAINING belongs
                %   to.  GROUP can be a categorical variable, numeric vector, a string
                %   array, or a cell array of strings.  TRAINING and GROUP must have the
                %   same number of rows.  CLASSIFY treats NaNs or empty strings in GROUP as
                %   missing values, and ignores the corresponding rows of TRAINING. CLASS
                %   indicates which group each row of SAMPLE has been assigned to, and is
                %   of the same type as GROUP.
                %
                %   CLASS = CLASSIFY(SAMPLE,TRAINING,GROUP,TYPE) allows you to specify the
                %   type of discriminant function, one of 'linear', 'quadratic',
                %   'diagLinear', 'diagQuadratic', or 'mahalanobis'.  Linear discrimination
                %   fits a multivariate normal density to each group, with a pooled
                %   estimate of covariance.  Quadratic discrimination fits MVN densities
                %   with covariance estimates stratified by group.  Both methods use
                %   likelihood ratios to assign observations to groups.  'diagLinear' and
                %   'diagQuadratic' are similar to 'linear' and 'quadratic', but with
                %   diagonal covariance matrix estimates.  These diagonal choices are
                %   examples of naive Bayes classifiers.  Mahalanobis discrimination uses
                %   Mahalanobis distances with stratified covariance estimates.  TYPE
                %   defaults to 'linear'.
                %
                %   CLASS = CLASSIFY(SAMPLE,TRAINING,GROUP,TYPE,PRIOR) allows you to
                %   specify prior probabilities for the groups in one of three ways.  PRIOR
                %   can be a numeric vector of the same length as the number of unique
                %   values in GROUP (or the number of levels defined for GROUP, if GROUP is
                %   categorical).  If GROUP is numeric or categorical, the order of PRIOR
                %   must correspond to the ordered values in GROUP, or, if GROUP contains
                %   strings, to the order of first occurrence of the values in GROUP. PRIOR
                %   can also be a 1-by-1 structure with fields 'prob', a numeric vector,
                %   and 'group', of the same type as GROUP, and containing unique values
                %   indicating which groups the elements of 'prob' correspond to. As a
                %   structure, PRIOR may contain groups that do not appear in GROUP. This
                %   can be useful if TRAINING is a subset of a larger training set.
                %   CLASSIFY ignores any groups that appear in the structure but not in the
                %   GROUPS array.  Finally, PRIOR can also be the string value 'empirical',
                %   indicating that the group prior probabilities should be estimated from
                %   the group relative frequencies in TRAINING.  PRIOR defaults to a
                %   numeric vector of equal probabilities, i.e., a uniform distribution.
                %   PRIOR is not used for discrimination by Mahalanobis distance, except
                %   for error rate calculation.
                %
                %   [CLASS,ERR] = CLASSIFY(...) returns ERR, an estimate of the
                %   misclassification error rate that is based on the training data.
                %   CLASSIFY returns the apparent error rate, i.e., the percentage of
                %   observations in the TRAINING that are misclassified, weighted by the
                %   prior probabilities for the groups.
                %
                %   [CLASS,ERR,POSTERIOR] = CLASSIFY(...) returns POSTERIOR, a matrix
                %   containing estimates of the posterior probabilities that the j'th
                %   training group was the source of the i'th sample observation, i.e.
                %   Pr{group j | obs i}.  POSTERIOR is not computed for Mahalanobis
                %   discrimination.
                %
                %   [CLASS,ERR,POSTERIOR,LOGP] = CLASSIFY(...) returns LOGP, a vector
                %   containing estimates of the logs of the unconditional predictive
                %   probability density of the sample observations, p(obs i) is the sum of
                %   p(obs i | group j)*Pr{group j} taken over all groups.  LOGP is not
                %   computed for Mahalanobis discrimination.
                %
                %   [CLASS,ERR,POSTERIOR,LOGP,COEF] = CLASSIFY(...) returns COEF, a
                %   structure array containing coefficients describing the boundary between
                %   the regions separating each pair of groups.  Each element COEF(I,J)
                %   contains information for comparing group I to group J, defined using
                %   the following fields:
                %       'type'      type of discriminant function, from TYPE input
                %       'name1'     name of first group of pair (group I)
                %       'name2'     name of second group of pair (group J)
                %       'const'     constant term of boundary equation (K)
                %       'linear'    coefficients of linear term of boundary equation (L)
                %       'quadratic' coefficient matrix of quadratic terms (Q)
                %
                %   For the 'linear' and 'diaglinear' types, the 'quadratic' field is
                %   absent, and a row x from the SAMPLE array is classified into group I
                %   rather than group J if
                %         0 < K + x*L
                %   For the other types, x is classified into group I if
                %         0 < K + x*L + x*Q*x'
                %
                %   Example:
                %      % Classify Fisher iris data using quadratic discriminant function
                %      load fisheriris
                %      x = meas(51:end,1:2);  % for illustrations use 2 species, 2 columns
                %      y = species(51:end);
                %      [c,err,post,logl,str] = classify(x,x,y,'quadratic');
                %      gscatter(x(:,1),x(:,2),y,'rb','v^')
                %
                %      % Classify a grid of values
                %      [X,Y] = meshgrid(linspace(4.3,7.9), linspace(2,4.4));
                %      X = X(:); Y = Y(:);
                %      C = classify([X Y],x,y,'quadratic');
                %      hold on; gscatter(X,Y,C,'rb','.',1,'off'); hold off
                %
                %      % Draw boundary between two regions
                %      hold on
                %      K = str(1,2).const;
                %      L = str(1,2).linear;
                %      Q = str(1,2).quadratic;
                %      f = sprintf('0 = %g + %g*x + %g*y + %g*x^2 + %g*x.*y + %g*y.^2', ...
                %                  K,L,Q(1,1),Q(1,2)+Q(2,1),Q(2,2));
                %      ezplot(f,[4 8 2 4.5]);
                %      hold off
                %      title('Classification of Fisher iris data')
                %
                %   See also TREEFIT.
                
                %   Copyright 1993-2008 The MathWorks, Inc.
                %   $Revision: 2.15.4.7.6.1 $  $Date: 2008/07/25 19:29:33 $
                
                %   References:
                %     [1] Krzanowski, W.J., Principles of Multivariate Analysis,
                %         Oxford University Press, Oxford, 1988.
                %     [2] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.
                
                if nargin < 3
                    error('stats:classify:TooFewInputs','Requires at least three arguments.');
                end
                
                % grp2idx sorts a numeric grouping var ascending, and a string grouping
                % var by order of first occurrence
                [gindex,groups] = grp2idx(group);
                nans = find(isnan(gindex));
                if ~isempty(nans)
                    training(nans,:) = [];
                    gindex(nans) = [];
                end
                ngroups = length(groups);
                gsize = hist(gindex,1:ngroups);
                nonemptygroups = find(gsize>0);
                
                [n,d] = size(training);
                if size(gindex,1) ~= n
                    error('stats:classify:InputSizeMismatch',...
                        'The length of GROUP must equal the number of rows in TRAINING.');
                elseif isempty(sample)
                    sample = zeros(0,d,class(sample));  % accept any empty array but force correct size
                elseif size(sample,2) ~= d
                    error('stats:classify:InputSizeMismatch',...
                        'SAMPLE and TRAINING must have the same number of columns.');
                end
                m = size(sample,1);
                
                if nargin < 4 || isempty(type)
                    type = 'linear';
                elseif ischar(type)
                    types = {'linear','quadratic','diaglinear','diagquadratic','mahalanobis'};
                    i = strmatch(lower(type), types);
                    if length(i) > 1
                        error('stats:classify:BadType','Ambiguous value for TYPE:  %s.', type);
                    elseif isempty(i)
                        error('stats:classify:BadType','Unknown value for TYPE:  %s.', type);
                    end
                    type = types{i};
                else
                    error('stats:classify:BadType','TYPE must be a string.');
                end
                
                % Default to a uniform prior
                if nargin < 5 || isempty(prior)
                    prior = ones(1, ngroups) / ngroups;
                    
                    % Estimate prior from relative group sizes
                elseif ischar(prior) && ~isempty(strmatch(lower(prior), 'empirical'))
                    prior = gsize(:)' / sum(gsize);
                    % Explicit prior
                elseif isnumeric(prior)
                    if min(size(prior)) ~= 1 || max(size(prior)) ~= ngroups
                        error('stats:classify:InputSizeMismatch',...
                            'PRIOR must be a vector one element for each group.');
                    elseif any(prior < 0)
                        error('stats:classify:BadPrior',...
                            'PRIOR cannot contain negative values.');
                    end
                    prior = prior(:)' / sum(prior); % force a normalized row vector
                elseif isstruct(prior)
                    [pgindex,pgroups] = grp2idx(prior.group);
                    ord = repmat(NaN,1,ngroups);
                    for i = 1:ngroups
                        j = strmatch(groups(i), pgroups(pgindex), 'exact');
                        if ~isempty(j)
                            ord(i) = j;
                        end
                    end
                    if any(isnan(ord))
                        error('stats:classify:BadPrior',...
                            'PRIOR.group must contain all of the unique values in GROUP.');
                    end
                    prior = prior.prob(ord);
                    if any(prior < 0)
                        error('stats:classify:BadPrior',...
                            'PRIOR.prob cannot contain negative values.');
                    end
                    prior = prior(:)' / sum(prior); % force a normalized row vector
                else
                    error('stats:classify:BadType',...
                        'PRIOR must be a a vector, a structure, or the string ''empirical''.');
                end
                
                % Add training data to sample for error rate estimation
                if nargout > 1
                    sample = [sample; training];
                    mm = m+n;
                else
                    mm = m;
                end
                
                gmeans = NaN(ngroups, d);
                for k = nonemptygroups
                    gmeans(k,:) = mean(training(gindex==k,:),1);
                end
                
                D = repmat(NaN, mm, ngroups);
                isquadratic = false;
                switch type
                    case 'linear'
                        if n <= ngroups
                            error('stats:classify:BadTraining',...
                                'TRAINING must have more observations than the number of groups.');
                        end
                        % Pooled estimate of covariance.  Do not do pivoting, so that A can be
                        % computed without unpermuting.  Instead use SVD to find rank of R.
                        [Q,R] = qr(training - gmeans(gindex,:), 0);
                        R = R / sqrt(n - ngroups); % SigmaHat = R'*R
                        s = svd(R);
                        %if any(s <= max(n,d) * eps(max(s)))
                        %    error('stats:classify:BadVariance',...
                        %        'The pooled covariance matrix of TRAINING must be positive definite.');
                        %end
                        logDetSigma = 2*sum(log(s+1E-6)); % avoid over/underflow
                        
                        % MVN relative log posterior density, by group, for each sample
                        for k = nonemptygroups
                            warning off all
                            A = bsxfun(@minus,sample, gmeans(k,:)) / R;
                            warning on all
                            D(:,k) = log(prior(k)) - .5*(sum(A .* A, 2) + logDetSigma);
                        end
                        
                    case 'diaglinear'
                        if n <= ngroups
                            error('stats:classify:BadTraining',...
                                'TRAINING must have more observations than the number of groups.');
                        end
                        % Pooled estimate of variance: SigmaHat = diag(S.^2)
                        S = std(training - gmeans(gindex,:)) * sqrt((n-1)./(n-ngroups))+1E-6;
                        
                        if any(S <= n * eps(max(S)))
                            error('stats:classify:BadVariance',...
                                'The pooled variances of TRAINING must be positive.');
                        end
                        logDetSigma = 2*sum(log(S)); % avoid over/underflow
                        
                        if nargout >= 5
                            R = S';
                        end
                        
                        % MVN relative log posterior density, by group, for each sample
                        for k = nonemptygroups
                            A=bsxfun(@times, bsxfun(@minus,sample,gmeans(k,:)),1./S);
                            D(:,k) = log(prior(k)) - .5*(sum(A .* A, 2) + logDetSigma);
                        end
                        
                    case {'quadratic' 'mahalanobis'}
                        if any(gsize == 1)
                            error('stats:classify:BadTraining',...
                                'Each group in TRAINING must have at least two observations.');
                        end
                        isquadratic = true;
                        logDetSigma = zeros(ngroups,1,class(training));
                        if nargout >= 5
                            R = zeros(d,d,ngroups,class(training));
                        end
                        for k = nonemptygroups
                            % Stratified estimate of covariance.  Do not do pivoting, so that A
                            % can be computed without unpermuting.  Instead use SVD to find rank
                            % of R.
                            [Q,Rk] = qr(bsxfun(@minus,training(gindex==k,:),gmeans(k,:)), 0);
                            Rk = Rk / sqrt(gsize(k) - 1); % SigmaHat = R'*R
                            s = svd(Rk);
                            %if any(s <= max(gsize(k),d) * eps(max(s)))
                            %    error('stats:classify:BadVariance',...
                            %        'The covariance matrix of each group in TRAINING must be positive definite.');
                            %end
                            logDetSigma(k) = 2*sum(log(s+1E-6)); % avoid over/underflow
                            
                            warning off all
                            A = bsxfun(@minus, sample, gmeans(k,:)) /Rk;
                            warning on all
                            switch type
                                case 'quadratic'
                                    % MVN relative log posterior density, by group, for each sample
                                    D(:,k) = log(prior(k)) - .5*(sum(A .* A, 2) + logDetSigma(k));
                                case 'mahalanobis'
                                    % Negative squared Mahalanobis distance, by group, for each
                                    % sample.  Prior probabilities are not used
                                    D(:,k) = -sum(A .* A, 2);
                            end
                            if nargout >=5 && ~isempty(Rk)
                                R(:,:,k) = inv(Rk);
                            end
                        end
                        
                    case 'diagquadratic'
                        if any(gsize <= 1)
                            error('stats:classify:BadTraining',...
                                'Each group in TRAINING must have at least two observations.');
                        end
                        isquadratic = true;
                        logDetSigma = zeros(ngroups,1,class(training));
                        if nargout >= 5
                            R = zeros(d,1,ngroups,class(training));
                        end
                        for k = nonemptygroups
                            % Stratified estimate of variance:  SigmaHat = diag(S.^2)
                            S = std(training(gindex==k,:))+1E-6;
                            if any(S <= gsize(k) * eps(max(S)))
                                error('stats:classify:BadVariance',...
                                    'The variances in each group of TRAINING must be positive.');
                            end
                            logDetSigma(k) = 2*sum(log(S)); % avoid over/underflow
                            
                            % MVN relative log posterior density, by group, for each sample
                            A=bsxfun(@times, bsxfun(@minus,sample,gmeans(k,:)),1./S);
                            D(:,k) = log(prior(k)) - .5*(sum(A .* A, 2) + logDetSigma(k));
                            if nargout >= 5
                                R(:,:,k) = 1./S';
                            end
                        end
                end
                
                % find nearest group to each observation in sample data
                [maxD,outclass] = max(D, [], 2);
                
                % Compute apparent error rate: percentage of training data that
                % are misclassified, weighted by the prior probabilities for the groups.
                if nargout > 1
                    trclass = outclass(m+(1:n));
                    outclass = outclass(1:m);
                    
                    miss = trclass ~= gindex;
                    e = repmat(NaN,ngroups,1);
                    for k = nonemptygroups
                        e(k) = sum(miss(gindex==k)) / gsize(k);
                    end
                    err = prior*e;
                end
                
                if nargout > 2
                    if strcmp(type, 'mahalanobis')
                        % Mahalanobis discrimination does not use the densities, so it's
                        % possible that the posterior probs could disagree with the
                        % classification.
                        posterior = [];
                        logp = [];
                    else
                        % Bayes' rule: first compute p{x,G_j} = p{x|G_j}Pr{G_j} ...
                        % (scaled by max(p{x,G_j}) to avoid over/underflow)
                        P = exp(bsxfun(@minus,D(1:m,:),maxD(1:m)));
                        sumP = nansum(P,2);
                        % ... then Pr{G_j|x) = p(x,G_j} / sum(p(x,G_j}) ...
                        % (numer and denom are both scaled, so it cancels out)
                        posterior = bsxfun(@times,P,1./(sumP));
                        if nargout > 3
                            % ... and unconditional p(x) = sum(p(x,G_j}).
                            % (remove the scale factor)
                            logp = log(sumP) + maxD(1:m) - .5*d*log(2*pi);
                        end
                    end
                end
                
                % Convert back to original grouping variable type
                if isa(group,'categorical')
                    labels = getlabels(group);
                    if isa(group,'nominal')
                        groups = nominal(groups,[],labels);
                    else
                        groups = ordinal(groups,[],getlabels(group));
                    end
                elseif isnumeric(group)
                    groups = str2num(char(groups));
                    groups=cast(groups,class(group));
                elseif islogical(group)
                    groups = logical(str2num(char(groups)));
                elseif ischar(group)
                    groups = char(groups);
                    %else may be iscellstr
                end
                if isvector(groups)
                    groups = groups(:);
                end
                outclass = groups(outclass,:);
                
                if nargout>=5
                    pairs = combnk(nonemptygroups,2)';
                    npairs = size(pairs,2);
                    K = zeros(1,npairs,class(training));
                    L = zeros(d,npairs,class(training));
                    if ~isquadratic
                        % Methods with equal covariances across groups
                        for j=1:npairs
                            % Compute const (K) and linear (L) coefficients for
                            % discriminating between each pair of groups
                            i1 = pairs(1,j);
                            i2 = pairs(2,j);
                            mu1 = gmeans(i1,:)';
                            mu2 = gmeans(i2,:)';
                            if ~strcmp(type,'diaglinear')
                                b = R \ ((R') \ (mu1 - mu2));
                            else
                                b = (1./R).^2 .*(mu1 -mu2);
                            end
                            L(:,j) = b;
                            K(j) = 0.5 * (mu1 + mu2)' * b;
                        end
                    else
                        % Methods with separate covariances for each group
                        Q = zeros(d,d,npairs,class(training));
                        for j=1:npairs
                            % As above, but compute quadratic (Q) coefficients as well
                            i1 = pairs(1,j);
                            i2 = pairs(2,j);
                            mu1 = gmeans(i1,:)';
                            mu2 = gmeans(i2,:)';
                            R1i = R(:,:,i1);    % note here the R array contains inverses
                            R2i = R(:,:,i2);
                            if ~strcmp(type,'diagquadratic')
                                Rm1 = R1i' * mu1;
                                Rm2 = R2i' * mu2;
                            else
                                Rm1 = R1i .* mu1;
                                Rm2 = R2i .* mu2;
                            end
                            K(j) = 0.5 * (sum(Rm1.^2) - sum(Rm2.^2));
                            if ~strcmp(type, 'mahalanobis')
                                K(j) = K(j) + 0.5 * (logDetSigma(i1)-logDetSigma(i2));
                            end
                            if ~strcmp(type,'diagquadratic')
                                L(:,j) = (R1i*Rm1 - R2i*Rm2);
                                Q(:,:,j) = -0.5 * (R1i*R1i' - R2i*R2i');
                            else
                                L(:,j) = (R1i.*Rm1 - R2i.*Rm2);
                                Q(:,:,j) = -0.5 * diag(R1i.^2 - R2i.^2);
                            end
                        end
                    end
                    
                    % For all except Mahalanobis, adjust for the priors
                    if ~strcmp(type, 'mahalanobis')
                        K = K - log(prior(pairs(1,:))) + log(prior(pairs(2,:)));
                    end
                    
                    % Return information as a structure
                    coeffs = struct('type',repmat({type},ngroups,ngroups));
                    for k=1:npairs
                        i = pairs(1,k);
                        j = pairs(2,k);
                        coeffs(i,j).name1 = groups(i,:);
                        coeffs(i,j).name2 = groups(j,:);
                        coeffs(i,j).const = -K(k);
                        coeffs(i,j).linear = L(:,k);
                        
                        coeffs(j,i).name1 = groups(j,:);
                        coeffs(j,i).name2 = groups(i,:);
                        coeffs(j,i).const = K(k);
                        coeffs(j,i).linear = -L(:,k);
                        
                        if isquadratic
                            coeffs(i,j).quadratic = Q(:,:,k);
                            coeffs(j,i).quadratic = -Q(:,:,k);
                        end
                    end
                end
                
            end
            
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
                if nargin<6 || isempty(targets)
                    targets=[1 2];
                end
                
                
                %% Prepare features
                fprintf('Preparing features...\n');
                
                %use precomputed features if ft and ftmrk were provided, otherwise
                %compute the features yourself
                if nargin<8 || isempty(ft) || isempty(ftmrk)
                    [ft,ftmrk]=ftprep(datafiles,predt,postdt,chid,commonmode,ftid);
                end
                
                %make features
                [trialExamples,trialLabels]=make_features(ft,ftmrk,ftid);
                
                
                %% Prepare trials
                %if trial-idx field of ft was passed, constrain the samples to include
                %only the specific trials selected in ft.tridx
                if isfield(ft,'tridx')
                    trialExamples=trialExamples(ft.tridx,:);
                    trialLabels=trialLabels(ft.tridx);
                end
                
                %constrain examples to contain only the desired 'targets'
                targets=sort(targets);
                idx=ismember(trialLabels,targets);
                trialExamples=trialExamples(idx,:);
                trialLabels=trialLabels(idx);
                
                fprintf('#########################\n');
                fprintf('Total examples %i\n',size(trialExamples,1));
                fprintf('Total features %i\n',size(trialExamples,2));
                fprintf('#########################\n');
                
                
                %% Prepare train-validation-test datasets
                %normalize features to zero mean/unit variance
                %(do it here so that affects all datasets below, no need to recompute)
                msamples=mean(trialExamples,1);
                ssamples=std(trialExamples,[],1)+1E-12;
                trialExamples=bsxfun(@minus,trialExamples,msamples);
                trialExamples=bsxfun(@rdivide,trialExamples,ssamples);
                
                %define {training-validation}/test split (always randomized)
                nn=size(trialExamples,1);        %number of examples
                if nargin<9 || isempty(testid)
                    testid=rand(1,nn)<testThr;
                end
                
                %define training/validation split (controlled by global xvalsequential)
                if xvalsequential
                    fprintf('Sequential x-validation split\n');
                    trainid=((1:nn)/nn)<xvalThr;
                else
                    fprintf('Random x-validation split\n');
                    trainid=rand(1,nn)<xvalThr;
                end
                
                %training dataset, randomize order
                idx=find(trainid & ~testid);
                idx=idx(randperm(length(idx)));
                idx=idx(1:min(length(idx),nnmax));
                trainExamples=trialExamples(idx,:);
                %break degeneracies (hurtful for some methods)
                trainExamples=trainExamples+1E-6*randn(size(trainExamples));
                trainTargets=trialLabels(idx);
                
                
                %validation dataset
                idx=find(~trainid & ~testid);
                idx=idx(randperm(length(idx)));
                valExamples=trialExamples(idx,:);
                valTargets=trialLabels(idx);
                
                %test dataset
                idx=find(testid);
                idx=idx(randperm(length(idx)));
                testExamples=trialExamples(idx,:);
                testTargets=trialLabels(idx);
                
                
                
                %% Prepare classifier object
                %perform feature ranking and selection
                [ftidx,ranks,nid]=parse_ftid(ftid,ft,trialExamples,trialLabels,targets);
                
                if nid>0
                    %restrict feature sets to such selected in ftidx for
                    %train and validation data
                    ctexamples=trainExamples(:,ftidx);
                    cttargets=trainTargets;
                    cvexamples=valExamples(:,ftidx);
                    cvtargets=valTargets;
                    
                    %train classifier, looping over any internal hyperparameters
                    clobj=funcTrain(ctexamples,cttargets,cvexamples,cvtargets);
                else
                    %automatically select the number of features to keep
                    dnn=25;           %initial num of features and the increment step
                    dnn_stop=4;       %number of increments without improvement before stop
                    dnn_stop2=10;     %number of increments without improvement before stop
                    dnn_cnt=0;        %counter of increments without improvement
                    dnn_cnt2=0;       %counter of increments without improvement
                    dnn_mininc=2E-2;  %minimal required improvement
                    dnn_mininc2=1E-3; %minimal required improvement
                    
                    xclobj=[];          %best classifier
                    xnid=[];          %best nid
                    xp=-[Inf,Inf];%previous performance values
                    for nid=dnn:dnn:length(ranks)
                        ftidx_=ftidx(1:nid);
                        
                        %restrict feature sets to such selected in ftidx_
                        ctexamples=trainExamples(:,ftidx_);
                        cttargets=trainTargets;
                        cvexamples=valExamples(:,ftidx_);
                        cvtargets=valTargets;
                        
                        %train classifier, looping over any internal hyperparameters
                        clobj=funcTrain(ctexamples,cttargets,cvexamples,cvtargets);
                        
                        %check performance on validation set
                        xtest=funcClassify(clobj,cvexamples);
                        p2=mean(cvtargets==xtest);
                        
                        %check performance on test set
                        ctexamples=testExamples(:,ftidx_);
                        cttargets=testTargets;
                        
                        xtest=funcClassify(clobj,ctexamples);
                        p3=mean(cttargets==xtest);
                        
                        %evaluate stopping condition -
                        % no improvement for dnn_stop increments
                        if (xp(1)>p2+dnn_mininc || xp(2)>p3+dnn_mininc)
                            dnn_cnt=dnn_cnt+1;
                        else
                            xclobj=clobj;
                            xnid=nid;
                            xp=[p2,p3];
                            dnn_cnt=0;
                        end
                        
                        if (p2<xp(1)+dnn_mininc2 && p3<xp(2)+dnn_mininc2)
                            dnn_cnt2=dnn_cnt2+1;
                        else
                            dnn_cnt2=0;
                        end
                        
                        fprintf(' best number of features search %i: (%g,%g)...\n',nid,p2,p3);
                        
                        if dnn_cnt>dnn_stop || dnn_cnt2>dnn_stop2
                            break;
                        end
                    end
                    
                    ftidx=ftidx(1:xnid);
                    clobj=xclobj;
                    
                    fprintf(' ===selected %i: (%g,%g)...\n',xnid,xp);
                end
                
                
                %check final performance on training set
                xtest=funcClassify(clobj,trainExamples(:,ftidx));
                p1=nanmean(trainTargets==xtest);
                fprintf('Training %g\n',p1);
                
                %check final performance on validation set
                xtest1=funcClassify(clobj,valExamples(:,ftidx));
                p2=nanmean(valTargets==xtest1);
                fprintf('X-validation %g\n',p2);
                
                
                %check final performance on test set
                xtest=funcClassify(clobj,testExamples(:,ftidx));
                p3=nanmean(testTargets==xtest);
                fprintf('Test %g\n',p3);
                
                %% output
                % train-validation-test performances
                pp=[p1 p2 p3];
                
                this.ConfusionTargets=valTargets;
                this.ConfusionOutputs=xtest1;
                %category classifier object
                ccobj=[];
                ccobj.ftidx=ftidx;                    %used features
                ccobj.meantrf=msamples(ftidx);        %subtracted means
                ccobj.stdtrf=ssamples(ftidx);         %divided STD
                ccobj.categoryClassifier=clobj;        %classifier object

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

