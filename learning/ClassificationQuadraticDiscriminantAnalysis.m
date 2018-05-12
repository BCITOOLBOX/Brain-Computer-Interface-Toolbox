classdef ClassificationQuadraticDiscriminantAnalysis <handle & SuperClass
    
    properties(Access = public)
        ctype
    end
    
    methods(Access=public)
        
        function object = ClassificationQuadraticDiscriminantAnalysis(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid,ctype)
            
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
        
        
        
        
        
        function Calculate(this)
            validatingInput(this)
            callTrainingScaffold(this)
        end
        
        function callTrainingScaffold(this)
            %% Calling training scaffold
            %function to be passed to gen_tr scaffold should only have the signature
            % function classifierObject=funcTrain(trainExamples,trainTargets,...
            %  validationExamples,validationTargets)
            func=@(train_examples,train_targets,val_examples,val_targets) ...
                owntrain(train_examples,train_targets,val_examples,val_targets,this.ctype);
            
            [ccobj, pp]=gen_tr(func,@ownclassify,this.datafiles,this.predt,this.postdt,...
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
            
        end
        
    end
end

