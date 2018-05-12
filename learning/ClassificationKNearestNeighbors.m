classdef ClassificationKNearestNeighbors <handle & SuperClass
    
    properties
    end
    
    methods(Access=public)
        
        
        function object = ClassificationKNearestNeighbors(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid)
            
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
        
        function Calculate(this)
            validatingInput(this)
            callTrainingScaffold(this)
        end
        
        function callTrainingScaffold(this)
            %% Calling training scaffold
            [ccobj, pp]=gen_tr(@owntrain,@ownclassify,this.datafiles,this.predt,this.postdt,...
                this.chid,this.ftid,this.targets,this.ft,this.ftmrk,this.testid);
            
            %Train classifier looping over the hyperparameter values
            %using cross-validation to select the best hyperparameter value
            function clobj=owntrain(train_examples,train_targets,val_examples,val_targets)
                fprintf('Building category classifier...\n');
                %loop over hyperparameter here
                %range of values to check for hyperparameter
                n_examples=size(train_examples,1);
                n_max=sqrt(n_examples)*3;
                hyper=[1:10,12:2:n_max];
                val_performance=zeros(size(hyper));
                val_classifier=cell(size(hyper));
                
                fprintf(' Selecting hyperparameter...\n');
                cnt=1;
                for k=hyper
                    %clobj=fitcknn(train_examples,train_targets,'NumNeighbors',k,'standardize',1);
                    clobj=ClassificationKNN.fit(train_examples,train_targets,'NumNeighbors',k);
                    labels=predict(clobj,val_examples);
                    
                    val_classifier{cnt}=clobj;
                    val_performance(cnt)=mean(labels==val_targets);
                    cnt=cnt+1;
                end
                
                %select best hyperparameter
                [~, best_xval_index]=max(val_performance);
                
                fprintf(' Selected hyperparameter %g...\n',hyper(best_xval_index));
                
                clobj=val_classifier{best_xval_index};
            end
            
            
            %classify using classifier model object
            function labels=ownclassify(clobj,examples)
                labels=predict(clobj,examples);
            end
            
        end
        
    end
end