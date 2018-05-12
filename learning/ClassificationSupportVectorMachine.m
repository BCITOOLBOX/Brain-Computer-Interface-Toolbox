classdef ClassificationSupportVectorMachine <handle & SuperClass
    
    properties
        eegsamples
        xvalthr=0.70; %train-validation split
        testthr=0.1; %train-validation--test split
        nnmax=10000; %max number of examples to draw for training
        act_flgtest
        commonmode       %common mode modifier
        xvalsequential   %sequential/random train-validation split modifier
    end
    
    methods(Access=public)
        
        function object = ClassificationSupportVectorMachine(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid,act_flgtest)
            
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
            
            if nargin<10 || isempty(act_flgtest)
                object.act_flgtest=rand(1,nn)<testthr;
            end
            
                if isempty(xvalsequential)
                object.xvalsequential=false; end
        end
        
        function callTrainingScaffold(this)
            
            [ccobj, pp]=gen_tr(@owntrain,@ownclassify,this.datafiles,this.predt,this.postdt,...
                this.chid,this.ftid,this.targets,this.ft,this.ftmrk,this.testid);

        end
        
    end
    
end

