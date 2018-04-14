classdef mcxda_tr 
    
    properties(Access = protected)
    end
    properties(Access = private)
    end
    properties(GetAccess = public,SetAccess=private)   
    end
    properties(GetAccess = public,SetAccess=protected)   
    end
    properties(Access = public)

        datafiles
        predt
        postdt
        chid
        ftid
        targets
        ft
        ftmrk
        testid
        ctype
       
    end
    methods(Access=private)
    end
    methods(Access=protected)
    end
    methods(Access=public)


%% Validating input parameters
function validatingInput(this)
if nargin<5 this.ftid=[]; end
if nargin<6 || isempty(this.targets) this.targets=[1 2]; end
if nargin<8
  this.ft=[];
  this.ftmrk=[];
end
if nargin<9
  this.testid=[];
end
if nargin<10
  this.ctype='linear';
end
end
%% Calling training scaffold
%function to be passed to gen_tr scaffold should only have the signature
% function classifierObject=funcTrain(trainExamples,trainTargets,...
%  validationExamples,validationTargets)
function callTrainingScaffold(this)
func=@(train_examples,train_targets,val_examples,val_targets) ...
  owntrain(train_examples,train_targets,val_examples,val_targets,ctype);

[ccobj, pp]=gen_tr(func,@ownclassify,this.datafiles,this.predt,this.postdt,...
  this.chid,this.ftid,this.targets,this.ft,this.ftmrk,this.testid);

end
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

