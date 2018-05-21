classdef (Abstract) SuperClass < handle
    %SUPERCLASS Summary of this class goes here
    %   Detailed explanation goes here
    
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
        ConfusionTargets
        ConfusionOutputs
    end
    
    methods(Abstract=true)
        callTrainingScaffold(this)
    end
    
    methods(Access=public)
        function object= SuperClass(datafiles,predt,postdt,chid,ftid,targets,ft,ftmrk,testid)
            
            if nargin<5 || isempty(ftid)
                ftid=[];
            end
            
            if nargin<6 || isempty(targets)
                targets=[1 2];
            end
            
            if nargin<7 || isempty(ft)
                ft=[];
            end
            
            if nargin<8 || isempty(ftmrk)
                ftmrk=[];
            end
            
            if nargin<9 || isempty(testid)
                testid=[];
            end
            
            object.datafiles=datafiles;
            object.predt=predt;
            object.postdt=postdt;
            object.chid=chid;
            object.ftid=ftid;
            object.targets=targets;
            object.ft=ft;
            object.ftmrk=ftmrk;
            object.testid=testid;
            
            
            
        end
        
        function Calculate(this)
            validatingInput(this)
            callTrainingScaffold(this)
        end
        
        function PlotConfusionMatrix(this)
            plotConfusionWrapper(this.ConfusionTargets,this.ConfusionOutputs);
            function plotConfusionWrapper(vals,vals1)
                % Neural Net toolbox's plotconfusion wrapper to use index arrays.
                % Usage:
                %  plotConfusionWrapper(vals,vals1)
                %  vals and vals1 are index (categorical) arrays of class labels
                %  to be compared in confusion matrix.
                %  'vals' are true class targets and 'vals1' are predicted classes.
                
                %convert classification index arrays to one-hot form
                %by first calculating indices of '1' in one-hot matrix
                %and then creating the one-hot matrix and setting those
                %indices to 1
                num_classes=max(max(vals),max(vals1));
                shape=[num_classes, length(vals)];
                if length(vals)~=length(vals1)
                    printf('Targets and predictions arrays are of different length, quiting')
                    return
                end
                idx_vals=sub2ind(shape,vals(:),[1:shape(2)]');
                idx_vals1=sub2ind(shape,vals1(:),[1:shape(2)]');
                
                % set '1' at positions specified by idx_vals/idx_vals1
                onehot_vals=zeros(shape);
                onehot_vals(idx_vals)=1;
                
                onehot_vals1=zeros(shape);
                onehot_vals1(idx_vals1)=1;
                
                plotconfusion(onehot_vals,onehot_vals1)
                
                
            end
            
        end
        
        function validatingInput(this)
            if isempty(this.ftid)
                this.ftid=[]; end
            if isempty(this.targets)
                this.targets=[1 2]; end
            if isempty(this.ft) || isempty(this.ftmrk)
                this.ft=[];
                this.ftmrk=[];
            end
            if isempty(this.testid)
                this.testid=[];
            end
            
        end
        
    end
    
end

