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
    end
    
    methods(Abstract=true)
            callTrainingScaffold(this);
    end
    
    methods
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

