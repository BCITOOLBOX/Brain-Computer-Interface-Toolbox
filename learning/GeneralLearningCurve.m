classdef GeneralLearningCurve < handle
    
    properties(Access = public)
        class
        datafiles
        predt
        postdt
        chid
        ftid
        targets
        testthr=0.10;
        Mtests=5;
        xvalsequential
        Ntrains
        prc
        act_flgtest
    end
    
    
    methods(Access=public)
        
        function object= GeneralLearningCurve(class,datafiles,predt,postdt,chid,ftid,targets)
            
            if nargin<6 || isempty(ftid)
                ftid=[];
            end
            
            if nargin<7 || isempty(targets)
                targets=1;
            end
            object.class=class;
            object.datafiles=datafiles;
            object.predt=predt;
            object.postdt=postdt;
            object.chid=chid;
            object.ftid=ftid;
            object.targets=targets;
            
            
        end
        
        function PlotLearningCurve(this)
            learning_curve_calculator(this)
        end
        
        function learning_curve_calculator(this)
            fprintf('Learning curves calculation for EEG BCI...\n');
            if isempty(this.xvalsequential)
                this.xvalsequential=false;
            end

            fprintf('Preparing samples...\n');
            [ft,ftmrk]=ftprep(this.datafiles,this.predt,this.postdt,this.chid);
            this.targets=sort(this.targets);
            ttidx=find(ismember(ftmrk,this.targets));
            nn=length(ttidx);     %number of epochs
            if (this.xvalsequential)     %test subset
                fprintf('Sequential x-validation split\n');
                this.act_flgtest=(1-(1:nn)/nn)<this.testthr;
            else
                fprintf('Random x-validation split\n');
                this.act_flgtest=rand(1,nn)<this.testthr;
            end
            
            this.Ntrains=50:50:floor(nn*3/4);
            this.Mtests=5;   %passes to evaluate errorbars
            this.prc=zeros(3,this.Mtests,length(this.Ntrains));
            pcnt=0;
            ecnt=0;
            for Ntrain=this.Ntrains
                pcnt=pcnt+1;
                
                m=1;
                while m<=this.Mtests
                    %% constrain trials
                    idx=1:nn;
                    idx=idx(randperm(length(idx)));
                    idx=idx(1:min(length(idx),Ntrain));
                    ft.tridx=ttidx(idx);
                    
                    %% train classifier
                    try
                        [obj ,pp]=this.class.callTrainingScaffold();%(this.datafiles,this.predt,this.postdt,this.chid,this.ftid,this.targets,ft,ftmrk,this.act_flgtest(idx));
                    catch E
                        if ecnt<3
                            ecnt=ecnt+1;
                            continue;
                        else
                            pp(:)=NaN(3,1);
                        end
                    end
                    
                    this.prc(:,m,pcnt)=pp(:);
                    ecnt=0;
                    m=m+1;
                end
            end

            figure,errorbar(repmat(this.Ntrains',1,3),squeeze(mean(this.prc,2))',squeeze(std(this.prc,[],2))')
            legend('Train','Validation','Test','Location','SouthEast')
            grid on
        end
        
        
        
        
    end
end
