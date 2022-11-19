classdef perceptron_n < handle
    properties
        n_couches
        couches = {}
        rho_adaptatif
    end
    methods
        function init(obj,n_couches,n_e,array_taille_couches)
            obj.n_couches = n_couches;
            obj.couches{1} = couche;
            obj.couches{1}.init(n_e,array_taille_couches(1));
            obj.rho_adaptatif=1;
            for i = 2:n_couches
                obj.couches{i} = couche;
                obj.couches{i}.init(array_taille_couches(i-1),array_taille_couches(i));
            end
        end


        %fonction principale d'itération,les autres sont des variantes pour le pas adaptatif
        %ou pour tracer le score en fonction des itérations
        function iteration(obj,c,data,itmax,rho)
            f = waitbar(0,'Processing perceptron');
            N=size(data,1);
            for i = 1:itmax
                Y1 = obj.sortie(data,obj.n_couches);
                % calcul de mL
                temp = Y1-c;
                for k = 1:obj.n_couches
                    j = obj.n_couches - k+1;
                    Y = Y1;
                    Y1 = obj.sortie(data,j-1);
                    obj.couches{j}.b = obj.couches{j}.b - rho*(1/(2*N)*sum(temp.*(Y-Y.^2)));
                    obj.couches{j}.w = obj.couches{j}.w - rho*(1/(2*N)*transpose(Y1)*(temp.*(Y-Y.^2)));
                    % calcul de mk
                    temp = (temp.*(Y-Y.^2))*transpose(obj.couches{j}.w);
                end
                waitbar(i/itmax,f,'Processing perceptron');
            end
            close (f);
        end

        %la fonction train sert à entrainer le perceptron, cette fonction ne sert qu'à gérer les options,
        %les itérations sont faites dans les fonctions appelées par train.
        function retour = train(obj,c,data,itmax,varargin)
            p = inputParser;
            addRequired(p,'c');
            addRequired(p,'data');
            addRequired(p,'itmax');
            addParameter(p,'rho',1);
            addParameter(p,'score',0);
            addParameter(p,'adaptative',1);
            addParameter(p,'scoreFig',1);
            addParameter(p,'scoreTitle',"Score evolution along iterations");
            parse(p,c,data,itmax,varargin{:});
            
            if p.Results.adaptative
                if p.Results.score
                    retour = obj.iterationscoreadaptatif(c,data,itmax,p.Results.scoreFig,p.Results.scoreTitle);
                else
                    obj.iterationadaptatif(c,data,itmax);
                    retour = {};
                end
            else
                if p.Results.score
                    retour=obj.iterationscore(c,data,itmax,p.Results.rho,p.Results.scoreFig,p.Results.scoreTitle);
                else
                    obj.iteration(c,data,itmax,p.Results.rho);
                    retour = {};
                end
            end
        end

        %la fonction iterationscore sert à calculer l'évolution du
        %pourcentage de réussite tout le long de l'apprentissage avec un
        %gradient à pas fixe
        function retour=iterationscore(obj,c,data,itmax,rho,nbrfigure,titre)
            f = waitbar(0,'Processing perceptron');
            N=size(data,1);
            score=[];
            abscissescore=[];
            iscore=0;
            for i = 1:itmax
                Y1 = obj.sortie(data,obj.n_couches);
                % calcul de mL
                temp = Y1-c;
                for k = 1:obj.n_couches
                    j = obj.n_couches - k+1;
                    Y = Y1;
                    Y1 = obj.sortie(data,j-1);
                    obj.couches{j}.b = obj.couches{j}.b - rho*(1/(2*N)*sum(temp.*(Y-Y.^2)));
                    obj.couches{j}.w = obj.couches{j}.w - rho*(1/(2*N)*transpose(Y1)*(temp.*(Y-Y.^2)));
                    % calcul de mk
                    temp = (temp.*(Y-Y.^2))*transpose(obj.couches{j}.w);
                end
                 if mod(i,100)==0
                    iscore=iscore+1;
                    abscissescore(iscore)=i;
                    score(iscore)=obj.pourcentage(c,data);
                end
                waitbar(i/itmax,f,'Processing perceptron');
            end
            close (f);
            figure(nbrfigure);
            plot(abscissescore,score);
            title(titre)
            retour={abscissescore, score};
        end
        %la fonction iterationadaptatif sert à calculer l'apprentissage
        %avec un gradient à pas variable
        function iterationadaptatif(obj,c,data,itmax)
            f = waitbar(0,'Processing perceptron');
            N=size(data,1);
            lesf = [1/(2*N)*sum(sum(power(obj.sortie(data,obj.n_couches)-c,2)))];
            for i = 1:itmax
                Y1 = obj.sortie(data,obj.n_couches);
                % calcul de mL
                temp = Y1-c;
                for k = 1:obj.n_couches
                    j = obj.n_couches - k+1;
                    Y = Y1;
                    Y1 = obj.sortie(data,j-1);
                    obj.couches{j}.btemp=obj.couches{j}.b;
                    obj.couches{j}.wtemp=obj.couches{j}.w;
                    obj.couches{j}.b = obj.couches{j}.b - obj.rho_adaptatif*(1/(2*N)*sum(temp.*(Y-Y.^2)));
                    obj.couches{j}.w = obj.couches{j}.w - obj.rho_adaptatif*(1/(2*N)*transpose(Y1)*(temp.*(Y-Y.^2)));
                    % calcul de mk
                    temp = (temp.*(Y-Y.^2))*transpose(obj.couches{j}.w);
                end
                lesf(i+1)=1/(2*N)*sum(sum(power(obj.sortie(data,obj.n_couches)-c,2)));
                if (lesf(i+1)<=lesf(i))  
                    obj.rho_adaptatif=obj.rho_adaptatif*1.5;   
                elseif (lesf(i+1)>lesf(i))
                    obj.rho_adaptatif=obj.rho_adaptatif/1.5;
                    for k = 1:obj.n_couches
                    obj.couches{k}.b=obj.couches{k}.btemp;
                    obj.couches{k}.w=obj.couches{k}.wtemp;
                    end
                    lesf(i+1) = lesf(i);
                end
                waitbar(i/itmax,f,'Processing perceptron');
            end
            close (f);
        end
        %la fonction iterationscoreadaptatif sert à calculer l'évolution du
        %pourcentage de réussite tout le long de l'apprentissage avec un
        %gradient à pas variable
        function retour = iterationscoreadaptatif(obj,c,data,itmax,nbrfigure,titre)
            f = waitbar(0,'Processing perceptron');
            N=size(data,1);
             score=[];
            abscissescore=[];
            iscore=0;
            lesf = [1/(2*N)*sum(sum(power(obj.sortie(data,obj.n_couches)-c,2)))];
            for i = 1:itmax
                Y1 = obj.sortie(data,obj.n_couches);
                % calcul de mL
                temp = Y1-c;
                for k = 1:obj.n_couches
                    j = obj.n_couches - k+1;
                    Y = Y1;
                    Y1 = obj.sortie(data,j-1);
                    obj.couches{j}.btemp=obj.couches{j}.b;
                    obj.couches{j}.wtemp=obj.couches{j}.w;
                    obj.couches{j}.b = obj.couches{j}.b - obj.rho_adaptatif*(1/(2*N)*sum(temp.*(Y-Y.^2)));
                    obj.couches{j}.w = obj.couches{j}.w - obj.rho_adaptatif*(1/(2*N)*transpose(Y1)*(temp.*(Y-Y.^2)));
                    % calcul de mk
                    temp = (temp.*(Y-Y.^2))*transpose(obj.couches{j}.w);
                end
                lesf(i+1)=1/(2*N)*sum(sum(power(obj.sortie(data,obj.n_couches)-c,2)));
                if (lesf(i+1)<=lesf(i))  
                    obj.rho_adaptatif=obj.rho_adaptatif*1.5;
                elseif (lesf(i+1)>lesf(i))
                    obj.rho_adaptatif=obj.rho_adaptatif/1.5;
                    for k = 1:obj.n_couches
                    obj.couches{k}.b=obj.couches{k}.btemp;
                    obj.couches{k}.w=obj.couches{k}.wtemp;
                    end
                    lesf(i+1) = lesf(i);
                end
                if mod(i,100)==0
                    iscore=iscore+1;
                    abscissescore(iscore)=i;
                    score(iscore)=obj.pourcentage(c,data);
                end
                waitbar(i/itmax,f,'Processing perceptron');
            end
            close (f);
            figure(nbrfigure);
            plot(abscissescore,score);
            title(titre);
            retour={abscissescore, score};
        end

        function sortie = sortie(obj,data,n)
            if (n == 1)
                sortie = obj.couches{n}.y(data);
            elseif (n==0)
                sortie = data;
            else
                sortie = obj.couches{n}.y(obj.sortie(data,n-1));
            end
        end
        %la fonction points sert à créer l'image de la répartition des
        %points pour les problèmes 1 et 2 en indiquant ce qui sont bons ou
        %non  
        function points(obj,c, data,titre)
          N=size(data,1);
          figure(4);clf
          m = 120 ;
          x = linspace(-12,12,m) ;  
          [X,Y] = meshgrid(x,x) ; 
          Z=[];
          for i = 1:m
            for j = 1:m
                Z(i,j) = obj.sortie([X(i,j) Y(i,j)],obj.n_couches);
            end
          end
          hold on
          plot3(13,13,-1,'+r')
          plot3(13,13,-1,'pentagram',"MarkerEdgeColor",'k',"MarkerFaceColor",'w')
          plot3(13,13,-1,'+b')
          plot3(13,13,-1,'pentagram',"MarkerEdgeColor",'k',"MarkerFaceColor",'k')
          s = surf(X,Y,Z);
          view(2);
          s.EdgeColor = 'none';
          bar=colorbar;
          ylabel(bar, 'sortie perceptron')
          for i=1:N
            if c(i)==0 && obj.sortie(data(i,:),obj.n_couches)<0.5
                % si la classe vaut 0 et la sortie est inférieur à 0.5 
                % on a bien classer le point dans classe 0 
                plot3(data(i,1),data(i,2),1,'+r')
            elseif c(i)==0 && obj.sortie(data(i,:),obj.n_couches)>0.5
                % si la classe vaut 0 et la sortie est supérieur à 0.5 
                % on a classer le point dans classe 1 au lieu de 0
                plot3(data(i,1),data(i,2),1,'pentagram',"MarkerEdgeColor",'k',"MarkerFaceColor",'w')
            elseif c(i)~=0 && obj.sortie(data(i,:),obj.n_couches)>0.5
                % si la classe vaut 1 et la sortie est supérieur à 0.5 
                % on a bien classer le point dans classe 1
                plot3(data(i,1),data(i,2),1,'+b')
            else
                % si la classe vaut 1 et la sortie est inférieur à 0.5 
                % on a classer le point dans classe 0 au lieu de 1
                plot3(data(i,1),data(i,2),1,'pentagram',"MarkerEdgeColor",'k',"MarkerFaceColor",'k')
            end
          end
          axis([-12 12 -12 12])
          xlabel("x");
          ylabel("y");
          title(titre)
          hold off
          legend("classe: 0 sortie: 0","classe: 0 sortie: 1","classe: 1 sortie: 1","classe: 1 sortie: 0")
        end
        %fonction pourcentagepermet de calculer le pourcentage de réussite,
        %c'est à dire le nombre de points en sortie qui sont bien à la même
        %classe que celle attendu
        function res=pourcentage(obj,c, data)
          N=size(data,1);
          good = 0;
          Y = obj.sortie(data,obj.n_couches);
          Ypol = arrayfun(@(x) obj.polari(x),Y);
          for i=1:N
            if isequal(c(i,:),Ypol(i,:))
                good=good + 1;
            end
          end
          res=(good/N)*100;
        end
        %fonction polari met à 0 tous les points en dessous de 0.5 sinon 1
        function polar=polari(obj,num)
            if num<0.5
                polar = 0;
            else
                polar = 1;
            end
        end
        %fonction confusion permet de tracer la matrice de confusion (utile
        %pour les problèmes multi-classes)
        function confusion(obj,c,data)
            figure
            vect = 0:9;
            [~,indmax]=max(obj.sortie(data,obj.n_couches),[],2);
            cm = confusionchart(vect * transpose(c), transpose(indmax)-1);
            cm.ColumnSummary = 'column-normalized';
            cm.RowSummary = 'row-normalized';
            cm.Title = 'My perceptron Confusion Matrix';
        end
    end
end