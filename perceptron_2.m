classdef perceptron_2 < handle
    properties
        l1
        l2
        rho_adaptatif
    end
    methods
        function init(obj,ne,nc,ns)
            obj.l1 = couche;
            obj.l1.init(ne,nc);
            obj.l2 = couche;
            obj.l2.init(nc,ns);
            obj.rho_adaptatif=1;
        end

        %fonction principale d'itération,les autres sont des variantes pour le pas adaptatif
        %ou pour tracer le score en fonction des itérations
        function iteration(obj,c,data,itmax,rho)
            f = waitbar(0,'Processing perceptron');
            for i = 1:itmax
                Y1 = obj.l1.y(data);
                Y2 = obj.l2.y(Y1);
                obj.l1.w = obj.l1.w -rho*obj.l1.grade(c,data,Y1,Y2,obj.l2.w);
                obj.l1.b = obj.l1.b -rho*obj.l1.gradbe(c,data,Y1,Y2,obj.l2.w);
                obj.l2.w = obj.l2.w -rho*obj.l2.grads(c,Y1,Y2);
                obj.l2.b = obj.l2.b -rho*obj.l2.gradbs(c,Y2);
                waitbar(i/itmax,f,'Processing perceptron');
            end
            close (f);
        end
        function sortie = sortie(obj,data)
            sortie = obj.l2.y(obj.l1.y(data));
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
            score=[];
            abscissescore=[];
            iscore=0;
            for i = 1:itmax
                Y1 = obj.l1.y(data);
                Y2 = obj.l2.y(Y1);
                obj.l1.w = obj.l1.w -rho*obj.l1.grade(c,data,Y1,Y2,obj.l2.w);
                obj.l1.b = obj.l1.b -rho*obj.l1.gradbe(c,data,Y1,Y2,obj.l2.w);
                obj.l2.w = obj.l2.w -rho*obj.l2.grads(c,Y1,Y2);
                obj.l2.b = obj.l2.b -rho*obj.l2.gradbs(c,Y2);
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
        %la fonction iterationscoreadaptatif sert à calculer l'évolution du
        %pourcentage de réussite tout le long de l'apprentissage avec un
        %gradient à pas variable
        function retour = iterationscoreadaptatif(obj,c,data,itmax,nbrfigure,titre)
            f = waitbar(0,'Processing perceptron');
            score=[];
            abscissescore=[];
            iscore=0;
            N=size(data,1);
            lesf = [1/(2*N)*sum(sum(power(obj.l2.y(obj.l1.y(data))-c,2)))];
            for i = 1:itmax
                Y1 = obj.l1.y(data);
                Y2 = obj.l2.y(Y1);
                w1=obj.l1.w;
                b1=obj.l1.b;
                w2=obj.l2.w;
                b2=obj.l2.b;
                obj.l1.w = obj.l1.w -obj.rho_adaptatif*obj.l1.grade(c,data,Y1,Y2,obj.l2.w);
                obj.l1.b = obj.l1.b -obj.rho_adaptatif*obj.l1.gradbe(c,data,Y1,Y2,obj.l2.w);
                obj.l2.w = obj.l2.w -obj.rho_adaptatif*obj.l2.grads(c,Y1,Y2);
                obj.l2.b = obj.l2.b -obj.rho_adaptatif*obj.l2.gradbs(c,Y2);
                lesf(i+1)=1/(2*N)*sum(sum(power(obj.l2.y(obj.l1.y(data))-c,2)));
                if (lesf(i+1)<=lesf(i))  
                    obj.rho_adaptatif=obj.rho_adaptatif*1.5;
                elseif (lesf(i+1)>lesf(i))
                    obj.rho_adaptatif=obj.rho_adaptatif/1.5;
                    obj.l1.w=w1;
                    obj.l1.b=b1;
                    obj.l2.w=w2;
                    obj.l2.b=b2;
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
        %la fonction iterationadaptatif sert à calculer l'apprentissage
        %avec un gradient à pas variable
        function iterationadaptatif(obj,c,data,itmax)
            N=size(data,1);
            lesf = [1/(2*N)*sum(sum(power(obj.l2.y(obj.l1.y(data))-c,2)))];
            f = waitbar(0,'Processing perceptron');
            for i = 1:itmax
                Y1 = obj.l1.y(data);
                Y2 = obj.l2.y(Y1);
                w1=obj.l1.w;
                b1=obj.l1.b;
                w2=obj.l2.w;
                b2=obj.l2.b;
                obj.l1.w = obj.l1.w -obj.rho_adaptatif*obj.l1.grade(c,data,Y1,Y2,obj.l2.w);
                obj.l1.b = obj.l1.b -obj.rho_adaptatif*obj.l1.gradbe(c,data,Y1,Y2,obj.l2.w);
                obj.l2.w = obj.l2.w -obj.rho_adaptatif*obj.l2.grads(c,Y1,Y2);
                obj.l2.b = obj.l2.b -obj.rho_adaptatif*obj.l2.gradbs(c,Y2);
                lesf(i+1)=1/(2*N)*sum(sum(power(obj.l2.y(obj.l1.y(data))-c,2)));
                if (lesf(i+1)<=lesf(i))  
                    obj.rho_adaptatif=obj.rho_adaptatif*1.5;
                elseif (lesf(i+1)>lesf(i))
                    obj.rho_adaptatif=obj.rho_adaptatif/1.5;
                    obj.l1.w=w1;
                    obj.l1.b=b1;
                    obj.l2.w=w2;
                    obj.l2.b=b2;
                    lesf(i+1) = lesf(i);
                end
                waitbar(i/itmax,f,'Processing perceptron');
            end
            close (f);
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
                Z(i,j) = obj.sortie([X(i,j) Y(i,j)]);
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
            if c(i)==0 && obj.sortie(data(i,:))<0.5
                % si la classe vaut 0 et la sortie est inférieur à 0.5 
                % on a bien classer le point dans classe 0 
                plot3(data(i,1),data(i,2),1,'+r')
            elseif c(i)==0 && obj.sortie(data(i,:))>0.5
                % si la classe vaut 0 et la sortie est supérieur à 0.5 
                % on a classer le point dans classe 1 au lieu de 0
                plot3(data(i,1),data(i,2),1,'pentagram',"MarkerEdgeColor",'k',"MarkerFaceColor",'w')
            elseif c(i)~=0 && obj.sortie(data(i,:))>0.5
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
          Y = obj.sortie(data);
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
        function confusion(obj,c,data,titre)
            figure
            vect = 0:9;
            [~,imax]=max(obj.sortie(data),[],2);
            cm = confusionchart(vect * transpose(c), transpose(imax)-1);
            cm.ColumnSummary = 'column-normalized';
            cm.RowSummary = 'row-normalized';
            cm.Title = titre;
        end
    end
end