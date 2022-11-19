Antoine BACH - Théo LAFOND

____________________________Readme Réseau de neurone____________________________

Description des fichiers :
  Le programme est composer de trois classes :
   - perceptron_1.m : décrit un objet perceptron d'une couche.
   - perceptron_2.m : décrit un objet perceptron de deux couches.
   - perceptron_n.m : décrit un objet perceptron de n couches.
   - couche.m : décrit une couche.
  Il faut les ouvrir dans matlab afin de commencer à les utiliser.

  Il y a également à disposition les jeux de donnés réorganisés afin de pouvoir
  les utiliser directement dans les programmes :
   - dataimg.mat : contient les images de chiffres manuscrits d'entraînement
   ainsi que leur classe.
   - dataimgt.mat : contient les images de chiffres manuscrits de test ainsi que
   leur classe.
   - dataPb12.mat : contient les données des problèmes 1 et 2 d'entraînement et
   de test transposées.
  Pour les charger, taper load('nomFichier') dans la console.

————————————————————————————————————————————————————————————————————————————————
Créer un perceptron :
  Pour créer un perceptron d'une couche, taper
                        monPerceptron = perceptron_1;
  Puis l'initialiser
                        monPerceptron.init(ne,ns);
  ne : nombre d'entrées
  ns : nombre de sorties
  ———————
  Pour créer un perceptron de deux couches, taper
                        monPerceptron = perceptron_2;
  Puis l'initialiser
                        monPerceptron.init(ne,nc,ns);
  ne : nombre d'entrées
  nc : nombre de neurones en couche 1
  ns : nombre de sorties
  ———————
  Pour créer un perceptron de n couches (n >= 1), taper
                        monPerceptron = perceptron_n;
  Puis l'initialiser
                        monPerceptron.init(n,ne,array_nc);
  n : nombre de couches n >= 1
  ne : nombre d'entrées
  array_nc : array de taille n qui contient le nombre de neurones sur chaque
  couche

————————————————————————————————————————————————————————————————————————————————
Entraîner un perceptron :
  Pour entraîner un perceptron taper
                        monPerceptron.train(c,data,itmax);

   - c : n*ns array (avec n nombre de données et ns nombre de sorties du
  perceptron) décrit la classe de data
   - data : n*ne array (ne nombre d'entrées du perceptron) est les données
  d'entrées
   - itmax : nombre d'iterations
  ———————
  Paramètres optionnels :
  Pour ajouter un parametre optionnel ajouter un argument avec le nom de celui-ci
  et un autre avec la valeur, exemple : monPerceptron.train(c,data,itmax,'rho',2);
   - rho : pas de la descente de gradient défaut = 1
   - score : 1 pour tracer l'évolution de la réussite en fonction des itérations,
  0 sinon. défaut = 0
   - adaptative : 1 pour faire une descente de gradient à pas adaptatif, 0 sinon.
  défaut = 1
   - scoreFig : numéro de la figure si on veut tracer le score. défaut = 1
   - scoreTitle : titre de la figure si on veut tracer le score.
   ———————
   Retour :
   Si on a tracé score, la valeur de retour des une liste de deux array contenant
   l'abscisse et l'ordonée du score en fonction des itérations.
   ———————
   Il est possible de lancer successivement train afin d'entrainer d'aventage le
   perceptron.
————————————————————————————————————————————————————————————————————————————————
Méthodes pour les perceptrons :

  Calculer l'array en n*ns de la sortie du perceptron pour les données d'entrées data
  Pour un objet de type perceptron_1 et perceptron_2
                        monPerceptron.sortie(data);
  Pour un objet de type perceptron_n (renvoie la sortie de la couche en parametre n)
                        monPerceptron.sortie(data,n);

  Calculer la réussite du perceptron (pourcentage de bonne classification)
                        monPerceptron.pourcentage(c,data);

  Afficher le graph des points et des sorties (décrit dans le rapport)
  Attention, ne marche qu'avec un perceptron à 2 entrées et 1 sortie.
                        monPerceptron.points(c,data,titre);

  Afficher la matrice de confusion du perceptron pour les données data et leur
  classification c.
                        monPerceptron.confusion(c,data);

  - c : n*ns array (avec n nombre de données et ns nombre de sorties du
  perceptron) décrit la classe de data
  - data : n*ne array (ne nombre d'entrées du perceptron) est les données
  d'entrées

————————————————————————————————————————————————————————————————————————————————
Enregistrement de perceptrons :
  Pour ne pas avoir à réentraîner des perceptrons nous avons enregistré dans
  bonsPerceptrons.mat des perceptrons pertinents déjà entraînés. Pour les ouvrir,
  faire load('bonsPerceptrons.mat')
