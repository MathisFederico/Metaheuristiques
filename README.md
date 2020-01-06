# Métaheuristiques pour TSPTW

Environnement de travail et algorithmes de métaheuristiques pour le problème du voyageur de commerce avec fenêtre temporelles (TSPTW).

Ce problème consiste à trouver le chemin le plus moins long pour passer par un nombre de villes donnée (points de coordonnées "x,y") dans chacune de leurs fenêtre de temps (entre les temps "ti" et "tf") sachant que chaque ville occupe le voyageur pour un temps ("dt") qui lui est propre.

## Liste des tâches à faire :

### Dévelloppement de l'environnement:

- Classe Problème pour le définition des paramètres du problème et le format des données
- Classe Instance pour la gestion d'une instance de problème
- Classe MetaHeuristique pour la partie à changer pour chaque algorithme de métaheuristique

### Dévelloppement des métaheuristiques:

- Recuit simulé
- Recuit quantique
- Algorithme Génétique
- Algorithme des fourmis
- Monte-Carlo Tree Search (MCTS)
- MCTS + Apprentissage par renforcement