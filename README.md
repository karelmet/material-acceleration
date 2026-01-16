# Projet accélération matérielle

L'objectif de ce projet est d'implémenter un pipeline de détection d'objet sur images satellites issues d'une base de données appelée *CADOT*. Celle-ci présente une diversité d'images avec des *bounding box* associées à plusieurs éléments de classes d'objets définis en amont, mais leur nombre est fortement déséquilibré. Afin d'utiliser les 2 modèles de détecteurs choisis que sont Faster R-CNN et YOLOv8, il faut d'abord procéder à une augmentation sélective de données. Ensuite, les modèles entraînés peuvent être évalués et analysés pour comparer leur performances.


Les premières étapes du projet étant faites sur des machines locales, l'utilisation de notebooks a été privilégiée. Cependant, les étapes d'entraînement des modèles étant plus demandantes en capacités de calculs, les notebooks ont été abandonnés pour des scripts python, utilisable sur les machines virtuelles de Gricad et disponibles dans le dossier `notebook/`. 
