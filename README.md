# Projet INF8225 - Détection de signatures forgées

## Installation du projet
- Récupérer les sets de données (Training set et Test set) sur le site [ICDAR 2011 Signature Verification Competition](http://iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_%28SigComp2011%29).

- Décompresser les deux archives à la racine du projet en conservant les noms de dossier "trainingSet" et "Testdata_SigComp2011".
- Executer setup.bat pour regouper les donner et normaliser.

OU pour le faire manuelement :

- Executer sort_Data.py qui céera un dossier data contenant toutes les photos.
- Executer la commande suivante pour créer le fichier .csv de classification : "python create_csv.py ./data/allSignatures.csv ./data"


Enfin les modèles sont définit dans inception.py et vgg16.py

## Installation des paquets
Voici la liste des paquets nécessaires pour faire tourner le projet :
- pytorch, torchvision, numpy, matplotlib
- panda
- skimage