# Copie toutes les images dans un seul dossier data et rajoute 'c_' ou 'd_' au début du fichier en fonction de si il
# s'agit d'une signature chinoise en danoise (afin de differentier les images ayant le meme nom)

# Il faut placer les dossier trainingSet et Testdata_SigComp2011 dans le même répertoire que le script

import os
import shutil

if not os.path.exists('./data'):
    os.mkdir('./data')

ctr = "./trainingSet/OfflineSignatures/Chinese"
dtr = "./trainingSet/OfflineSignatures/Dutch"
ctst = "./Testdata_SigComp2011/SigComp11-Offlinetestset/Chinese"
dtst = "./Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch"

folders = {ctr, dtr, ctst, dtst}

for fsrc in folders:
    print(fsrc)
    for root, dirs, files in os.walk(fsrc):
        for file in files:
          if(file == "d_0102014_01.png"):
            print(os.path.join(root,file))
          path_file = os.path.join(root,file)
          src = "./data/" + file
          if 'Chinese' in fsrc :
            dst = "./data/c_" + file
          else :
            dst = "./data/d_" + file
          shutil.copyfile(path_file,dst)