# -*- coding: utf-8 -*-
import os
import sys
from shutil import copyfile

csvPath = sys.argv[1]
folder = sys.argv[2]
seuil = 12

files = os.listdir(folder)
csv = open(csvPath, "w")

csv.write('image,language,genuine\n')
for f in files:
    if 'd_' in f:
        if len(f) > seuil:
            csv.write(f+',dutch,false\n')
        else:
            csv.write(f+',dutch,true\n')
    elif 'f_' in f:
        if len(f) > seuil+3:
            csv.write(f+',french,false\n')
        else:
            csv.write(f+',french,true\n')
