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
    elif 'c_' in f:
        if len(f) > seuil:
            csv.write(f+',chinese,false\n')
        else:
            csv.write(f+',chinese,true\n')
