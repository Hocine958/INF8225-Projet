@echo off
echo "Copie des images"
python sort_Data.py
echo "Création du csv"
python create_csv.py ./data/allSignatures.csv ./data