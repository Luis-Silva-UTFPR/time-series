#!/bin/bash

# Script para executar o downloader.py para intervalos de anos de 2015 a 2022

for start_year in {2015..2022}; do
    end_year=$((start_year + 1))
    base_path="all_clouds"
    echo "Executando downloader.py para os anos $start_year e $end_year"
    python downloader.py $start_year $end_year $base_path
    echo "Conjunto de anos $start_year-$end_year concluído"
done

echo "Todos os downloads concluídos."
