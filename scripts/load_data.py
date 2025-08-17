import pandas as pd 
from pathlib import Path

# ----------------------------------------------------------
# Fonction pour charger le fichier "demographics.xls"
# Ce fichier contient les infos sur les patients (ID, groupe, etc.)
# ----------------------------------------------------------
def load_demographics(data_dir):
    # Construire le chemin complet vers le fichier demographics.xls
    DEMO_FILE = Path(data_dir) / "demographics.xls"
    
    # Lire le fichier Excel
    demographics = pd.read_excel(DEMO_FILE)
    
    # Remplacer les codes de groupe :
    # "PD" (Parkinson Disease) devient "PT" (Patient)
    demographics['Group'] = demographics['Group'].replace({'PD':'PT'})
    
    # Retourne un DataFrame pandas avec colonnes (ID, Group, etc.)
    return demographics


# ----------------------------------------------------------
# Fonction pour relier chaque fichier de signaux (.txt)
# à un label (CO = contrôle sain, PT = patient Parkinson)
# ----------------------------------------------------------
def get_file_labels(data_dir, demographics):
    # Créer un dictionnaire ID → label
    # Exemple : {"Ga01" : "CO", "Ju12" : "PT"}
    id_to_label = dict(zip(demographics['ID'], demographics['Group']))
    
    # Dossier contenant les fichiers bruts (.txt)
    RAW_DIR = Path(data_dir)
    
    # Récupérer seulement les fichiers dont le nom commence par Ga, Ju ou Si
    # Exemple : "Ga01_01.txt"
    all_files = [f for f in RAW_DIR.glob("*.txt") if f.name.startswith(("Ga","Ju","Si"))]
    
    # Créer un dictionnaire fichier → label
    file_labels = {}
    for f in all_files:
        # Récupérer l'ID à partir du nom de fichier
        # Exemple : "Ga01_01.txt" → "Ga01"
        file_id = f.stem.split("_")[0]
        
        # Trouver le label associé grâce au dictionnaire id_to_label
        label = id_to_label.get(file_id, "Unknown")
        
        # Ajouter l'entrée dans le dictionnaire
        # Exemple : {"Ga01_01.txt" : "CO"}
        file_labels[f.name] = label
    
    # Retourne la liste des fichiers et leur dictionnaire de labels
    return all_files, file_labels
