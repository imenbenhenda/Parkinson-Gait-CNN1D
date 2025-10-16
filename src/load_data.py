import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

def load_demographics(data_dir: str) -> pd.DataFrame:
    """
    Charge le fichier demographics.xls, nettoie les noms de groupe et le retourne.

    Args:
        data_dir (str): Le chemin vers le dossier contenant les données.

    Returns:
        pd.DataFrame: Un DataFrame pandas avec les informations démographiques.
    """
    demo_file = Path(data_dir) / "demographics.xls"
    demographics = pd.read_excel(demo_file)
    
    # Remplacer 'PD' (Parkinson Disease) par 'PT' (Patient) pour la cohérence.
    demographics['Group'] = demographics['Group'].replace({'PD': 'PT'})
    
    return demographics


def get_file_labels(data_dir: str, demographics: pd.DataFrame) -> Tuple[List[Path], Dict[str, str]]:
    """
    Associe chaque fichier de signal (.txt) à son label (CO ou PT).

    Args:
        data_dir (str): Le chemin vers le dossier contenant les fichiers de signaux.
        demographics (pd.DataFrame): Le DataFrame chargé par load_demographics.

    Returns:
        Tuple[List[Path], Dict[str, str]]: 
            - Une liste des chemins complets vers les fichiers de signaux.
            - Un dictionnaire associant le nom de chaque fichier à son label.
    """
    id_to_label = dict(zip(demographics['ID'], demographics['Group']))
    
    raw_dir = Path(data_dir)
    
    # Sélectionner uniquement les fichiers de signaux pertinents.
    all_files = [f for f in raw_dir.glob("*.txt") if f.name.startswith(("Ga", "Ju", "Si"))]
    
    file_labels = {}
    for f in all_files:
        # Extraire l'ID du patient depuis le nom du fichier (ex: "Ga01_01.txt" -> "Ga01").
        file_id = f.stem.split("_")[0]
        label = id_to_label.get(file_id, "Unknown")
        file_labels[f.name] = label
        
    return all_files, file_labels
