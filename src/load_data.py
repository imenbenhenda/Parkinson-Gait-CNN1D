import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

def load_demographics(data_dir: str) -> pd.DataFrame:
    """
    Loads the demographics.xls file, cleans group names and returns it.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A pandas DataFrame with demographic information.
    """
    demo_file = Path(data_dir) / "demographics.xls"
    demographics = pd.read_excel(demo_file)
    
    # Replace 'PD' (Parkinson Disease) with 'PT' (Patient) for consistency.
    demographics['Group'] = demographics['Group'].replace({'PD': 'PT'})
    
    return demographics


def get_file_labels(data_dir: str, demographics: pd.DataFrame) -> Tuple[List[Path], Dict[str, str]]:
    """
    Maps each signal file (.txt) to its label (CO or PT).

    Args:
        data_dir (str): The path to the directory containing signal files.
        demographics (pd.DataFrame): The DataFrame loaded by load_demographics.

    Returns:
        Tuple[List[Path], Dict[str, str]]: 
            - A list of full paths to signal files.
            - A dictionary mapping each filename to its label.
    """
    id_to_label = dict(zip(demographics['ID'], demographics['Group']))
    
    raw_dir = Path(data_dir)
    
    # Select only relevant signal files.
    all_files = [f for f in raw_dir.glob("*.txt") if f.name.startswith(("Ga", "Ju", "Si"))]
    
    file_labels = {}
    for f in all_files:
        # Extract patient ID from filename (ex: "Ga01_01.txt" -> "Ga01").
        file_id = f.stem.split("_")[0]
        label = id_to_label.get(file_id, "Unknown")
        file_labels[f.name] = label
        
    return all_files, file_labels
