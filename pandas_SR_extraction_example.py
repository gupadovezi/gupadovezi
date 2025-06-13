import pandas as pd
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Define and validate file paths."""
    base_dir = Path.home() / "Downloads"
    input_file = base_dir / "psychological_trials.xlsx"
    output_file = base_dir / "psychological_trials_filtered.xlsx"
    
    if not input_file.exists():
        raise FileNotFoundError(f"❌ Erro: O arquivo não foi encontrado em:\n{input_file}")
    
    return input_file, output_file

def filter_studies(df):
    """Filter studies based on type and participant conditions."""
    # Filter by study type
    valid_study_types = [
        "Parallel RCT", "Cross-over RCT", "Cluster RCT", "Stepped-wedge RCT"
    ]
    study_type_column = "Type of studies (Parallel RCT, Cross-Over RCT, Cluster RCT)"
    
    df_filtered = df[df[study_type_column].isin(valid_study_types)]
    logger.info(f"Estudos filtrados por tipo: {len(df_filtered)}/{len(df)}")
    
    # Filter out non-musculoskeletal conditions
    excluded_conditions = [
        "headache", "migraine", "abdominal", "ibs", "rap", "tth", "dysmenorrhoea",
        "irritable", "functional", "sickle", "neurofibromatosis", "ibd", "gut", "bowel"
    ]
    pattern = '|'.join(excluded_conditions)
    
    df_filtered = df_filtered[
        ~df_filtered["Type of participants"].str.lower().str.contains(pattern, na=False)
    ]
    logger.info(f"Estudos após exclusão de condições não musculoesqueléticas: {len(df_filtered)}")
    
    return df_filtered

def main():
    try:
        # Setup paths
        input_file, output_file = setup_paths()
        logger.info("✅ Arquivo encontrado. Iniciando leitura...")
        
        # Read Excel file
        df = pd.read_excel(input_file)
        logger.info(f"Arquivo lido com sucesso. Total de registros: {len(df)}")
        
        # Apply filters
        df_filtered = filter_studies(df)
        
        # Save results
        df_filtered.to_excel(output_file, index=False)
        logger.info(f"✅ Filtro aplicado com sucesso. Planilha salva em:\n{output_file}")
        
    except Exception as e:
        logger.error(f"Erro durante o processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 