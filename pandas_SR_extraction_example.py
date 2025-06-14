# New (clear) window for pandas_SR_extraction_example.py

import pandas as pd
import os
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Define and validate file paths."""
    input_file = Path("/Users/gustavopadovezi/Downloads/review_485828_select_csv_20250614050901.xlsx")
    output_file = input_file.parent / f"{input_file.stem}_filtered{input_file.suffix}"
    
    if not input_file.exists():
        raise FileNotFoundError(f"❌ Erro: O arquivo não foi encontrado em:\n{input_file}")
    
    return input_file, output_file

def filter_studies(df):
    """Filter studies based on type mentioned in the abstract and participant conditions."""
    # Filter by study type keywords in the Abstract
    valid_study_types = [
        "randomized controlled trial", "randomised controlled trial", "RCT",
        "randomized trial", "randomised trial", "controlled trial",
        "parallel group", "cross-over", "cluster randomized", "cluster randomised",
        "stepped wedge", "stepped-wedge"
    ]
    pattern = '|'.join([t.lower() for t in valid_study_types])
    
    df_filtered = df[df["Abstract"].str.lower().str.contains(pattern, na=False)]
    logger.info(f"Estudos filtrados por tipo no abstract: {len(df_filtered)}/{len(df)}")
    
    # Filter out non-musculoskeletal conditions (if you want to keep this step and have a relevant column)
    # If not, you can comment out or remove the next block
    excluded_conditions = [
        "headache", "migraine", "abdominal", "ibs", "rap", "tth", "dysmenorrhoea",
        "irritable", "functional", "sickle", "neurofibromatosis", "ibd", "gut", "bowel"
    ]
    pattern_exclude = '|'.join(excluded_conditions)
    if "Type of participants" in df_filtered.columns:
        df_filtered = df_filtered[
            ~df_filtered["Type of participants"].str.lower().str.contains(pattern_exclude, na=False)
        ]
        logger.info(f"Estudos após exclusão de condições não musculoesqueléticas: {len(df_filtered)}")
    
    return df_filtered

def extract_info_from_abstract(row):
    """Extract structured information from abstract text."""
    abstract = str(row["Abstract"]).lower()
    study_id = row["Covidence #"]
    reference = f"{row['Authors'].split(';')[0]} et al., {row['Published Year']}"
    
    # Initialize result dictionary
    result = {
        "Study ID": study_id,
        "Reference /Bibliography": reference,
        "Type of studies": "NR",
        "Year of publication": row["Published Year"],
        "Language": "English",  # Assuming English since we're reading it
        "Country": "NR",
        "Clinical trial registration": "NR",
        "Sample size (n randomised)": "NR",
        "Type of participants": "NR",
        "Type of intervention": "NR",
        "Comparators": "NR",
        "Outcomes": "NR",
        "Measures of treatment effect": "NR",
        "Duration of participation": "NR",
        "Missing data": "NR",
        "Reason for missing data": "NR",
        "What were the inclusion criteria?": "NR",
        "What was the mean/SD baseline pain?": "NR",
        "Participants age (baseline)": "NR",
        "Age SD (baseline)": "NR",
        "Sex (M/F)": "NR",
        "Duration of pain": "NR",
        "Frequency of intervention receipt": "NR",
        "Duration of intervention receipt": "NR",
        "Duration follow-up": "NR",
        "Timepoint outcome assessment": "NR"
    }
    
    # Extract study type
    study_types = {
        "randomized controlled trial": "RCT",
        "randomised controlled trial": "RCT",
        "randomized trial": "RCT",
        "randomised trial": "RCT",
        "controlled trial": "Controlled Trial",
        "parallel group": "Parallel RCT",
        "cross-over": "Cross-over RCT",
        "cluster randomized": "Cluster RCT",
        "cluster randomised": "Cluster RCT",
        "stepped wedge": "Stepped Wedge RCT",
        "stepped-wedge": "Stepped Wedge RCT"
    }
    
    for type_text, type_value in study_types.items():
        if type_text in abstract:
            result["Type of studies"] = type_value
            break
    
    # Extract sample size
    sample_size_patterns = [
        r'n\s*=\s*(\d+)',  # n = 100
        r'(\d+)\s*participants',  # 100 participants
        r'(\d+)\s*patients',  # 100 patients
        r'(\d+)\s*subjects',  # 100 subjects
        r'(\d+)\s*children',  # 100 children
        r'(\d+)\s*adolescents',  # 100 adolescents
        r'(\d+)\s*individuals'  # 100 individuals
    ]
    
    for pattern in sample_size_patterns:
        match = re.search(pattern, abstract)
        if match:
            result["Sample size (n randomised)"] = match.group(1)
            break
    
    # Extract participant type
    participant_patterns = [
        r'children|adolescents|youth|pediatric|paediatric',
        r'patients with (\w+)',
        r'individuals with (\w+)',
        r'subjects with (\w+)'
    ]
    
    for pattern in participant_patterns:
        match = re.search(pattern, abstract)
        if match:
            result["Type of participants"] = match.group(0)
            break
    
    # Extract intervention type
    intervention_terms = [
        "exercise", "therapy", "treatment", "intervention", "program",
        "rehabilitation", "training", "care", "management", "approach",
        "protocol", "regimen", "strategy", "technique", "method",
        "physical therapy", "occupational therapy", "cognitive behavioral therapy",
        "cbt", "mindfulness", "meditation", "yoga", "pilates",
        "strength training", "aerobic exercise", "stretching", "massage",
        "acupuncture", "manual therapy", "education", "counseling",
        "support group", "self-management", "lifestyle modification"
    ]
    
    # Exclude terms that might indicate objectives or other non-intervention content
    exclude_terms = [
        "objective", "purpose", "aim", "goal", "question", "hypothesis",
        "evaluate", "assess", "examine", "investigate", "study", "trial",
        "background", "settings", "methods", "however", "according"
    ]
    
    # Split abstract into sentences (simple split on period, can be improved)
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    intervention_sentence = None
    for sentence in sentences:
        # Skip sentences that start with or contain exclude terms
        if any(term in sentence.lower() for term in exclude_terms):
            continue
        # Look for sentences containing intervention terms
        if any(term in sentence.lower() for term in intervention_terms):
            intervention_sentence = sentence.strip()
            break
    
    # If no specific intervention found, try to find sentences with intervention-related keywords
    if not intervention_sentence:
        intervention_keywords = [
            "intervention", "treatment", "therapy", "program", "approach"
        ]
        for sentence in sentences:
            # Skip sentences that start with or contain exclude terms
            if any(term in sentence.lower() for term in exclude_terms):
                continue
            if any(keyword in sentence.lower() for keyword in intervention_keywords):
                intervention_sentence = sentence.strip()
                break
    
    result["Type of intervention"] = intervention_sentence if intervention_sentence else "NR"
    
    # Extract outcomes
    outcome_terms = [
        "pain", "disability", "nprs", "rolland morris disability", "quality of life",
        "anxiety", "depression", "health", "symptom", "function", "mobility",
        "strength", "endurance", "flexibility", "balance", "questionnaire",
        "scale", "index", "measure", "assessment", "test", "outcome measure",
        "visual analogue scale", "vas", "numeric rating scale", "nrs",
        "functional disability", "physical function", "mental health",
        "psychological", "emotional", "social", "cognitive", "behavioral",
        "fatigue", "sleep", "activity", "participation", "satisfaction",
        "adherence", "compliance", "side effects", "adverse events"
    ]
    
    # First try to find sentences with specific outcome terms
    for sentence in sentences:
        if any(term in sentence.lower() for term in outcome_terms):
            result["Outcomes"] = sentence.strip()
            break
    
    # If no specific outcomes found, try to find sentences with outcome-related keywords
    if result["Outcomes"] == "NR":
        outcome_keywords = [
            "outcome", "result", "finding", "effect", "impact",
            "measured", "assessed", "evaluated", "analyzed"
        ]
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in outcome_keywords):
                result["Outcomes"] = sentence.strip()
                break
    
    # Extract duration
    duration_patterns = [
        r'(\d+)\s*(?:week|month|year)s?',
        r'duration[s]?\s*:\s*([^.]*)',
        r'period[s]?\s*:\s*([^.]*)',
        r'follow-up[s]?\s*:\s*([^.]*)'
    ]
    
    for pattern in duration_patterns:
        match = re.search(pattern, abstract)
        if match:
            result["Duration of participation"] = match.group(0)
            break
    
    # Extract age information
    age_patterns = [
        r'age[s]?\s*:\s*(\d+(?:-\d+)?)',
        r'(\d+(?:-\d+)?)\s*years?',
        r'mean age[s]?\s*:\s*(\d+(?:-\d+)?)',
        r'average age[s]?\s*:\s*(\d+(?:-\d+)?)'
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, abstract)
        if match:
            result["Participants age (baseline)"] = match.group(1)
            break
    
    # Extract comparators
    comparator_patterns = [
        r'compared with\s*([^.]*)',
        r'control group\s*([^.]*)',
        r'versus\s*([^.]*)'
    ]
    
    for pattern in comparator_patterns:
        match = re.search(pattern, abstract)
        if match:
            result["Comparators"] = match.group(1).strip()
            break
    
    # Extract measures of treatment effect
    treatment_effect_patterns = [
        r'measured by\s*([^.]*)',
        r'assessed by\s*([^.]*)',
        r'statistical significance\s*([^.]*)',
        r'p\s*<\s*0\.05\s*([^.]*)'  # Updated to include a capturing group
    ]
    
    for pattern in treatment_effect_patterns:
        match = re.search(pattern, abstract)
        if match:
            result["Measures of treatment effect"] = match.group(1).strip()
            break
    
    return result

def main():
    try:
        # Setup paths
        input_file, output_file = setup_paths()
        logger.info("✅ Arquivo encontrado. Iniciando leitura...")
        
        # Read Excel file
        df = pd.read_excel(input_file, header=1)
        logger.info(f"Arquivo lido com sucesso. Total de registros: {len(df)}")
        
        # Print available columns
        logger.info("Colunas disponíveis no arquivo:")
        for col in df.columns:
            logger.info(f"- {col}")
        
        # Print first 5 rows to inspect data
        logger.info("Primeiras 5 linhas do DataFrame:")
        logger.info(f"\n{df.head().to_string()}")
        
        # Apply filters
        df_filtered = filter_studies(df)
        logger.info(f"Estudos após filtragem inicial: {len(df_filtered)}")
        
        if df_filtered.empty:
            logger.info("Nenhum estudo encontrado após o filtro. Nada será salvo.")
            return
        
        # Extract information from abstracts
        logger.info("Iniciando extração de informações dos abstracts...")
        extracted_rows = df_filtered.apply(extract_info_from_abstract, axis=1)
        final_df = pd.DataFrame(extracted_rows.tolist())
        logger.info(f"Extração concluída. Total de estudos processados: {len(final_df)}")
        
        # Print sample of extracted data
        logger.info("\nAmostra dos dados extraídos (primeiros 3 estudos):")
        sample_df = final_df.head(3)
        for _, row in sample_df.iterrows():
            logger.info("\nEstudo:")
            for col in final_df.columns:
                if row[col] != "NR":  # Only show non-NR values
                    logger.info(f"{col}: {row[col]}")
        
        # Save results
        final_df.to_excel(output_file, index=False)
        logger.info(f"✅ Processamento concluído. Planilha salva em:\n{output_file}")
        
    except Exception as e:
        logger.error(f"Erro durante o processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 