import pandas as pd
import numpy as np
import logging
import sys
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class DataCuration:
    """
    A class to handle the data curation process for the given datasets.
    """

    ALLOWABLE_VALUES = {
        'Sex': ['MALE', 'FEMALE'],
        'Sample_General_Pathology': ['PRIMARY', 'METASTATIC', 'NORMAL', 'NA'],
        'Material_Type': ['SERUM', 'RNA'],
        'Result_Units': ['RPKM', 'g/L'],
        'Status': ['NA', 'NOT DONE', 'ERROR']
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.dataframes = {}
        self.final_report = pd.DataFrame()

    def load_data(self, sheet_names: List[str]) -> None:
        """
        Load data from an Excel file into pandas DataFrames.
        """
        try:
            self.dataframes = {
                name: pd.read_excel(self.file_path, sheet_name=name)
                for name in sheet_names
            }
            logging.info("Data loaded successfully from Excel file.")
        except Exception as e:
            logging.error(f"Error loading data from Excel file: {e}")
            raise

    @staticmethod
    def process_patient_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process patient clinical data.
        """
        df = df.copy()
        df['Patient_ID'] = pd.to_numeric(df['Patient  Number'], errors='coerce')
        df['Study_ID'] = df['Study_ID'].str.upper()
        df['Unique_Patient_ID'] = df['Study_ID'] + '_' + df['Patient_ID'].astype(str)
        df['Sex'] = df['Sex'].str.upper().replace({'M': 'MALE', 'F': 'FEMALE'})
        df['Sex'] = df['Sex'].where(df['Sex'].isin(['MALE', 'FEMALE']), 'NA')
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').round().astype(int)
        df.drop(columns=['Patient  Number'], inplace=True)
        logging.info("Patient data processed successfully.")
        return df

    @staticmethod
    def process_tissue_sample_metadata(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process tissue sample metadata.
        """
        df = df.copy()
        df.rename(columns={
            'Patient  Number': 'Patient_ID',
            'Sample': 'Sample_ID',
            'Sample type': 'Sample_Type',
            'Material': 'Material_Type'
        }, inplace=True)
        df['Sample_ID'] = df['Sample_ID'].str.upper()
        sample_type_mapping = {
            'Normal': 'NORMAL',
            'Liver Tumor': 'PRIMARY',
            'Metastic Lung': 'METASTATIC'
        }
        df['Sample_General_Pathology'] = df['Sample_Type'].map(sample_type_mapping).fillna('NA')
        df['Material_Type'] = df['Material_Type'].str.upper()
        df.fillna('NA', inplace=True)
        df.drop(columns=['Sample_Type', 'RIN', 'Total Reads(millions)'], inplace=True)
        logging.info("Tissue sample metadata processed successfully.")
        return df

    @staticmethod
    def process_rna_seq_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process RNA-seq data.
        """
        df_melted = df.melt(id_vars=["GeneID"], var_name="Sample_ID", value_name="Result")
        df_melted["Material_Type"] = "RNA"
        df_melted["Result_Units"] = "RPKM"
        df_melted.rename(columns={"GeneID": "Gene_Symbol"}, inplace=True)
        logging.info("RNA-seq data processed successfully.")
        return df_melted

    @staticmethod
    def process_serum_protein_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process serum protein data.
        """
        df = df.copy()
        df.rename(columns={'Sample': 'Sample_ID', 'Patient': 'Patient_ID'}, inplace=True)

        def process_value(value, conversion_factor=1):
            if value == 'QNS':
                return np.nan, 'NOT DONE'
            elif value == '<LLOQ':
                return np.nan, 'ERROR'
            else:
                try:
                    return float(value) * conversion_factor, np.nan
                except ValueError:
                    return np.nan, 'ERROR'

        df[['IL6_Value', 'IL6_Status']] = df['Serum IL-6 (g/L)'].apply(
            lambda x: pd.Series(process_value(x))
        )
        df[['IL6R_Value', 'IL6R_Status']] = df['Serum IL-6 Receptor (mg/L)'].apply(
            lambda x: pd.Series(process_value(x, conversion_factor=0.001))
        )

        df_values_long = df.melt(
            id_vars=['Patient_ID', 'Sample_ID'],
            value_vars=['IL6_Value', 'IL6R_Value'],
            var_name='Gene_Symbol',
            value_name='Result'
        )
        df_status_long = df.melt(
            id_vars=['Patient_ID', 'Sample_ID'],
            value_vars=['IL6_Status', 'IL6R_Status'],
            var_name='Gene_Symbol',
            value_name='Status'
        )

        df_values_long['Gene_Symbol'] = df_values_long['Gene_Symbol'].str.replace('_Value', '')
        df_status_long['Gene_Symbol'] = df_status_long['Gene_Symbol'].str.replace('_Status', '')

        df_long = pd.merge(
            df_values_long,
            df_status_long,
            on=['Patient_ID', 'Sample_ID', 'Gene_Symbol'],
            how='left'
        )

        df_long['Result_Units'] = 'g/L'
        df_long['Material_Type'] = 'SERUM'
        df_long['Sample_General_Pathology'] = 'NA'
        df_long.fillna('NA', inplace=True)

        logging.info("Serum protein data processed successfully.")
        return df_long[[
            'Patient_ID', 'Sample_ID', 'Gene_Symbol', 'Result',
            'Result_Units', 'Material_Type', 'Status', 'Sample_General_Pathology'
        ]]

    def merge_dataframes(
        self,
        df_patient: pd.DataFrame,
        df_tissue: pd.DataFrame,
        df_rna: pd.DataFrame,
        df_serum: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all processed DataFrames into a final report.
        """
        merged_patient_tissue = pd.merge(
            df_patient, df_tissue, on='Patient_ID', how='outer'
        )
        merged_patient_tissue_rna = pd.merge(
            merged_patient_tissue, df_rna, on=['Sample_ID', 'Material_Type'], how='outer'
        )

        merged_patient_tissue_rna['Status'] = merged_patient_tissue_rna['Result'].apply(
            lambda x: 'NOT DONE' if pd.isnull(x) else np.nan
        )

        merged_serum_output = pd.merge(
            df_patient, df_serum, on='Patient_ID', how='outer'
        )

        combined_output = pd.concat(
            [merged_patient_tissue_rna, merged_serum_output],
            ignore_index=True,
            sort=False
        )

        column_order = [
        'Study_ID', 'Patient_ID', 'Unique_Patient_ID', 'Sex', 'Age', 'Sample_ID',
        'Sample_General_Pathology', 'Material_Type', 'Gene_Symbol', 'Result',
        'Result_Units', 'Status'
         ]
        
        combined_output = combined_output[column_order]

        combined_output.fillna('NA', inplace=True)
        logging.info("DataFrames merged successfully.")
        return combined_output

    def validate_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the final report to ensure it meets data specifications.
        """
        df = df.copy()
        # Check data types
        try:
            if not df['Study_ID'].map(type).eq(str).all():
                raise ValueError("Validation Error: 'Study_ID' should be of type String.")

            if not pd.to_numeric(df['Patient_ID'], errors='coerce').notnull().all():
                raise ValueError("Validation Error: 'Patient_ID' should be numeric.")

            if not df['Unique_Patient_ID'].map(type).eq(str).all():
                raise ValueError("Validation Error: 'Unique_Patient_ID' should be of type String.")

            if not df['Sex'].isin(self.ALLOWABLE_VALUES['Sex']).all():
                raise ValueError("Validation Error: 'Sex' contains invalid values.")

            if not pd.to_numeric(df['Age'], errors='coerce').notnull().all():
                raise ValueError("Validation Error: 'Age' should be of type Integer.")

            if not df['Sample_ID'].map(type).eq(str).all():
                raise ValueError("Validation Error: 'Sample_ID' should be of type String.")

            if not df['Sample_General_Pathology'].isin(
                self.ALLOWABLE_VALUES['Sample_General_Pathology']
            ).all():
                raise ValueError("Validation Error: 'Sample_General_Pathology' contains invalid values.")

            if not df['Material_Type'].isin(self.ALLOWABLE_VALUES['Material_Type']).all():
                raise ValueError("Validation Error: 'Material_Type' contains invalid values.")

            if not df['Gene_Symbol'].map(type).eq(str).all():
                raise ValueError("Validation Error: 'Gene_Symbol' should be of type String.")

            is_valid_result = df['Result'].apply(
                lambda x: pd.to_numeric(x, errors='coerce') is not None or x == 'NA'
            )
            if not is_valid_result.all():
                raise ValueError("Validation Error: 'Result' should be numeric or 'NA'.")

            if not df['Result_Units'].isin(self.ALLOWABLE_VALUES['Result_Units']).all():
                raise ValueError("Validation Error: 'Result_Units' contains invalid values.")

            if not df['Status'].isin(self.ALLOWABLE_VALUES['Status']).all():
                raise ValueError("Validation Error: 'Status' contains invalid values.")

            # Check all-caps format
            for col in ['Study_ID', 'Sample_ID', 'Gene_Symbol']:
                if not df[col].str.isupper().all():
                    raise ValueError(f"Validation Error: '{col}' should be in uppercase.")

            # Ensure all missing values are represented as 'NA'
            if df.isnull().values.any():
                raise ValueError("Validation Error: All missing values should be represented as 'NA'.")

            logging.info("Data validation passed.")
            return df

        except ValueError as e:
            logging.error(e)
            raise

    def execute(self) -> None:
        """
        Execute the data curation workflow.
        """
        sheet_names = [
            "Patient_clinical_data",
            "Tissue Sample Metadata",
            "Serum Protein data",
            "RNA-seq (RPKM)"
        ]
        self.load_data(sheet_names)

        patient_data = self.process_patient_data(self.dataframes['Patient_clinical_data'])
        tissue_metadata = self.process_tissue_sample_metadata(self.dataframes['Tissue Sample Metadata'])
        rna_data = self.process_rna_seq_data(self.dataframes['RNA-seq (RPKM)'])
        serum_data = self.process_serum_protein_data(self.dataframes['Serum Protein data'])

        final_report = self.merge_dataframes(patient_data, tissue_metadata, rna_data, serum_data)

        # Standardize 'Result_Units' based on 'Material_Type'
        final_report['Result_Units'] = final_report.apply(
            lambda x: 'RPKM' if x['Material_Type'] == 'RNA' else
                      ('g/L' if x['Material_Type'] == 'SERUM' else x['Result_Units']),
            axis=1
        )

        # Ensure 'Sample_ID' is uppercase
        final_report['Sample_ID'] = final_report['Sample_ID'].str.upper()

        # Sort the final report
        final_report.sort_values(by=['Patient_ID', 'Material_Type', 'Gene_Symbol'], inplace=True)

        # Validate the final report
        self.final_report = self.validate_report(final_report)

        # Save to CSV
        self.final_report.to_csv('curated_data.csv', index=False)
        logging.info("Final report generated and saved to 'curated_data.csv'.")

if __name__ == "__main__":
    FILE_PATH = "raw_data.xlsx"
    data_curation = DataCuration(FILE_PATH)
    data_curation.execute()
