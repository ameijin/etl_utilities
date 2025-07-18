"""
Excel to MSSQL ETL Pipeline Example
==================================

This script demonstrates a complete ETL pipeline that:
1. Extracts data from an Excel spreadsheet
2. Analyzes data quality using the comprehensive analyzer
3. Transforms and cleans the data based on analysis findings
4. Loads the data into MSSQL Server using the optimized loader

Features demonstrated:
- Modern database connector with driver flexibility
- Comprehensive data quality analysis
- Automated data cleaning and transformation
- Efficient bulk loading with progress tracking
- Error handling and logging
- Configuration-driven approach

Requirements:
- pandas
- openpyxl (for Excel reading)
- sqlalchemy
- pyodbc
- rich (for progress bars)

Author: ETL Utilities Team
Date: July 2025
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from etl.database.connector import DatabaseConnector
from etl.database.mssql_loader import MsSqlLoader
from etl.dataframe.analyzer import DataFrameAnalyzer, DataQualityReport
from etl.logger import Logger


class ExcelToMSSQLETL:
    """
    Complete ETL pipeline for processing Excel files to MSSQL Server.

    This class encapsulates the entire ETL process with comprehensive
    data quality analysis and intelligent data transformation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ETL pipeline with configuration.

        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.logger = Logger().get_logger()

        # Initialize database connector
        self.db_connector = DatabaseConnector(
            host=config['database']['host'],
            instance=config['database']['instance'],
            database=config['database']['database'],
            username=config['database'].get('username', ''),
            password=config['database'].get('password', ''),
            pool_size=config['database'].get('pool_size', 10),
            max_overflow=config['database'].get('max_overflow', 20)
        )

        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.quality_report: Optional[DataQualityReport] = None

        self.logger.info("ETL Pipeline initialized successfully")

    def extract_from_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Extract data from Excel file with comprehensive error handling.

        Args:
            file_path: Path to the Excel file
            sheet_name: Optional sheet name (uses first sheet if None)

        Returns:
            DataFrame containing the extracted data
        """
        self.logger.info(f"Starting data extraction from: {file_path}")

        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Excel file not found: {file_path}")

            # Read Excel file with error handling
            read_kwargs = {
                'sheet_name': sheet_name or 0,
                'header': self.config['excel'].get('header_row', 0),
                'na_values': self.config['excel'].get('na_values', ['', 'NULL', 'null', 'N/A', 'n/a']),
                'keep_default_na': True,
                'dtype': str  # Read everything as string initially for better analysis
            }

            # Handle multiple sheets if needed
            if self.config['excel'].get('read_all_sheets', False):
                data_dict = pd.read_excel(file_path, sheet_name=None, **read_kwargs)
                # Combine all sheets if multiple sheets exist
                if len(data_dict) > 1:
                    combined_data = []
                    for sheet, df in data_dict.items():
                        df['source_sheet'] = sheet  # Add sheet name as column
                        combined_data.append(df)
                    self.raw_data = pd.concat(combined_data, ignore_index=True)
                else:
                    self.raw_data = list(data_dict.values())[0]
            else:
                self.raw_data = pd.read_excel(file_path, **read_kwargs)

            # Basic validation
            if self.raw_data.empty:
                raise ValueError("Extracted DataFrame is empty")

            self.logger.info(f"Successfully extracted {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            self.logger.info(f"Column names: {list(self.raw_data.columns)}")

            return self.raw_data

        except Exception as e:
            self.logger.error(f"Error during data extraction: {str(e)}")
            raise

    def analyze_data_quality(self) -> DataQualityReport:
        """
        Perform comprehensive data quality analysis on the extracted data.

        Returns:
            DataQualityReport containing detailed analysis results
        """
        if self.raw_data is None:
            raise ValueError("No data available for analysis. Run extract_from_excel first.")

        self.logger.info("Starting comprehensive data quality analysis")

        try:
            # Initialize analyzer with configuration
            analyzer_config = self.config.get('analyzer', {})
            analyzer = DataFrameAnalyzer(
                df=self.raw_data,
                sample_size=analyzer_config.get('sample_size'),
                parallel_processing=analyzer_config.get('parallel_processing', True),
                max_workers=analyzer_config.get('max_workers')
            )

            # Perform comprehensive analysis
            self.quality_report = analyzer.analyze_comprehensive(
                primary_key=self.config.get('primary_key'),
                decimal_places=analyzer_config.get('decimal_places', 2),
                categorical_threshold=analyzer_config.get('categorical_threshold', 0.05),
                outlier_method=analyzer_config.get('outlier_method', 'iqr')
            )

            # Log summary of findings
            self.logger.info(f"Data Quality Analysis Complete:")
            self.logger.info(f"  - Total Rows: {self.quality_report.total_rows:,}")
            self.logger.info(f"  - Total Columns: {self.quality_report.total_columns}")
            self.logger.info(f"  - Duplicate Rows: {self.quality_report.duplicate_rows} ({self.quality_report.duplicate_percentage:.1f}%)")
            self.logger.info(f"  - Data Consistency Score: {self.quality_report.data_consistency_score}/100")
            self.logger.info(f"  - Issues Found: {len(self.quality_report.issues_found)}")

            # Log specific issues
            for issue_type, description, _ in self.quality_report.issues_found:
                self.logger.warning(f"  - {issue_type.value}: {description}")

            # Log column insights
            self.logger.info(f"  - Unique Columns: {len(self.quality_report.unique_columns)}")
            self.logger.info(f"  - Empty Columns: {len(self.quality_report.empty_columns)}")
            self.logger.info(f"  - Categorical Columns: {len(self.quality_report.categorical_columns)}")
            self.logger.info(f"  - Potential Primary Keys: {self.quality_report.potential_primary_keys}")

            return self.quality_report

        except Exception as e:
            self.logger.error(f"Error during data quality analysis: {str(e)}")
            raise

    def transform_and_clean_data(self) -> pd.DataFrame:
        """
        Transform and clean data based on quality analysis findings.

        Returns:
            Cleaned and transformed DataFrame
        """
        if self.raw_data is None or self.quality_report is None:
            raise ValueError("No data or quality report available. Run extract and analyze first.")

        self.logger.info("Starting data transformation and cleaning")

        try:
            # Start with a copy of raw data
            self.cleaned_data = self.raw_data.copy()

            # 1. Remove completely empty columns
            if self.quality_report.empty_columns:
                self.logger.info(f"Removing {len(self.quality_report.empty_columns)} empty columns")
                self.cleaned_data = self.cleaned_data.drop(columns=self.quality_report.empty_columns)

            # 2. Handle duplicates based on configuration
            if self.quality_report.duplicate_rows > 0:
                duplicate_action = self.config.get('cleaning', {}).get('duplicate_action', 'remove')
                if duplicate_action == 'remove':
                    initial_count = len(self.cleaned_data)
                    self.cleaned_data = self.cleaned_data.drop_duplicates()
                    removed_count = initial_count - len(self.cleaned_data)
                    self.logger.info(f"Removed {removed_count} duplicate rows")
                elif duplicate_action == 'flag':
                    self.cleaned_data['is_duplicate'] = self.cleaned_data.duplicated()
                    self.logger.info("Flagged duplicate rows with 'is_duplicate' column")

            # 3. Clean and convert data types based on column profiles
            for profile in self.quality_report.column_profiles:
                column = profile.column_name

                if column not in self.cleaned_data.columns:
                    continue

                # Convert data types intelligently
                if profile.data_type == 'integer':
                    self.cleaned_data[column] = pd.to_numeric(
                        self.cleaned_data[column],
                        errors='coerce'
                    ).astype('Int64')  # Nullable integer

                elif profile.data_type == 'float':
                    self.cleaned_data[column] = pd.to_numeric(
                        self.cleaned_data[column],
                        errors='coerce'
                    )

                elif profile.data_type == 'boolean':
                    # Convert common boolean representations
                    bool_map = {
                        'true': True, 'false': False, 'yes': True, 'no': False,
                        'y': True, 'n': False, '1': True, '0': False,
                        1: True, 0: False
                    }
                    self.cleaned_data[column] = self.cleaned_data[column].str.lower().map(bool_map)

                elif profile.data_type == 'datetime':
                    self.cleaned_data[column] = pd.to_datetime(
                        self.cleaned_data[column],
                        errors='coerce',
                        infer_datetime_format=True
                    )

                # Handle missing values based on configuration
                missing_action = self.config.get('cleaning', {}).get('missing_value_action', 'keep')
                if profile.null_percentage > 0:
                    if missing_action == 'drop_rows':
                        self.cleaned_data = self.cleaned_data.dropna(subset=[column])
                    elif missing_action == 'fill_mean' and profile.data_type in ['integer', 'float']:
                        self.cleaned_data[column] = self.cleaned_data[column].fillna(profile.mean_value)
                    elif missing_action == 'fill_mode':
                        mode_value = self.cleaned_data[column].mode()
                        if not mode_value.empty:
                            self.cleaned_data[column] = self.cleaned_data[column].fillna(mode_value.iloc[0])
                    elif missing_action == 'fill_custom':
                        fill_values = self.config.get('cleaning', {}).get('custom_fill_values', {})
                        if column in fill_values:
                            self.cleaned_data[column] = self.cleaned_data[column].fillna(fill_values[column])

            # 4. Handle outliers if configured
            outlier_action = self.config.get('cleaning', {}).get('outlier_action', 'keep')
            if outlier_action != 'keep':
                for profile in self.quality_report.column_profiles:
                    if profile.outlier_count > 0 and profile.data_type in ['integer', 'float']:
                        column = profile.column_name
                        if column not in self.cleaned_data.columns:
                            continue

                        # Calculate outlier bounds using IQR method
                        Q1 = self.cleaned_data[column].quantile(0.25)
                        Q3 = self.cleaned_data[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outlier_mask = (
                            (self.cleaned_data[column] < lower_bound) |
                            (self.cleaned_data[column] > upper_bound)
                        )

                        if outlier_action == 'remove':
                            self.cleaned_data = self.cleaned_data[~outlier_mask]
                        elif outlier_action == 'cap':
                            self.cleaned_data.loc[self.cleaned_data[column] < lower_bound, column] = lower_bound
                            self.cleaned_data.loc[self.cleaned_data[column] > upper_bound, column] = upper_bound
                        elif outlier_action == 'flag':
                            self.cleaned_data[f'{column}_outlier'] = outlier_mask

            # 5. Add metadata columns if configured
            if self.config.get('add_metadata', True):
                self.cleaned_data['etl_processed_at'] = datetime.now()
                self.cleaned_data['etl_source_file'] = self.config.get('source_file', 'unknown')

            # 6. Validate final data
            if self.cleaned_data.empty:
                raise ValueError("Cleaned data is empty after transformation")

            self.logger.info(f"Data cleaning completed:")
            self.logger.info(f"  - Final rows: {len(self.cleaned_data):,}")
            self.logger.info(f"  - Final columns: {len(self.cleaned_data.columns)}")
            self.logger.info(f"  - Memory usage: {self.cleaned_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

            return self.cleaned_data

        except Exception as e:
            self.logger.error(f"Error during data transformation: {str(e)}")
            raise

    def load_to_mssql(self) -> None:
        """
        Load the cleaned data to MSSQL Server using the optimized loader.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Run transform_and_clean_data first.")

        self.logger.info("Starting data load to MSSQL Server")

        try:
            # Initialize MSSQL loader
            loader = MsSqlLoader(
                connector=self.db_connector,
                df=self.cleaned_data,
                schema=self.config['target']['schema'],
                table=self.config['target']['table'],
                driver=self.config['database'].get('driver', 'ODBC DRIVER 17 for SQL SERVER'),
                trusted=self.config['database'].get('trusted_connection', False)
            )

            # Check if table exists and validate structure if needed
            table_exists = loader.validate_table_exists()
            if table_exists:
                self.logger.info(f"Target table {self.config['target']['schema']}.{self.config['target']['table']} exists")

                # Get table info for validation
                table_info = loader.get_table_info()
                self.logger.info(f"Table has {len(table_info)} columns")

                # Optional: validate column compatibility
                if self.config.get('validate_schema', True):
                    self._validate_schema_compatibility(table_info)
            else:
                self.logger.warning(f"Target table does not exist. Ensure it's created before loading.")

            # Choose loading method based on configuration
            load_method = self.config.get('load_method', 'bulk')
            batch_size = self.config.get('batch_size', 1000)

            if load_method == 'bulk':
                self.logger.info(f"Using bulk insert with batch size: {batch_size}")
                loader.insert_to_table_fast(batch_size=batch_size)

            elif load_method == 'upsert':
                key_columns = self.config.get('upsert_keys', [])
                if not key_columns:
                    raise ValueError("Upsert keys must be specified for upsert operation")
                self.logger.info(f"Using upsert operation with keys: {key_columns}")
                loader.upsert_to_table(key_columns=key_columns, batch_size=batch_size)

            else:  # standard insert
                self.logger.info("Using standard row-by-row insert")
                loader.insert_to_table()

            self.logger.info("Data load completed successfully")

        except Exception as e:
            self.logger.error(f"Error during data load: {str(e)}")
            raise

    def _validate_schema_compatibility(self, table_info: pd.DataFrame) -> None:
        """
        Validate that the cleaned data is compatible with the target table schema.

        Args:
            table_info: DataFrame containing table schema information
        """
        self.logger.info("Validating schema compatibility")

        table_columns = set(table_info['COLUMN_NAME'].str.lower())
        data_columns = set(col.lower() for col in self.cleaned_data.columns)

        # Check for missing columns in target table
        missing_in_table = data_columns - table_columns
        if missing_in_table:
            self.logger.warning(f"Columns in data but not in table: {missing_in_table}")

        # Check for required columns missing in data
        required_columns = table_info[table_info['IS_NULLABLE'] == 'NO']['COLUMN_NAME'].str.lower()
        missing_required = set(required_columns) - data_columns
        if missing_required:
            raise ValueError(f"Required columns missing from data: {missing_required}")

        self.logger.info("Schema validation completed")

    def run_full_pipeline(self, excel_file_path: str) -> None:
        """
        Run the complete ETL pipeline from Excel to MSSQL.

        Args:
            excel_file_path: Path to the Excel file to process
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING EXCEL TO MSSQL ETL PIPELINE")
        self.logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # Update config with source file
            self.config['source_file'] = excel_file_path

            # Step 1: Extract
            self.logger.info("STEP 1: EXTRACTING DATA FROM EXCEL")
            self.extract_from_excel(excel_file_path)

            # Step 2: Analyze
            self.logger.info("STEP 2: ANALYZING DATA QUALITY")
            self.analyze_data_quality()

            # Step 3: Transform
            self.logger.info("STEP 3: TRANSFORMING AND CLEANING DATA")
            self.transform_and_clean_data()

            # Step 4: Load
            self.logger.info("STEP 4: LOADING DATA TO MSSQL")
            self.load_to_mssql()

            # Pipeline completion
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("=" * 80)
            self.logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total Duration: {duration}")
            self.logger.info(f"Processed {len(self.cleaned_data):,} rows")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"ETL PIPELINE FAILED: {str(e)}")
            raise

    def export_quality_report(self, output_path: str) -> None:
        """
        Export the data quality report to a JSON file.

        Args:
            output_path: Path where to save the quality report
        """
        if self.quality_report is None:
            raise ValueError("No quality report available. Run analyze_data_quality first.")

        import json

        with open(output_path, 'w') as f:
            json.dump(self.quality_report.to_dict(), f, indent=2, default=str)

        self.logger.info(f"Quality report exported to: {output_path}")


def create_sample_config() -> Dict[str, Any]:
    """
    Create a sample configuration for the ETL pipeline.

    Returns:
        Dictionary containing sample configuration
    """
    return {
        # Database connection settings
        'database': {
            'host': 'localhost',
            'instance': 'SQLEXPRESS',  # or None for default instance
            'database': 'TestDatabase',
            'username': '',  # Leave empty for Windows authentication
            'password': '',  # Leave empty for Windows authentication
            'trusted_connection': True,  # Use Windows authentication
            'driver': 'ODBC DRIVER 18 for SQL SERVER',  # or 'ODBC DRIVER 17 for SQL SERVER'
            'pool_size': 10,
            'max_overflow': 20
        },

        # Excel file settings
        'excel': {
            'header_row': 0,  # Row containing column headers (0-indexed)
            'read_all_sheets': False,  # Whether to read all sheets
            'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', '#N/A', 'None']
        },

        # Target table settings
        'target': {
            'schema': 'dbo',
            'table': 'employee_data'
        },

        # Data quality analyzer settings
        'analyzer': {
            'sample_size': None,  # None for full dataset, or specify number for large datasets
            'parallel_processing': True,
            'max_workers': None,  # None for auto-detection
            'decimal_places': 2,
            'categorical_threshold': 0.05,
            'outlier_method': 'iqr'  # 'iqr', 'zscore'
        },

        # Data cleaning settings
        'cleaning': {
            'duplicate_action': 'remove',  # 'remove', 'flag', 'keep'
            'missing_value_action': 'keep',  # 'keep', 'drop_rows', 'fill_mean', 'fill_mode', 'fill_custom'
            'custom_fill_values': {
                # 'column_name': 'fill_value'
            },
            'outlier_action': 'keep'  # 'keep', 'remove', 'cap', 'flag'
        },

        # Loading settings
        'load_method': 'bulk',  # 'bulk', 'upsert', 'standard'
        'batch_size': 1000,
        'upsert_keys': [],  # Column names to use as keys for upsert

        # Additional settings
        'primary_key': None,  # Known primary key column
        'validate_schema': True,
        'add_metadata': True
    }


def main():
    """
    Main function demonstrating the ETL pipeline usage.
    """
    # Create sample configuration
    config = create_sample_config()

    # Initialize ETL pipeline
    etl = ExcelToMSSQLETL(config)

    # Example Excel file path (update this to your actual file)
    excel_file_path = r"C:\path\to\your\sample_data.xlsx"

    # Check if file exists for demo
    if not os.path.exists(excel_file_path):
        print(f"Sample Excel file not found at: {excel_file_path}")
        print("Please update the excel_file_path variable with your actual Excel file path")

        # Create a sample Excel file for demonstration
        create_sample_excel_file("sample_employee_data.xlsx")
        excel_file_path = "sample_employee_data.xlsx"
        print(f"Created sample Excel file: {excel_file_path}")

    try:
        # Run the complete ETL pipeline
        etl.run_full_pipeline(excel_file_path)

        # Export quality report
        etl.export_quality_report("data_quality_report.json")

    except Exception as e:
        print(f"ETL Pipeline failed: {str(e)}")
        # In a production environment, you might want to send alerts here


def create_sample_excel_file(filename: str) -> None:
    """
    Create a sample Excel file for demonstration purposes.

    Args:
        filename: Name of the Excel file to create
    """
    # Create sample employee data
    data = {
        'employee_id': range(1, 101),
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'] * 20,
        'last_name': ['Smith', 'Doe', 'Johnson', 'Williams', 'Brown'] * 20,
        'email': [f'user{i}@company.com' for i in range(1, 101)],
        'department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'] * 20,
        'salary': np.random.normal(75000, 15000, 100).round(2),
        'hire_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'is_active': [True] * 95 + [False] * 5,  # Mostly active employees
        'manager_id': [None] + list(range(1, 100)),  # First employee has no manager
    }

    # Introduce some data quality issues for demonstration
    df = pd.DataFrame(data)

    # Add some missing values
    df.loc[10:15, 'manager_id'] = None
    df.loc[20:22, 'email'] = None

    # Add some duplicates
    df = pd.concat([df, df.iloc[[5, 15, 25]]], ignore_index=True)

    # Add some outliers
    df.loc[0, 'salary'] = 500000  # CEO salary
    df.loc[1, 'salary'] = 0      # Intern

    # Save to Excel
    df.to_excel(filename, index=False, sheet_name='Employees')
    print(f"Sample Excel file created: {filename}")


if __name__ == "__main__":
    main()
