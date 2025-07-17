from typing import Optional, Union, Any
from contextlib import contextmanager
import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from sqlalchemy import Connection, text
from sqlalchemy.exc import SQLAlchemyError

from .connector import DatabaseConnector
from .loader import Loader
from .. import constants
from ..logger import Logger

logger = Logger().get_logger()


def prepare_data(df: pd.DataFrame, schema: str, table: str) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Prepare DataFrame for MSSQL insertion with proper column handling and placeholders."""
    column_list = df.columns.tolist()
    column_list = [f'[{column}]' for column in column_list]
    column_string = ", ".join(column_list)
    location = f"[{schema}].[{table}]"
    placeholders = []

    for column in df.columns:
        series = df[column]
        series_type = series.dtype
        str_column = series.apply(str)
        max_size = str_column.str.len().max()

        if max_size > 256:
            placeholders.append('cast (:param{} as nvarchar(max))'.format(len(placeholders)))
        else:
            placeholders.append(':param{}'.format(len(placeholders)))

        # Convert numpy types to native Python types for better compatibility
        if (series_type in constants.NUMPY_BOOL_TYPES or
            series_type in constants.NUMPY_INT_TYPES or
            series_type in constants.NUMPY_FLOAT_TYPES):
            df[column] = series.tolist()

    return df, column_string, location, placeholders


class MsSqlLoader:
    """Optimized MSSQL loader that integrates with DatabaseConnector."""

    def __init__(self, connector: DatabaseConnector, df: pd.DataFrame, schema: str, table: str,
                 driver: str = "ODBC DRIVER 17 for SQL SERVER", trusted: bool = False):
        """Initialize MSSQL loader with database connector.

        Args:
            connector: DatabaseConnector instance
            df: DataFrame to load
            schema: Target schema name
            table: Target table name
            driver: ODBC driver to use
            trusted: Whether to use Windows authentication
        """
        self.connector = connector
        self.df = df.copy()  # Create a copy to avoid modifying original
        self.schema = schema
        self.table = table
        self.driver = driver
        self.trusted = trusted

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        try:
            with self.connector.get_mssql_connection(trusted=self.trusted, driver=self.driver) as conn:
                yield conn
        except Exception as e:
            logger.error(f"Failed to connect to MSSQL: {str(e)}")
            raise RuntimeError(f"Database connection failed: {str(e)}")

    def insert_to_table(self) -> None:
        """Insert DataFrame to table using standard method."""
        if self.df.empty:
            logger.warning("DataFrame is empty, skipping insert")
            return

        df, column_string, location, placeholders = prepare_data(self.df, self.schema, self.table)
        df = df.replace({np.nan: None})

        placeholder_list = ", ".join(placeholders)
        query = text(f'INSERT INTO {location} ({column_string}) VALUES ({placeholder_list})')

        with self._get_connection() as conn:
            trans = conn.begin()
            try:
                progress_location = location.replace('[', '').replace(']', '')
                with Progress(TextColumn("[progress.description]{task.description}"),
                            BarColumn(), TaskProgressColumn(), MofNCompleteColumn()) as progress:

                    task = progress.add_task(f'Loading {progress_location}', total=len(df))

                    for _, row in df.iterrows():
                        # Create parameter dict for named parameters
                        params = {f'param{i}': value for i, value in enumerate(row)}
                        conn.execute(query, params)
                        progress.update(task, advance=1)

                trans.commit()
                logger.info(f"Successfully inserted {len(df)} rows into {location}")

            except Exception as e:
                trans.rollback()
                logger.error(f'Error inserting data into {location}: {str(e)}')
                raise RuntimeError(f'Error inserting data into {location}: {str(e)}')

    def insert_to_table_fast(self, batch_size: int = 1000) -> None:
        """Insert DataFrame to table using fast bulk insert method."""
        if self.df.empty:
            logger.warning("DataFrame is empty, skipping insert")
            return

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        df, column_string, location, placeholders = prepare_data(self.df, self.schema, self.table)
        df = df.replace({np.nan: None})

        # Use positional parameters for bulk insert
        positional_placeholders = ['?' for _ in placeholders]
        placeholder_list = ", ".join(positional_placeholders)
        query = f'INSERT INTO {location} ({column_string}) VALUES ({placeholder_list})'

        # Convert DataFrame to list of tuples
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]

        with self._get_connection() as conn:
            trans = conn.begin()
            try:
                # Get raw DBAPI connection for fast_executemany
                raw_conn = conn.connection
                cursor = raw_conn.cursor()
                cursor.fast_executemany = True

                progress_location = location.replace('[', '').replace(']', '')
                with Progress(TextColumn("[progress.description]{task.description}"),
                            BarColumn(), TaskProgressColumn(), MofNCompleteColumn()) as progress:

                    task = progress.add_task(f'Fast loading {progress_location}', total=len(data))

                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        cursor.executemany(query, batch)
                        progress.update(task, advance=len(batch))

                trans.commit()
                logger.info(f"Successfully bulk inserted {len(data)} rows into {location}")

            except Exception as e:
                trans.rollback()
                logger.error(f'Error bulk inserting data into {location}: {str(e)}')
                raise RuntimeError(f'Error bulk inserting data into {location}: {str(e)}')

    def upsert_to_table(self, key_columns: list[str], batch_size: int = 1000) -> None:
        """Perform upsert (MERGE) operation on the target table.

        Args:
            key_columns: List of column names to use as merge keys
            batch_size: Number of rows to process in each batch
        """
        if self.df.empty:
            logger.warning("DataFrame is empty, skipping upsert")
            return

        if not key_columns:
            raise ValueError("key_columns cannot be empty for upsert operation")

        df, column_string, location, _ = prepare_data(self.df, self.schema, self.table)
        df = df.replace({np.nan: None})

        # Build MERGE statement
        temp_table = f"#temp_{self.table}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        # Create column definitions for temp table
        column_defs = []
        for col in df.columns:
            if df[col].dtype == 'object':
                column_defs.append(f"[{col}] NVARCHAR(MAX)")
            elif df[col].dtype in ['int64', 'int32']:
                column_defs.append(f"[{col}] BIGINT")
            elif df[col].dtype in ['float64', 'float32']:
                column_defs.append(f"[{col}] FLOAT")
            elif df[col].dtype == 'bool':
                column_defs.append(f"[{col}] BIT")
            else:
                column_defs.append(f"[{col}] NVARCHAR(MAX)")

        create_temp_sql = f"CREATE TABLE {temp_table} ({', '.join(column_defs)})"

        # Build MERGE statement
        key_conditions = " AND ".join([f"target.[{col}] = source.[{col}]" for col in key_columns])
        update_sets = ", ".join([f"target.[{col}] = source.[{col}]" for col in df.columns if col not in key_columns])
        insert_columns = ", ".join([f"[{col}]" for col in df.columns])
        insert_values = ", ".join([f"source.[{col}]" for col in df.columns])

        merge_sql = f"""
        MERGE {location} AS target
        USING {temp_table} AS source
        ON {key_conditions}
        WHEN MATCHED THEN
            UPDATE SET {update_sets}
        WHEN NOT MATCHED THEN
            INSERT ({insert_columns})
            VALUES ({insert_values});
        """

        with self._get_connection() as conn:
            trans = conn.begin()
            try:
                # Create temp table
                conn.execute(text(create_temp_sql))

                # Load data into temp table using fast method
                temp_loader = MsSqlLoader(self.connector, df, "", temp_table.replace("#temp_", ""))
                temp_loader.table = temp_table  # Override to use temp table name directly

                # Insert data in batches to temp table
                data = [tuple(row) for row in df.itertuples(index=False, name=None)]
                placeholders = ", ".join(["?" for _ in df.columns])
                insert_temp_sql = f"INSERT INTO {temp_table} VALUES ({placeholders})"

                raw_conn = conn.connection
                cursor = raw_conn.cursor()
                cursor.fast_executemany = True

                progress_location = location.replace('[', '').replace(']', '')
                with Progress(TextColumn("[progress.description]{task.description}"),
                            BarColumn(), TaskProgressColumn(), MofNCompleteColumn()) as progress:

                    task = progress.add_task(f'Upserting {progress_location}', total=len(data))

                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        cursor.executemany(insert_temp_sql, batch)
                        progress.update(task, advance=len(batch))

                # Execute MERGE
                result = conn.execute(text(merge_sql))

                trans.commit()
                logger.info(f"Successfully upserted data into {location}")

            except Exception as e:
                trans.rollback()
                logger.error(f'Error upserting data into {location}: {str(e)}')
                raise RuntimeError(f'Error upserting data into {location}: {str(e)}')

    def validate_table_exists(self) -> bool:
        """Check if the target table exists."""
        query = text("""
            SELECT COUNT(*) as table_count
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
        """)

        with self._get_connection() as conn:
            result = conn.execute(query, {"schema": self.schema, "table": self.table})
            return result.scalar() > 0

    def get_table_info(self) -> pd.DataFrame:
        """Get information about the target table structure."""
        query = text("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            ORDER BY ORDINAL_POSITION
        """)

        with self._get_connection() as conn:
            result = conn.execute(query, {"schema": self.schema, "table": self.table})
            return pd.DataFrame(result.fetchall(), columns=result.keys())


# Backward compatibility with the old interface
class MsSqlLoaderLegacy(Loader):
    """Legacy MSSQL loader for backward compatibility."""

    def __init__(self, cursor, df: pd.DataFrame, schema: str, table: str) -> None:
        super().__init__(cursor, df, schema, table)

    @staticmethod
    def insert_to_table(cursor, df: pd.DataFrame, schema: str, table: str) -> None:
        df, column_string, location, placeholders = prepare_data(df, schema, table)
        Loader._insert_to_table(column_string, cursor, df, location, placeholders)

    @staticmethod
    def insert_to_table_fast(cursor, df: pd.DataFrame, schema: str, table: str, batch_size: int = 1000) -> None:
        df, column_string, location, placeholders = prepare_data(df, schema, table)
        df = df.replace({np.nan: None})
        placeholder_list = ", ".join(placeholders)
        query = f'INSERT INTO {location} ({column_string}) VALUES ({placeholder_list});'
        logger.debug(f'Query: {query}')

        # Convert DataFrame to list of tuples
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]

        # Perform the bulk insert
        cursor.fast_executemany = True
        progress_location = location.replace('[', '').replace(']', '').replace('`', '')
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
                      MofNCompleteColumn()) as progress:
            try:
                table_task = progress.add_task(f'fast loading {progress_location}', total=len(data))
                for i in range(0, len(data), batch_size):
                    actual_batch_size = min(batch_size, len(data) - i)
                    cursor.executemany(query, data[i:i + actual_batch_size])
                    progress.update(table_task, advance=actual_batch_size)
            except Exception as e:
                cursor.rollback()
                logger.error(f'Error inserting data into {location}: {str(e)}')
                raise RuntimeError(f'Error inserting data into {location}: {str(e)}')

    def to_table(self) -> None:
        return self.insert_to_table(self._cursor, self._df, self._schema, self._table)

    def to_table_fast(self, batch_size: int = 1000) -> None:
        return self.insert_to_table_fast(self._cursor, self._df, self._schema, self._table, batch_size)
