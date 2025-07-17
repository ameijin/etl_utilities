import urllib.parse
from contextlib import contextmanager
from typing import Optional, Dict
from sqlalchemy import create_engine, Engine
from sqlalchemy.pool import QueuePool


class DatabaseConnector:
    def __init__(self, host: str, port: Optional[int] = None, instance: Optional[str] = None,
                 database: str = "", username: str = "", password: str = "", **engine_kwargs):
        self.host = host
        self.port = port
        self.instance = instance
        self.database = database
        self.username = username
        self.password = password
        self.engine_kwargs = engine_kwargs
        self._engines: Dict[str, Engine] = {}

    def _get_or_create_engine(self, connection_string: str, engine_key: str) -> Engine:
        """Get cached engine or create new one with connection pooling."""
        if engine_key not in self._engines:
            default_kwargs = {
                'poolclass': QueuePool,
                'pool_size': 5,
                'max_overflow': 10,
                'pool_pre_ping': True
            }
            default_kwargs.update(self.engine_kwargs)
            self._engines[engine_key] = create_engine(connection_string, **default_kwargs)
        return self._engines[engine_key]

    @contextmanager
    def get_mssql_connection(self, trusted: bool = False, driver: str = "ODBC DRIVER 17 for SQL SERVER"):
        """Get MSSQL connection with proper resource management.

        Args:
            trusted: Whether to use Windows authentication
            driver: ODBC driver to use (default: "ODBC DRIVER 17 for SQL SERVER")
                   Common alternatives: "ODBC DRIVER 18 for SQL SERVER", "SQL Server"
        """
        if trusted:
            conn_str = f"SERVER={self.host}\\{self.instance};DATABASE={self.database};Trusted_Connection=yes;"
            key = f"mssql_trusted_{self.host}_{self.instance}_{self.database}_{driver.replace(' ', '_')}"
        else:
            conn_str = f"SERVER={self.host}\\{self.instance};DATABASE={self.database};UID={self.username};PWD={self.password};"
            key = f"mssql_user_{self.host}_{self.instance}_{self.database}_{self.username}_{driver.replace(' ', '_')}"

        quoted = urllib.parse.quote_plus(f"DRIVER={{{driver}}};" + conn_str)
        engine_url = f'mssql+pyodbc:////?odbc_connect={quoted}'
        engine = self._get_or_create_engine(engine_url, key)

        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def get_postgres_connection(self):
        """Get PostgreSQL connection with proper resource management."""
        connection_string = f'postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}'
        key = f"postgres_{self.host}_{self.port}_{self.database}_{self.username}"
        engine = self._get_or_create_engine(connection_string, key)

        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def get_mysql_connection(self):
        """Get MySQL connection with proper resource management."""
        connection_string = f'mysql+pymysql://{self.username}:{self.password}@{self.host}/{self.database}?charset=utf8mb4'
        key = f"mysql_{self.host}_{self.database}_{self.username}"
        engine = self._get_or_create_engine(connection_string, key)

        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    def close_all(self):
        """Close all cached engines and their connections."""
        for engine in self._engines.values():
            engine.dispose()
        self._engines.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()