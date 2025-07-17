# Database Utilities

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
   - [Connector Class](#1-connector-class)
   - [Loader Classes](#2-loader-classes)
   - [Validator Class](#3-validator-class)
3. [General Considerations](#general-considerations)

## Overview

These utilities provide tools for connecting to various SQL databases, efficiently loading data into tables, and validating data before uploading.

## Components

### 1. Connector Class

The `Connector` class is used to establish connections to SQL databases including SQL Server, PostgreSQL, and MySQL.

#### Usage

```python
# Context manager for automatic cleanup
with DatabaseConnector(host="localhost", instance="SQLEXPRESS", database="mydb", 
                      username="user", password="pass") as db:
    
    # MSSQL with trusted connection
    with db.get_mssql_connection(trusted=True) as conn:
        result = conn.execute("SELECT * FROM users")
    
    # PostgreSQL
    with DatabaseConnector(host="localhost", port=5432, database="mydb", 
                          username="user", password="pass") as pg_db:
        with pg_db.get_postgres_connection() as conn:
            result = conn.execute("SELECT * FROM products")
```

### 2. Loader Classes

The `Loader` class and its derivatives (`MySqlLoader` and `MsSqlLoader`) handle data insertion into database tables using Pandas DataFrames.

#### Usage

```python
from etl.database.mysql_loader import MySqlLoader
from etl.database.mssql_loader import MsSqlLoader

import pandas as pd
from sqlalchemy import create_engine

# Create a database connection
engine = create_engine('your_connection_string')
connection = engine.connect()

# Assume df is your DataFrame and already exists
df = pd.DataFrame(...)

# MySQL Loader Example
mysql_loader = MySqlLoader(cursor=connection, df=df, schema='your_schema', table='your_table')
mysql_loader.to_table()

# MSSQL Loader Example
mssql_loader = MsSqlLoader(cursor=connection, df=df, schema='your_schema', table='your_table')
mssql_loader.to_table()
```

#### Fast Insertion for MSSQL

```python
mssql_loader.to_table_fast(batch_size=500)
```

### 3. Validator Class

The `Validator` class ensures that the DataFrame you are trying to upload is structured correctly to match the database schema.

#### Usage

```python
from etl.database.validator import Validator

# Assume df is your DataFrame and already exists
df = pd.DataFrame(...)

# Validate the DataFrame against the database schema
validator = Validator(connection=connection, df=df, schema='your_schema', table='your_table')
validator.validate()
```

#### Handling Exceptions

- **ExtraColumnsException**: Raised when extra columns in the DataFrame do not exist in the target table.
- **ColumnDataException**: Raised when there are data type mismatches or truncation issues.

## General Considerations

- Ensure your SQLAlchemy engine and connection strings are correctly initialized.
- Pre-process DataFrames for NaN values before database loading.
- For large datasets, use `fast_executemany` or batch insertion for efficiency.
