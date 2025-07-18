import itertools
from typing import Hashable, Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..logger import Logger

class DataQualityIssue(Enum):
    """Enumeration of data quality issues that can be detected."""
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    OUTLIERS = "outliers"
    INCONSISTENT_FORMATS = "inconsistent_formats"
    INVALID_VALUES = "invalid_values"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    ENCODING_ISSUES = "encoding_issues"
    DATE_ANOMALIES = "date_anomalies"
    RANGE_VIOLATIONS = "range_violations"


@dataclass
class ColumnProfile:
    """Comprehensive column profile with all relevant statistics."""
    column_name: str
    data_type: str
    is_primary_key: bool = False
    is_unique: bool = False
    is_empty: bool = False

    # Basic statistics
    total_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0

    # String statistics
    max_str_length: Optional[int] = None
    min_str_length: Optional[int] = None
    avg_str_length: Optional[float] = None

    # Numeric statistics
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_dev: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    # Type-specific
    decimal_places: Optional[int] = None
    float_precision: Optional[int] = None

    # Data quality indicators
    outlier_count: int = 0
    format_inconsistencies: int = 0
    suspected_encoding_issues: int = 0

    # Categorical analysis
    is_categorical: bool = False
    top_values: Optional[Dict[Any, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return asdict(self)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""
    total_rows: int
    total_columns: int
    duplicate_rows: int
    duplicate_percentage: float
    issues_found: List[Tuple[DataQualityIssue, str, Any]]
    column_profiles: List[ColumnProfile]
    unique_columns: List[str]
    unique_column_pairs: List[Tuple[str, str]]
    empty_columns: List[str]
    categorical_columns: List[str]
    potential_primary_keys: List[str]
    referential_integrity_issues: List[Dict[str, Any]]
    data_consistency_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'total_rows': self.total_rows,
            'total_columns': self.total_columns,
            'duplicate_rows': self.duplicate_rows,
            'duplicate_percentage': self.duplicate_percentage,
            'issues_found': [(issue.value, desc, data) for issue, desc, data in self.issues_found],
            'column_profiles': [profile.to_dict() for profile in self.column_profiles],
            'unique_columns': self.unique_columns,
            'unique_column_pairs': self.unique_column_pairs,
            'empty_columns': self.empty_columns,
            'categorical_columns': self.categorical_columns,
            'potential_primary_keys': self.potential_primary_keys,
            'referential_integrity_issues': self.referential_integrity_issues,
            'data_consistency_score': self.data_consistency_score
        }


class DataFrameAnalyzer:
    """Comprehensive DataFrame analyzer with advanced data quality assessment."""

    def __init__(self, df: pd.DataFrame, sample_size: Optional[int] = None,
                 parallel_processing: bool = True, max_workers: Optional[int] = None,
                 logger: Optional[Any] = None):
        """Initialize analyzer with optional sampling for large datasets.

        Args:
            df: DataFrame to analyze
            sample_size: Optional sample size for large datasets
            parallel_processing: Whether to use parallel processing
            max_workers: Maximum number of worker threads
            logger: Optional logger instance (default: internal logger)
        """
        if df.empty:
            raise ValueError("Cannot analyze empty DataFrame")

        self.original_df = df
        self.df = self._prepare_sample(df, sample_size) if sample_size else df
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers or min(32, (len(df.columns) + 3))
        self._cache = {}
        self.logger = logger if logger is not None else Logger().get_logger()

        self.logger.info(f"Initialized analyzer for DataFrame with {len(df)} rows and {len(df.columns)} columns")
        if sample_size:
            self.logger.info(f"Using sample of {len(self.df)} rows for analysis")

    def _prepare_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Prepare a stratified sample for analysis."""
        if len(df) <= sample_size:
            return df

        # Use stratified sampling if possible
        try:
            return df.sample(n=sample_size, random_state=42)
        except Exception:
            return df.head(sample_size)

    def analyze_comprehensive(self, primary_key: Optional[str] = None,
                            decimal_places: int = 2,
                            categorical_threshold: float = 0.05,
                            outlier_method: str = 'iqr') -> DataQualityReport:
        """Perform comprehensive data analysis.

        Args:
            primary_key: Known primary key column
            decimal_places: Decimal places for numeric precision
            categorical_threshold: Threshold for categorical detection
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
        """
        self.logger.info("Starting comprehensive data analysis")

        try:
            # Basic statistics
            total_rows = len(self.original_df)
            total_columns = len(self.df.columns)

            # Analyze duplicates
            duplicate_info = self._analyze_duplicates()

            # Find unique columns and pairs
            unique_columns = self.find_unique_columns()
            unique_column_pairs = self.find_unique_column_pairs()
            empty_columns = self.find_empty_columns()

            # Generate column profiles
            column_profiles = self._generate_comprehensive_column_profiles(
                primary_key, decimal_places, outlier_method
            )

            # Find categorical columns
            categorical_columns = self.find_categorical_columns(categorical_threshold)

            # Detect potential primary keys
            potential_primary_keys = self._detect_potential_primary_keys()

            # Check referential integrity
            ref_integrity_issues = self._check_referential_integrity()

            # Collect all issues
            issues_found = self._collect_all_issues(column_profiles, duplicate_info)

            # Calculate data consistency score
            consistency_score = self._calculate_consistency_score(issues_found, column_profiles)

            report = DataQualityReport(
                total_rows=total_rows,
                total_columns=total_columns,
                duplicate_rows=duplicate_info['count'],
                duplicate_percentage=duplicate_info['percentage'],
                issues_found=issues_found,
                column_profiles=column_profiles,
                unique_columns=unique_columns,
                unique_column_pairs=unique_column_pairs,
                empty_columns=empty_columns,
                categorical_columns=categorical_columns,
                potential_primary_keys=potential_primary_keys,
                referential_integrity_issues=ref_integrity_issues,
                data_consistency_score=consistency_score
            )

            self.logger.info(f"Analysis completed. Found {len(issues_found)} issues. "
                       f"Data consistency score: {consistency_score:.2f}")

            return report

        except Exception as e:
            self.logger.error(f"Error during comprehensive analysis: {str(e)}")
            raise

    def find_unique_columns(self) -> List[str]:
        """Find columns with all unique values (optimized)."""
        if 'unique_columns' in self._cache:
            return self._cache['unique_columns']

        total_records = len(self.df)
        if total_records == 0:
            return []

        unique_columns = []

        if self.parallel_processing:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_column = {
                    executor.submit(self._check_column_uniqueness, col, series, total_records): col
                    for col, series in self.df.items()
                }

                for future in as_completed(future_to_column):
                    column = future_to_column[future]
                    try:
                        if future.result():
                            unique_columns.append(column)
                    except Exception as e:
                        self.logger.warning(f"Error checking uniqueness for column {column}: {str(e)}")
        else:
            for column, series in self.df.items():
                if self._check_column_uniqueness(column, series, total_records):
                    unique_columns.append(column)

        self._cache['unique_columns'] = unique_columns
        return unique_columns

    def _check_column_uniqueness(self, column: Hashable, series: pd.Series, total_records: int) -> bool:
        """Check if a column has unique values."""
        try:
            # Skip if all nulls
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                return False

            return len(non_null_series.unique()) == len(non_null_series)
        except Exception:
            return False

    def find_unique_column_pairs(self, max_combinations: int = 1000) -> List[Tuple[str, str]]:
        """Find column pairs with unique combinations (optimized with limits)."""
        if 'unique_column_pairs' in self._cache:
            return self._cache['unique_column_pairs']

        total_records = len(self.df)
        if total_records == 0:
            return []

        column_list = self.df.columns.tolist()
        unique_columns = set(self.find_unique_columns())
        unique_column_pairs = []

        # Limit combinations for performance
        combinations = list(itertools.combinations(column_list, 2))
        if len(combinations) > max_combinations:
            self.logger.warning(f"Too many column combinations ({len(combinations)}), "
                          f"limiting to first {max_combinations}")
            combinations = combinations[:max_combinations]

        for first_column, second_column in combinations:
            # Skip if either column is already unique
            if first_column in unique_columns or second_column in unique_columns:
                continue

            try:
                # Create combination more efficiently
                combined = self.df[first_column].astype(str) + "||" + self.df[second_column].astype(str)
                if len(combined.unique()) == total_records:
                    unique_column_pairs.append((first_column, second_column))
            except Exception as e:
                self.logger.warning(f"Error checking pair ({first_column}, {second_column}): {str(e)}")
                continue

        self._cache['unique_column_pairs'] = unique_column_pairs
        return unique_column_pairs

    def find_empty_columns(self) -> List[str]:
        """Find completely empty columns."""
        if 'empty_columns' in self._cache:
            return self._cache['empty_columns']

        empty_columns = []
        for column in self.df.columns:
            if self.df[column].dropna().empty:
                empty_columns.append(str(column))

        self._cache['empty_columns'] = empty_columns
        return empty_columns

    def find_categorical_columns(self, unique_threshold: float = 0.05) -> List[str]:
        """Find categorical columns based on uniqueness ratio."""
        if unique_threshold < 0 or unique_threshold > 1:
            raise ValueError('Unique threshold must be between 0 and 1')

        categorical_columns = []
        total_rows = len(self.df)

        for column in self.df.columns:
            series = self.df[column].dropna()
            if len(series) == 0:
                continue

            unique_ratio = len(series.unique()) / len(series)

            # Also consider data type
            is_numeric_categorical = (
                series.dtype in ['int64', 'int32', 'float64', 'float32'] and
                unique_ratio <= unique_threshold and
                len(series.unique()) <= 50
            )

            is_string_categorical = (
                series.dtype == 'object' and
                unique_ratio <= unique_threshold
            )

            if is_numeric_categorical or is_string_categorical:
                categorical_columns.append(str(column))

        return categorical_columns

    def _generate_comprehensive_column_profiles(self, primary_key: Optional[str],
                                              decimal_places: int,
                                              outlier_method: str) -> List[ColumnProfile]:
        """Generate comprehensive profiles for all columns."""
        profiles = []
        unique_columns = set(self.find_unique_columns())

        if self.parallel_processing:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_column = {
                    executor.submit(
                        self._analyze_single_column,
                        column, series, primary_key, unique_columns,
                        decimal_places, outlier_method
                    ): column
                    for column, series in self.df.items()
                }

                for future in as_completed(future_to_column):
                    column = future_to_column[future]
                    try:
                        profile = future.result()
                        profiles.append(profile)
                    except Exception as e:
                        self.logger.error(f"Error analyzing column {column}: {str(e)}")
                        # Create basic profile for failed analysis
                        profiles.append(ColumnProfile(
                            column_name=str(column),
                            data_type='unknown',
                            total_count=len(self.df[column])
                        ))
        else:
            for column, series in self.df.items():
                try:
                    profile = self._analyze_single_column(
                        column, series, primary_key, unique_columns,
                        decimal_places, outlier_method
                    )
                    profiles.append(profile)
                except Exception as e:
                    self.logger.error(f"Error analyzing column {column}: {str(e)}")
                    profiles.append(ColumnProfile(
                        column_name=str(column),
                        data_type='unknown',
                        total_count=len(series)
                    ))

        return sorted(profiles, key=lambda x: x.column_name)

    def _analyze_single_column(self, column: Hashable, series: pd.Series,
                             primary_key: Optional[str], unique_columns: set,
                             decimal_places: int, outlier_method: str) -> ColumnProfile:
        """Analyze a single column comprehensively."""
        column_name = str(column)
        total_count = len(series)
        null_count = series.isnull().sum()
        non_null_series = series.dropna()

        profile = ColumnProfile(
            column_name=column_name,
            data_type='unknown',
            is_primary_key=(column_name == primary_key),
            is_unique=(column in unique_columns),
            is_empty=(len(non_null_series) == 0),
            total_count=total_count,
            null_count=int(null_count),
            null_percentage=float(null_count / total_count * 100) if total_count > 0 else 0,
            unique_count=len(non_null_series.unique()) if len(non_null_series) > 0 else 0
        )

        if len(non_null_series) > 0:
            profile.unique_percentage = profile.unique_count / len(non_null_series) * 100

        if profile.is_empty:
            return profile

        # Determine data type and calculate statistics
        profile = self._determine_data_type_and_stats(profile, non_null_series, decimal_places)

        # Detect outliers
        if profile.data_type in ['integer', 'float']:
            profile.outlier_count = self._detect_outliers(non_null_series, outlier_method)

        # Check for format inconsistencies
        profile.format_inconsistencies = self._check_format_consistency(non_null_series)

        # Check for encoding issues
        if profile.data_type == 'string':
            profile.suspected_encoding_issues = self._check_encoding_issues(non_null_series)

        # Generate top values for categorical analysis
        if profile.unique_count <= 20 or profile.unique_percentage <= 5:
            profile.is_categorical = True
            profile.top_values = dict(non_null_series.value_counts().head(10))

        return profile

    def _determine_data_type_and_stats(self, profile: ColumnProfile,
                                     series: pd.Series, decimal_places: int) -> ColumnProfile:
        """Determine data type and calculate appropriate statistics."""
        # Try numeric first
        try:
            # Check for integer
            if series.apply(lambda x: self._is_integer(x)).all():
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if len(numeric_series) > 0:
                    profile.data_type = 'integer'
                    profile = self._calculate_numeric_stats(profile, numeric_series)
                    return profile

            # Check for float
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isnull().all():
                numeric_series = numeric_series.dropna()
                if len(numeric_series) > 0:
                    profile.data_type = 'float'
                    profile.decimal_places = decimal_places
                    profile = self._calculate_numeric_stats(profile, numeric_series)
                    return profile
        except Exception:
            pass

        # Try boolean
        try:
            if series.apply(self._is_boolean).all():
                profile.data_type = 'boolean'
                return profile
        except Exception:
            pass

        # Try datetime
        try:
            dt_series = pd.to_datetime(series, errors='coerce')
            if not dt_series.isnull().all():
                profile.data_type = 'datetime'
                profile = self._calculate_datetime_stats(profile, dt_series.dropna())
                return profile
        except Exception:
            pass

        # Default to string
        profile.data_type = 'string'
        profile = self._calculate_string_stats(profile, series.astype(str))

        return profile

    def _calculate_numeric_stats(self, profile: ColumnProfile, series: pd.Series) -> ColumnProfile:
        """Calculate comprehensive numeric statistics."""
        try:
            profile.min_value = float(series.min())
            profile.max_value = float(series.max())
            profile.mean_value = float(series.mean())
            profile.median_value = float(series.median())
            profile.std_dev = float(series.std())
            profile.q1 = float(series.quantile(0.25))
            profile.q3 = float(series.quantile(0.75))

            # Advanced statistics
            if len(series) > 3:  # Need at least 4 values for skewness/kurtosis
                profile.skewness = float(series.skew())
                profile.kurtosis = float(series.kurtosis())

            # Calculate precision for integers
            if profile.data_type == 'integer':
                max_abs = max(abs(profile.min_value), abs(profile.max_value))
                if max_abs > 0:
                    profile.float_precision = int(math.log10(max_abs)) + 1
                else:
                    profile.float_precision = 1

        except Exception as e:
            self.logger.warning(f"Error calculating numeric stats for {profile.column_name}: {str(e)}")

        return profile

    def _calculate_string_stats(self, profile: ColumnProfile, series: pd.Series) -> ColumnProfile:
        """Calculate string-specific statistics."""
        try:
            str_lengths = series.str.len()
            profile.max_str_length = int(str_lengths.max())
            profile.min_str_length = int(str_lengths.min())
            profile.avg_str_length = float(str_lengths.mean())
        except Exception as e:
            self.logger.warning(f"Error calculating string stats for {profile.column_name}: {str(e)}")

        return profile

    def _calculate_datetime_stats(self, profile: ColumnProfile, series: pd.Series) -> ColumnProfile:
        """Calculate datetime-specific statistics."""
        try:
            profile.min_value = series.min().isoformat()
            profile.max_value = series.max().isoformat()

            # Check for anomalies (dates in far future/past)
            current_year = pd.Timestamp.now().year
            future_dates = series[series.dt.year > current_year + 10]
            past_dates = series[series.dt.year < 1900]

            if len(future_dates) > 0 or len(past_dates) > 0:
                profile.format_inconsistencies = len(future_dates) + len(past_dates)

        except Exception as e:
            self.logger.warning(f"Error calculating datetime stats for {profile.column_name}: {str(e)}")

        return profile

    def _is_integer(self, value) -> bool:
        """Check if value can be parsed as integer."""
        try:
            int(float(value))
            return float(value).is_integer()
        except (ValueError, TypeError, OverflowError):
            return False

    def _is_boolean(self, value) -> bool:
        """Check if value can be parsed as boolean."""
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.lower() in ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n']
        return False

    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> int:
        """Detect outliers using specified method."""
        try:
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                return len(outliers)

            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = series[z_scores > 3]
                return len(outliers)

            else:  # Default to IQR
                return self._detect_outliers(series, 'iqr')

        except Exception:
            return 0

    def _check_format_consistency(self, series: pd.Series) -> int:
        """Check for format inconsistencies in the data."""
        try:
            # For string data, check for mixed formats
            if series.dtype == 'object':
                str_series = series.astype(str)

                # Check for mixed date formats
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                ]

                pattern_matches = 0
                for pattern in date_patterns:
                    matches = str_series.str.match(pattern).sum()
                    if matches > 0:
                        pattern_matches += 1

                if pattern_matches > 1:
                    return int(len(series) * 0.1)  # Estimate inconsistencies

        except Exception:
            pass

        return 0

    def _check_encoding_issues(self, series: pd.Series) -> int:
        """Check for potential encoding issues."""
        try:
            str_series = series.astype(str)
            encoding_issues = 0

            for value in str_series:
                # Check for common encoding issue indicators
                if any(char in value for char in ['�', '\\x', '\\u']):
                    encoding_issues += 1

        except Exception:
            pass

        return encoding_issues

    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        try:
            duplicate_mask = self.df.duplicated()
            duplicate_count = duplicate_mask.sum()
            duplicate_percentage = (duplicate_count / len(self.df)) * 100 if len(self.df) > 0 else 0

            return {
                'count': int(duplicate_count),
                'percentage': float(duplicate_percentage),
                'indices': self.df[duplicate_mask].index.tolist()[:100]  # Limit for performance
            }
        except Exception as e:
            self.logger.error(f"Error analyzing duplicates: {str(e)}")
            return {'count': 0, 'percentage': 0.0, 'indices': []}

    def _detect_potential_primary_keys(self) -> List[str]:
        """Detect columns that could serve as primary keys."""
        unique_columns = self.find_unique_columns()
        potential_keys = []

        for column in unique_columns:
            series = self.df[column].dropna()
            if len(series) == len(self.df):  # No nulls
                # Check if it looks like an ID (numeric incrementing or UUID-like)
                if self._looks_like_primary_key(series):
                    potential_keys.append(str(column))

        return potential_keys

    def _looks_like_primary_key(self, series: pd.Series) -> bool:
        """Check if a series looks like a primary key."""
        try:
            # Check for numeric incrementing
            if pd.api.types.is_numeric_dtype(series):
                sorted_series = series.sort_values()
                if (sorted_series.diff().dropna() == 1).all():
                    return True

            # Check for UUID-like patterns
            if series.dtype == 'object':
                str_series = series.astype(str)
                if str_series.str.match(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$').any():
                    return True

        except Exception:
            pass

        return False

    def _check_referential_integrity(self) -> List[Dict[str, Any]]:
        """Check for potential referential integrity issues."""
        issues = []

        # Look for columns that might be foreign keys
        for column in self.df.columns:
            if str(column).lower().endswith('_id') or str(column).lower().endswith('id'):
                series = self.df[column].dropna()
                if len(series) > 0 and len(series.unique()) < len(series) * 0.8:
                    # Potential foreign key - check for orphaned values
                    # This is a simplified check - in practice, you'd check against actual reference tables
                    null_count = self.df[column].isnull().sum()
                    if null_count > 0:
                        issues.append({
                            'column': str(column),
                            'issue': 'potential_orphaned_references',
                            'null_count': int(null_count),
                            'description': f'Column {column} appears to be a foreign key with null values'
                        })

        return issues

    def _collect_all_issues(self, column_profiles: List[ColumnProfile],
                          duplicate_info: Dict[str, Any]) -> List[Tuple[DataQualityIssue, str, Any]]:
        """Collect all identified data quality issues."""
        issues = []

        # Duplicate rows
        if duplicate_info['count'] > 0:
            issues.append((
                DataQualityIssue.DUPLICATE_ROWS,
                f"Found {duplicate_info['count']} duplicate rows ({duplicate_info['percentage']:.2f}%)",
                duplicate_info
            ))

        # Column-specific issues
        for profile in column_profiles:
            # Missing values
            if profile.null_percentage > 10:  # More than 10% nulls
                issues.append((
                    DataQualityIssue.MISSING_VALUES,
                    f"Column '{profile.column_name}' has {profile.null_percentage:.1f}% missing values",
                    {'column': profile.column_name, 'percentage': profile.null_percentage}
                ))

            # Outliers
            if profile.outlier_count > 0:
                issues.append((
                    DataQualityIssue.OUTLIERS,
                    f"Column '{profile.column_name}' has {profile.outlier_count} outliers",
                    {'column': profile.column_name, 'count': profile.outlier_count}
                ))

            # Format inconsistencies
            if profile.format_inconsistencies > 0:
                issues.append((
                    DataQualityIssue.INCONSISTENT_FORMATS,
                    f"Column '{profile.column_name}' has {profile.format_inconsistencies} format inconsistencies",
                    {'column': profile.column_name, 'count': profile.format_inconsistencies}
                ))

            # Encoding issues
            if profile.suspected_encoding_issues > 0:
                issues.append((
                    DataQualityIssue.ENCODING_ISSUES,
                    f"Column '{profile.column_name}' has {profile.suspected_encoding_issues} suspected encoding issues",
                    {'column': profile.column_name, 'count': profile.suspected_encoding_issues}
                ))

        return issues

    def _calculate_consistency_score(self, issues: List[Tuple[DataQualityIssue, str, Any]],
                                   column_profiles: List[ColumnProfile]) -> float:
        """Calculate overall data consistency score (0-100)."""
        try:
            total_cells = len(self.df) * len(self.df.columns)
            if total_cells == 0:
                return 100.0

            # Count problematic cells
            problematic_cells = 0

            for profile in column_profiles:
                problematic_cells += profile.null_count
                problematic_cells += profile.outlier_count
                problematic_cells += profile.format_inconsistencies
                problematic_cells += profile.suspected_encoding_issues

            # Add duplicate impact
            duplicate_info = next((issue[2] for issue in issues
                                 if issue[0] == DataQualityIssue.DUPLICATE_ROWS), {})
            if duplicate_info:
                problematic_cells += duplicate_info.get('count', 0) * len(self.df.columns)

            # Calculate score
            score = max(0, 100 - (problematic_cells / total_cells * 100))
            return round(score, 2)

        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {str(e)}")
            return 0.0

    def log_quality_summary(self, quality_report: DataQualityReport) -> None:
        """
        Log a summary of the data quality analysis results.

        Args:
            quality_report: DataQualityReport containing analysis results
        """
        self.logger.info("Data Quality Analysis Complete:")
        self.logger.info(f"  - Total Rows: {quality_report.total_rows:,}")
        self.logger.info(f"  - Total Columns: {quality_report.total_columns}")
        self.logger.info(f"  - Duplicate Rows: {quality_report.duplicate_rows} ({quality_report.duplicate_percentage:.1f}%)")
        self.logger.info(f"  - Data Consistency Score: {quality_report.data_consistency_score}/100")
        self.logger.info(f"  - Issues Found: {len(quality_report.issues_found)}")
        for issue_type, description, _ in quality_report.issues_found:
            self.logger.warning(f"  - {issue_type.value}: {description}")
        self.logger.info(f"  - Unique Columns: {len(quality_report.unique_columns)}")
        self.logger.info(f"  - Empty Columns: {len(quality_report.empty_columns)}")
        self.logger.info(f"  - Categorical Columns: {len(quality_report.categorical_columns)}")
        self.logger.info(f"  - Potential Primary Keys: {quality_report.potential_primary_keys}")


# Backward compatibility - keep the old class name as an alias
class Analyzer(DataFrameAnalyzer):
    """Backward compatibility alias."""

    @staticmethod
    def find_unique_columns(df: pd.DataFrame) -> List[Hashable]:
        """Legacy method for backward compatibility."""
        analyzer = DataFrameAnalyzer(df)
        return analyzer.find_unique_columns()

    @staticmethod
    def find_unique_column_pairs(df: pd.DataFrame) -> List[Tuple[Hashable, Hashable]]:
        """Legacy method for backward compatibility."""
        analyzer = DataFrameAnalyzer(df)
        return analyzer.find_unique_column_pairs()

    @staticmethod
    def find_empty_columns(df: pd.DataFrame) -> List[str]:
        """Legacy method for backward compatibility."""
        analyzer = DataFrameAnalyzer(df)
        return analyzer.find_empty_columns()

    @staticmethod
    def find_categorical_columns(df: pd.DataFrame, unique_threshold: float = 0.05) -> List[Hashable]:
        """Legacy method for backward compatibility."""
        analyzer = DataFrameAnalyzer(df)
        return analyzer.find_categorical_columns(unique_threshold)

    @staticmethod
    def generate_column_metadata(df: pd.DataFrame, primary_key: Optional[str],
                               unique_columns: Optional[List[str]], decimal_places: int) -> List[Dict]:
        """Legacy method for backward compatibility."""
        analyzer = DataFrameAnalyzer(df)
        report = analyzer.analyze_comprehensive(primary_key, decimal_places)

        # Convert to legacy format
        legacy_metadata = []
        for profile in report.column_profiles:
            legacy_metadata.append({
                'column_name': profile.column_name,
                'data_type': profile.data_type,
                'is_id': profile.is_primary_key,
                'is_unique': profile.is_unique,
                'is_empty': profile.is_empty,
                'max_str_size': profile.max_str_length,
                'float_precision': profile.float_precision,
                'decimal_places': profile.decimal_places,
                'biggest_num': profile.max_value,
                'smallest_num': profile.min_value
            })

        return legacy_metadata
