"""
MCP Server for Data Insights
Provides three tools: ingest_excel, query_data, generate_insights
"""
import base64
import sqlite3
import pandas as pd
import json
from io import BytesIO
from mcp.server.fastmcp import FastMCP
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
)

# Initialize FastMCP server
# mcp = FastMCP("DataInsightServer")

mcp = FastMCP(
    name="DataInsightServer",
    port=8001,
    host="127.0.0.1",
    log_level="INFO",
    # warn_on_duplicate_tools=True
)
print(mcp.settings.port)

# Database configuration
DB_PATH = "data.db"

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_catalog_table():
    """Initialize catalog table to track ingested files"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS catalog (
            file_id TEXT PRIMARY KEY,
            file_name TEXT,
            sheets_count INTEGER,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize catalog on startup
init_catalog_table()


@mcp.resource("db://schema")
def get_schema() -> Dict[str, Any]:
    """
    MCP Resource: Returns database schema information
    Lists all tables and their columns for LLM awareness
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all table names except catalog
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'catalog';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_info = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
        schema_info[table] = columns

    conn.close()

    return {
        "tables": list(schema_info.keys()),
        "schema": schema_info,
        "total_tables": len(schema_info)
    }


@mcp.tool()
def ingest_excel(file_name: str, base64_content: str) -> Dict[str, Any]:
    """
    Tool 1: Ingest Excel file and store in SQLite database

    Accepts base64-encoded Excel file, reads all sheets with pandas,
    generates metadata catalogue, and stores data in SQLite tables.
    Each sheet becomes a table named: {file_id}_{sheet_name}

    Args:
        file_name: Name of the Excel file
        base64_content: Base64-encoded content of the Excel file

    Returns:
        Dictionary with metadata and table names
    """
    try:
        # Decode base64 content
        file_data = base64.b64decode(base64_content)
        excel_file = BytesIO(file_data)

        # Read all sheets from Excel
        sheets_dict = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')

        # Generate file_id (sanitized file name)
        file_id = file_name.replace('.xlsx', '').replace('.xls', '').replace(' ', '_').replace('-', '_')
        file_id = ''.join(c for c in file_id if c.isalnum() or c == '_')[:20]

        conn = get_db_connection()
        cursor = conn.cursor()

        catalog_metadata = []
        table_names = []

        # Process each sheet
        for sheet_name, df in sheets_dict.items():
            # Sanitize sheet name for table
            sanitized_sheet = sheet_name.replace(' ', '_').replace('-', '_')
            sanitized_sheet = ''.join(c for c in sanitized_sheet if c.isalnum() or c == '_')
            table_name = f"{file_id}_{sanitized_sheet}"

            # Store data in SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            # Generate metadata for this sheet
            metadata_entry = {
                "table_name": table_name,
                "original_sheet": sheet_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
            }
            catalog_metadata.append(metadata_entry)
            table_names.append(table_name)

        # Store catalog entry
        cursor.execute("""
            INSERT OR REPLACE INTO catalog (file_id, file_name, sheets_count, metadata)
            VALUES (?, ?, ?, ?)
        """, (file_id, file_name, len(sheets_dict), json.dumps(catalog_metadata)))

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "file_id": file_id,
            "file_name": file_name,
            "sheets_ingested": len(sheets_dict),
            "table_names": table_names,
            "metadata": catalog_metadata,
            "message": f"Successfully ingested {len(sheets_dict)} sheet(s) from {file_name}"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to ingest Excel file: {str(e)}"
        }


@mcp.tool()
def query_data(nl_query: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Tool 2: Natural language to SQL query with entity resolution

    Converts natural language query to SQL, executes it with multiple retry attempts,
    performs entity resolution to map user terms to actual table/column names.

    Args:
        nl_query: Natural language query from user
        max_attempts: Maximum number of SQL generation attempts (default: 3)

    Returns:
        Query results as list of records
    """
    try:
        # Get current database schema
        schema = get_schema()

        if not schema.get("tables") or len(schema["tables"]) == 0:
            return {
                "status": "error",
                "message": "No data ingested yet. Please upload an Excel file first."
            }

        # Prepare schema context for LLM
        schema_context = "Database Schema:\n"
        for table, columns in schema["schema"].items():
            schema_context += f"\nTable: {table}\n"
            schema_context += "Columns: " + ", ".join([f"{col['name']} ({col['type']})" for col in columns]) + "\n"

        # Attempt SQL generation with retries
        for attempt in range(max_attempts):
            try:
                # Generate SQL using Azure OpenAI
                system_prompt = """You are a SQL expert. Generate safe, read-only SELECT SQL queries based on user questions.

Rules:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Use proper SQLite syntax
3. Handle entity resolution: map user terms to actual column names
4. Include appropriate WHERE, JOIN, GROUP BY, ORDER BY clauses as needed
5. Return ONLY the SQL query, no explanations or markdown
6. If the query involves aggregations, use appropriate functions (SUM, AVG, COUNT, etc.)
7. For text matching, use LIKE with wildcards when appropriate
8. Always limit results to reasonable amounts (add LIMIT clause if not specified)"""

                error_context = ""
                if attempt > 0:
                    error_context = f"\n\nPrevious attempt failed. Please fix the SQL query."

                user_prompt = f"""{schema_context}

User Question: {nl_query}{error_context}

Generate the SQL query:"""

                response = azure_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )

                sql_query = response.choices[0].message.content.strip()

                # Clean up SQL query (remove markdown, extra whitespace)
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

                # Security check: ensure it's a SELECT query
                if not sql_query.upper().startswith("SELECT"):
                    return {
                        "status": "error",
                        "message": "Only SELECT queries are allowed for security reasons."
                    }

                # Execute SQL query
                conn = get_db_connection()
                df = pd.read_sql_query(sql_query, conn)
                conn.close()

                # Convert to records format
                results = df.to_dict('records')

                return {
                    "status": "success",
                    "query": nl_query,
                    "sql": sql_query,
                    "results": results,
                    "row_count": len(results),
                    "columns": list(df.columns),
                    "attempt": attempt + 1
                }

            except Exception as query_error:
                if attempt == max_attempts - 1:
                    # Last attempt failed
                    return {
                        "status": "error",
                        "message": f"Failed to execute query after {max_attempts} attempts: {str(query_error)}",
                        "last_sql": sql_query if 'sql_query' in locals() else None
                    }
                # Continue to next attempt
                continue

    except Exception as e:
        return {
            "status": "error",
            "message": f"Query processing error: {str(e)}"
        }


@mcp.tool()
def generate_insights(tables: Optional[List[str]] = None, spec: str = "") -> Dict[str, Any]:
    """
    Tool 3: Generate statistical and ML-based insights from data

    Analyzes specified tables using statistical techniques and basic ML.
    Provides descriptive statistics, correlations, outlier detection, and trends.

    Args:
        tables: List of table names to analyze (if None, analyzes all tables)
        spec: Optional specification for targeted insights (e.g., "correlations", "outliers", "trends")

    Returns:
        Dictionary with comprehensive insights
    """
    try:
        print("Starting insights generation...")
        conn = get_db_connection()

        # Get tables to analyze
        if tables is None or len(tables) == 0:
            schema = get_schema()
            tables = schema.get("tables", [])

        if not tables:
            return {
                "status": "error",
                "message": "No tables available for analysis. Please ingest data first."
            }

        all_insights = []

        # log the spec
        print(f"Generating insights with spec: {spec}")

        for table in tables:
            print(f"Analyzing table: {table}")
            try:
                # Read table data
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

                if df.empty:
                    continue

                print(f"Table {table} has {len(df)} rows and {len(df.columns)} columns.")
                table_insights = {
                    "table_name": table,
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }

                # Descriptive statistics for numerical columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    desc_stats = df[numeric_cols].describe().to_dict()
                    table_insights["descriptive_statistics"] = desc_stats

                    # Calculate additional statistics
                    table_insights["additional_stats"] = {}
                    for col in numeric_cols:
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            table_insights["additional_stats"][col] = {
                                "median": float(col_data.median()),
                                "mode": float(col_data.mode()[0]) if len(col_data.mode()) > 0 else None,
                                "variance": float(col_data.var()),
                                "skewness": float(stats.skew(col_data)),
                                "kurtosis": float(stats.kurtosis(col_data))
                            }

                # Correlations (if requested or if multiple numeric columns exist)
                if (spec.lower() == "correlations" or "corr" in spec.lower()) and len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()

                    # Find strong correlations
                    strong_correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            corr_value = corr_matrix.iloc[i, j]
                            if abs(corr_value) > 0.5:  # Threshold for "strong"
                                strong_correlations.append({
                                    "column1": corr_matrix.columns[i],
                                    "column2": corr_matrix.columns[j],
                                    "correlation": float(corr_value),
                                    "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                                })

                    table_insights["correlations"] = {
                        "matrix": corr_matrix.to_dict(),
                        "strong_correlations": strong_correlations
                    }

                # Outlier detection using IQR method
                if spec.lower() == "outliers" or "outlier" in spec.lower():
                    outliers_info = {}
                    for col in numeric_cols:
                        col_data = df[col].dropna()
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        if len(outliers) > 0:
                            outliers_info[col] = {
                                "count": len(outliers),
                                "percentage": round(len(outliers) / len(col_data) * 100, 2),
                                "lower_bound": float(lower_bound),
                                "upper_bound": float(upper_bound),
                                "outlier_values": outliers.tolist()[:10]  # Limit to first 10
                            }

                    if outliers_info:
                        table_insights["outliers"] = outliers_info

                # Missing data analysis
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    table_insights["missing_data"] = {
                        col: {
                            "count": int(count),
                            "percentage": round(count / len(df) * 100, 2)
                        }
                        for col, count in missing_data.items() if count > 0
                    }

                # Data type distribution
                table_insights["data_types"] = {
                    "numeric_columns": numeric_cols,
                    "text_columns": df.select_dtypes(include=['object']).columns.tolist(),
                    "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
                }

                # Categorical analysis for text columns
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    categorical_insights = {}
                    for col in text_cols[:5]:  # Limit to first 5 text columns
                        unique_count = df[col].nunique()
                        if unique_count < 50:  # Only for low-cardinality columns
                            value_counts = df[col].value_counts().head(10)
                            categorical_insights[col] = {
                                "unique_values": unique_count,
                                "top_values": value_counts.to_dict()
                            }

                    if categorical_insights:
                        table_insights["categorical_analysis"] = categorical_insights

                all_insights.append(table_insights)

            except Exception as table_error:
                all_insights.append({
                    "table_name": table,
                    "error": str(table_error)
                })

        conn.close()

        # Generate summary insights using LLM
        try:
            summary_prompt = f"""Analyze these data insights and provide a brief executive summary (3-5 key points):

{json.dumps(all_insights, indent=2, default=str)}

Focus on the most interesting patterns, anomalies, or actionable insights."""

            summary_response = azure_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": "You are a data analyst providing concise, actionable insights."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            executive_summary = summary_response.choices[0].message.content

        except Exception as summary_error:
            executive_summary = f"Could not generate summary: {str(summary_error)}"

        return {
            "status": "success",
            "tables_analyzed": len(all_insights),
            "specification": spec if spec else "general",
            "insights": all_insights,
            "executive_summary": executive_summary
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Insights generation error: {str(e)}"
        }



if __name__ == "__main__":
    # If your version ignores constructor host/port, force it:
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = 8001

    # choose transport explicitly
    mcp.run(transport="sse")              # SSE endpoint typically /sse
    # OR
    # mcp.run(transport="streamable-http")  # endpoint typically /mcp
