# Data Insight Platform - MCP Application

A Model Context Protocol (MCP) based application for Excel data analysis with natural language queries and AI-powered insights.

## Features

- **Excel Ingestion**: Upload and process Excel files with automatic metadata extraction
- **Natural Language Queries**: Ask questions about your data in plain English
- **AI-Powered Insights**: Generate statistical analysis, correlations, outlier detection, and trends
- **Enterprise UX**: Clean, responsive HTML5 interface with real-time updates
- **Azure OpenAI Integration**: Powered by GPT-4 for intelligent text-to-SQL and insights generation

## Architecture

### MCP Server (`mcp_server.py`)
Single MCP server exposing three tools:
1. **ingest_excel** - Process Excel files and store in SQLite
2. **query_data** - Natural language to SQL with entity resolution
3. **generate_insights** - Statistical and ML-based data analysis

Plus one resource:
- **db://schema** - Database schema information for LLM context

### MCP Client (`client.py`)
FastAPI application serving HTML5 frontend and managing MCP communication.

### Database
SQLite in-memory database (`data.db`) with dynamic table generation.

## Prerequisites

- Python 3.10 or higher
- Azure OpenAI account with API access
- Windows operating system

## Setup Instructions

### 1. Install Python Dependencies

Open Command Prompt or PowerShell in the project directory and run:

```bash
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

Copy the example environment file and configure your Azure OpenAI credentials:

```bash
copy .env.example .env
```

Edit `.env` file with your Azure OpenAI details:

```
AZURE_OPENAI_API_KEY=your_actual_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

**Where to find these values:**
- Go to [Azure Portal](https://portal.azure.com)
- Navigate to your Azure OpenAI resource
- Find `Keys and Endpoint` in the left sidebar
- Copy `KEY 1` as your API key
- Copy the `Endpoint` URL
- Use your deployment name (e.g., `gpt-4o`, `gpt-35-turbo`)

### 3. Start the MCP Server

Open a terminal window and run:

```bash
python mcp_server.py
```

You should see:
```
Starting MCP Data Insight Server on http://127.0.0.1:8001
Available tools: ingest_excel, query_data, generate_insights
Available resources: db://schema
```

**Keep this terminal window open** - the MCP server must be running.

### 4. Start the MCP Client

Open a **second terminal window** (new Command Prompt or PowerShell) and run:

```bash
python client.py
```

You should see:
```
Starting MCP Client on http://127.0.0.1:8000
Make sure MCP server is running on http://127.0.0.1:8001
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 5. Access the Application

Open your web browser and navigate to:

```
http://localhost:8000
```

## Usage Guide

### Upload Excel File

1. Click or drag-and-drop your Excel file (.xlsx format) in the upload area
2. Click "Upload File" button
3. Wait for the success message showing ingested sheets

### Ask Natural Language Queries

Once your file is uploaded, type questions in the chat interface:

**Example queries:**
- "Show me all the data"
- "What are the top 10 sales records?"
- "Calculate the average revenue by region"
- "Find customers who spent more than $1000"
- "Show me records from 2024"

The system will:
1. Convert your question to SQL
2. Execute the query safely
3. Return formatted results

### Generate Insights

1. Select an insight type from the dropdown:
   - **General Overview**: Descriptive statistics for all numeric columns
   - **Correlations**: Find relationships between numeric columns
   - **Outlier Detection**: Identify anomalies using IQR method
   - **Trend Analysis**: General trends in your data

2. Click "Generate Insights"
3. View the executive summary and detailed analysis

## Technical Details

### MCP Protocol Compliance

This application follows the Model Context Protocol (MCP) specification:
- **Tools**: Callable functions exposed by the MCP server
- **Resources**: Contextual information (database schema) for LLM awareness
- **Transport**: Server-Sent Events (SSE) for communication

### Data Storage

- **Database**: SQLite (`data.db`)
- **Table Naming**: `{file_id}_{sheet_name}` (e.g., `sales_2024_Sheet1`)
- **Catalog Table**: Tracks ingested files and metadata
- **Persistence**: Data persists across sessions until `data.db` is deleted

### Security Features

- SQL queries are validated to ensure only SELECT statements
- No direct file system access - all Excel data passed via base64 encoding
- Parameterized queries prevent SQL injection
- Maximum retry attempts prevent infinite loops

### Error Handling

The application includes robust error handling:
- Connection failures to MCP server
- SQL generation and execution errors
- File upload validation
- Azure OpenAI API errors

## Troubleshooting

### MCP Server Connection Error

**Error**: "Could not connect to MCP server"

**Solution**:
- Ensure `mcp_server.py` is running in a separate terminal
- Check that port 8001 is not in use by another application
- Verify firewall settings allow local connections

### Azure OpenAI API Error

**Error**: "Authentication failed" or "Invalid deployment"

**Solution**:
- Verify `.env` file has correct API key and endpoint
- Check that your deployment name matches the one in Azure Portal
- Ensure your Azure OpenAI resource is active and has available quota

### Excel Upload Fails

**Error**: "Failed to ingest Excel file"

**Solution**:
- Ensure file is in .xlsx format (not .xls)
- Check file is not corrupted
- Try with a smaller file first
- Verify file contains valid Excel sheets with data

### No Results from Query

**Issue**: Query returns empty results

**Solution**:
- Verify data was uploaded successfully (check upload confirmation)
- Try simpler queries first (e.g., "show all data")
- Check column names match your query (case-insensitive)
- Review the generated SQL in the response

## File Structure

```
autodash/
├── mcp_server.py          # MCP server with 3 tools + 1 resource
├── client.py              # FastAPI client with embedded HTML5 UI
├── requirements.txt       # Python dependencies
├── .env.example          # Template for environment variables
├── .env                  # Your actual config (not in git)
├── data.db               # SQLite database (created on first run)
└── README.md             # This file
```

## Limitations

- **In-Memory Database**: Data persists in `data.db` but resetting requires deleting the file
- **No Authentication**: This is a local development application
- **Single User**: Not designed for concurrent multi-user access
- **No CI/CD**: Manual deployment only
- **Excel Only**: Only .xlsx format supported

## Dependencies

Key libraries used:
- `fastmcp` - MCP server framework
- `fastapi` - Web framework for client
- `pandas` - Excel reading and data manipulation
- `openpyxl` - Excel file processing
- `openai` - Azure OpenAI SDK
- `scikit-learn` - ML features for insights
- `scipy` - Statistical analysis

See [requirements.txt](requirements.txt) for complete list.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all prerequisites are met
3. Review terminal output for error messages
4. Ensure `.env` configuration is correct

## License

Internal use only. Enterprise application for data analysis.

## Version

1.0.0 - Initial release with MCP protocol compliance
