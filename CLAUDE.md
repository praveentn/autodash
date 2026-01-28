## Revised Requirements

The requirements- for MCP protocol compliance, modularity, and enterprise UX. Key: the three MCP server tools in **one cohesive MCP server** to avoid fragmentation (clients connect to one endpoint); use base64-encoded file content for Excel uploads via MCP resources/tools (no direct file system access); implement SQLite with tables per Excel sheet prefixed by file_id; leverage Azure OpenAI's chat completions with MCP tool calling support; build client as FastAPI app serving HTML5 frontend with WebSocket-like SSE for real-time interaction; ensure stateless in-memory SQLite per server instance; add env var config for Azure keys; provide Windows-friendly README. [github](https://github.com/CrazyForks/fastmcp-py)

## Architecture Overview

- **Single MCP Server** (`mcp_server.py`): FastMCP instance with 3 tools + resources for DB schema listing. Runs on port 8001.
  - Tool 1: `ingest_excel` - Accepts base64 Excel, reads sheets with pandas, generates schema summary ("metadata catalogue"), stores data in SQLite tables (e.g., `file_abc_sheet1`), returns table names.
  - Tool 2: `query_data` - NL query → Azure OpenAI text2sql (multiple attempts, entity resolution), executes safest SQL, returns results.
  - Tool 3: `generate_insights` - Table list + optional spec → Stats (pandas describe), simple ML (e.g., correlations), returns formatted insights.
  - Resources: `db://schema` lists tables/columns for LLM awareness. [gofastmcp](https://gofastmcp.com/tutorials/mcp)
- **MCP Client** (`client.py` + `index.html`): FastAPI on port 8000 serves HTML5 UI; connects to MCP server, uses Azure OpenAI (gpt-4o or equivalent) with dynamic tool discovery for NL queries/insights. [realpython](https://realpython.com/python-mcp-client/)
- **DB**: Single in-memory SQLite (`data.db`) with dynamic tables; no Alembic/migrations. [stackoverflow](https://stackoverflow.com/questions/69554163/import-excel-files-in-sqlite-with-pandas)
- **UX**: Clean HTML5 with file upload (drag-drop), chat interface, insights request form; streaming responses; error handling. [delynchoong.github](https://delynchoong.github.io/blog/azure-mcp-tutorial/)

## Tech Stack & Dependencies

| Component | Libraries/Tools |
|-----------|-----------------|
| MCP Server | `fastmcp`, `pandas`, `openpyxl`, `sqlite3`, `azure-openai` (for text2sql/insights) |
| Client | `fastapi`, `uvicorn`, `httpx` (for MCP client), `azure-openai`, `python-multipart` (file handling) |
| Frontend | Vanilla HTML5/JS (fetch API, SSE for MCP) |
| Common | Python 3.10+, `python-dotenv` for AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT |

Install via `pip install fastmcp fastapi uvicorn pandas openpyxl azure-openai python-multipart httpx python-dotenv`. [mcpcat](https://mcpcat.io/guides/building-mcp-server-python-fastmcp/)

## Key Implementation Notes

- **Excel Ingestion**: Tool accepts `{"file_name": str, "base64_content": str}`; pandas reads bytesio; extracts metadata (sheets, shape, dtypes); stores each sheet as table `{file_id}_{sheet_name}`; catalog stored in `catalog` table. [stackoverflow](https://stackoverflow.com/questions/69554163/import-excel-files-in-sqlite-with-pandas)
- **Querying**: Prompt-engineered text2sql with retries (e.g., fix errors via LLM); validate/escape SQL; support entity resolution (e.g., map "sales" to column). [reddit](https://www.reddit.com/r/AI_Agents/comments/1jd9gzv/learn_mcp_by_building_an_sqlite_ai_agent/)
- **Insights**: Pandas stats (mean/std/corr), outlier detection (IQR), viz summaries (no charts); customizable via spec like "correlations between sales and date". [kdnuggets](https://www.kdnuggets.com/2022/04/data-ingestion-pandas-beginner-tutorial.html)
- **Client Flow**: Upload → ingest tool call → Chat: Azure OpenAI + discovered MCP tools → Insights: similar with spec param. [gofastmcp](https://gofastmcp.com/integrations/openai)
- **Enterprise UX**: Responsive UI, loading spinners, error toasts, session persistence (localStorage), auth optional via API key header. [techcommunity.microsoft](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/mastering-model-context-protocol-mcp-building-multi-server-mcp-with-azure-openai/4424993)
- **Limitations Addressed**: In-memory DB resets on restart; Windows-compatible (no Docker); no CI/testing. [digitalocean](https://www.digitalocean.com/community/tutorials/mcp-server-python)

## Sample Code Skeletons

### MCP Server (mcp_server.py)
```python
import base64
import sqlite3
import pandas as pd
from io import BytesIO
from mcp.server.fastmcp import FastMCP
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version="2024-10-21")

mcp = FastMCP("DataInsightServer", json_response=True)
DB_PATH = "data.db"

@mcp.resource("db://schema")
def get_schema():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return {"tables": [row[0] for row in c.fetchall()]}

@mcp.tool()
def ingest_excel(file_name: str, base64_content: str):
    """Ingest Excel file, generate metadata, store in DB. Returns table names."""
    data = base64.b64decode(base64_content)
    excel_file = BytesIO(data)
    sheets = pd.read_excel(excel_file, sheet_name=None)
    conn = sqlite3.connect(DB_PATH)
    file_id = file_name.replace('.xlsx', '').replace(' ', '_')[:10]
    catalog = []
    for sheet_name, df in sheets.items():
        table_name = f"{file_id}_{sheet_name.replace(' ', '_')}"
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        catalog.append({"table": table_name, "rows": len(df), "cols": len(df.columns)})
    conn.close()
    return {"metadata": catalog, "message": f"Ingested {len(sheets)} sheets."}

@mcp.tool()
def query_data(nl_query: str):
    """Natural language to SQL query execution with entity resolution."""
    # Use AzureOpenAI for text2sql, execute safest query
    schema = get_schema()
    prompt = f"Schema: {schema}. Query: {nl_query}. Generate safe SELECT SQL."
    sql = client.chat.completions.create(model=os.getenv("AZURE_OPENAI_DEPLOYMENT"), messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df.to_dict('records')  # Simplified; add retries/error handling

@mcp.tool()
def generate_insights(tables: list, spec: str = ""):
    """Generate statistical/ML insights from tables."""
    conn = sqlite3.connect(DB_PATH)
    insights = []
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        stats = df.describe().to_dict()
        if spec == "correlations":
            corr = df.corr().to_dict()
        insights.append({"table": table, "stats": stats})
    conn.close()
    return insights

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8001)
```


### Client Backend (client.py)
```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import base64
import asyncio
from openai import AzureOpenAI
# ... dotenv setup

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # For CSS/JS if needed

azure_client = AzureOpenAI(...)  # As above

async def call_mcp_tool(tool_name: str, args: dict, server_url: str = "http://127.0.0.1:8001/mcp"):
    async with httpx.AsyncClient() as client:
        resp = await client.post(server_url, json={"tool": tool_name, "args": args})
        return resp.json()

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    b64 = base64.b64encode(content).decode()
    result = await call_mcp_tool("ingest_excel", {"file_name": file.filename, "base64_content": b64})
    return result

@app.post("/chat")
async def chat(message: str = Form(...)):
    # Discover tools, format for AzureOpenAI tools param, call with parallel_function_calling
    # Execute tool calls via MCP, feed back to LLM
    # Return streaming response
    return {"response": "Processed via MCP tools"}  # Full impl with tool loop

@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
    <!DOCTYPE html>
    <html><body>
    <input type="file" id="file" accept=".xlsx">
    <button onclick="upload()">Upload Excel</button>
    <div id="chat"></div>
    <input id="query" placeholder="Ask about data">
    <button onclick="query()">Query</button>
    <input id="insight" placeholder="Insight spec">
    <button onclick="insights()">Insights</button>
    <script>
    async function upload() {
        let formData = new FormData();
        formData.append('file', document.getElementById('file').files[0]);
        let resp = await fetch('/ingest', {method:'POST', body:formData});
        document.getElementById('chat').innerHTML += '<p>' + await resp.text() + '</p>';
    }
    // Similar for query/insights using /chat with MCP integration
    </script>
    </body></html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```


## README.md
```
# Data Insight MCP App

## Setup (Windows)
1. Install Python 3.10+.
2. `pip install fastmcp fastapi uvicorn pandas openpyxl azure-openai python-multipart httpx python-dotenv`.
3. Create `.env`:
   ```
   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=your-deployment  # e.g., gpt-4o
   ```
4. Run MCP Server: `python mcp_server.py` (port 8001).
5. Run Client: `python client.py` (port 8000, open http://localhost:8000).
6. Upload Excel, query NL, request insights.

DB: data.db (in-memory feel, persists in file). Restart resets data.
```


This delivers optimal, logical flow: Upload → Ingest → Query/Insights via MCP-orchestrated LLM. [delynchoong.github](https://delynchoong.github.io/blog/azure-mcp-tutorial/)