"""
MCP Client - FastAPI Application
Serves HTML5 UI and connects to MCP server for data operations
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import json
import asyncio
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack, asynccontextmanager
from mcp import ClientSession
from mcp.client.sse import sse_client
import time

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
)

# Initialize FastAPI app
app = FastAPI(title="Data Insight MCP Client")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP server configuration
MCP_SSE_URL = "http://127.0.0.1:8001/sse"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and keep a single MCP ClientSession open for the FastAPI app lifetime."""
    exit_stack = AsyncExitStack()
    try:
        # Connect via SSE transport. URL should match your server's SSE endpoint.
        read, write = await exit_stack.enter_async_context(sse_client(MCP_SSE_URL))
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        app.state.mcp_session = session
        yield
    finally:
        await exit_stack.aclose()


# Initialize FastAPI app (after defining lifespan)
app = FastAPI(title="Data Insight MCP Client", lifespan=lifespan)

def _normalize_tool_result(call_result):
    """Convert MCP CallToolResult into plain Python (dict/str) for JSON responses."""
    # call_result usually has: content (list), isError (bool)
    is_error = getattr(call_result, 'isError', False)
    content = getattr(call_result, 'content', call_result)

    parts = []
    if isinstance(content, list):
        for item in content:
            if hasattr(item, 'text'):
                parts.append(item.text)
            else:
                # Fall back to pydantic dump / string
                dumped = None
                if hasattr(item, 'model_dump'):
                    try:
                        dumped = item.model_dump()
                    except Exception:
                        dumped = None
                parts.append(dumped if dumped is not None else str(item))
    else:
        parts.append(content)

    # If single text blob that looks like JSON, parse it.
    if len(parts) == 1 and isinstance(parts[0], str):
        txt = parts[0].strip()
        if (txt.startswith('{') and txt.endswith('}')) or (txt.startswith('[') and txt.endswith(']')):
            import json
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, dict):
                    parsed.setdefault('status', 'error' if is_error else parsed.get('status', 'success'))
                return parsed
            except Exception:
                pass
        return {'status': 'error' if is_error else 'success', 'response': parts[0]}

    return {'status': 'error' if is_error else 'success', 'content': parts}


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call a tool on the connected MCP server using the MCP protocol (SSE transport)."""
    session: ClientSession = app.state.mcp_session
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
        return _normalize_tool_result(result)
    except Exception as e:
        return {"status": "error", "message": f"Error calling MCP tool '{tool_name}': {e}"}


async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Discover available tools from MCP server."""
    session: ClientSession = app.state.mcp_session
    try:
        res = await session.list_tools()
        tools = getattr(res, 'tools', res)
        out = []
        for t in tools:
            # Tool objects typically have name/description/inputSchema
            name = getattr(t, 'name', None)
            desc = getattr(t, 'description', '')
            schema = getattr(t, 'inputSchema', None)
            if schema is None and hasattr(t, 'model_dump'):
                try:
                    schema = t.model_dump().get('inputSchema')
                except Exception:
                    schema = None
            out.append({"name": name, "description": desc, "inputSchema": schema})
        return out
    except Exception:
        return []


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serve the HTML5 frontend
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Insight Platform</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            padding: 30px;
        }

        .sidebar {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            height: fit-content;
        }

        .section {
            margin-bottom: 30px;
        }

        .section h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: #f8f9fa;
        }

        .upload-area.dragover {
            border-color: #764ba2;
            background: #e9ecef;
        }

        input[type="file"] {
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
            margin-top: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .chat-container {
            background: #f8f9fa;
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            height: 700px;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 25px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            padding: 15px 20px;
            border-radius: 12px;
            max-width: 80%;
            word-wrap: break-word;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
        }

        .message.assistant {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            margin-right: auto;
        }

        .message.system {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffc107;
            margin: 0 auto;
            font-size: 0.9em;
        }

        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            margin: 0 auto;
        }

        .chat-input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        input[type="text"], textarea, select {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
            font-family: inherit;
        }

        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        textarea {
            resize: vertical;
            min-height: 60px;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .file-info {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }

        .file-info.visible {
            display: block;
        }

        .file-info h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .file-info ul {
            list-style: none;
            padding-left: 0;
        }

        .file-info li {
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }

        .file-info li:last-child {
            border-bottom: none;
        }

        .insights-form {
            background: white;
            padding: 15px;
            border-radius: 8px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #667eea;
            font-weight: 600;
        }

        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
        }

        code {
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-indicator.online {
            background: #28a745;
        }

        .status-indicator.offline {
            background: #dc3545;
        }

        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Data Insight Platform</h1>
            <p>Excel Analysis with Natural Language Queries & AI-Powered Insights</p>
        </div>

        <div class="main-content">
            <div class="sidebar">
                <div class="section">
                    <h2>
                        <span class="status-indicator online"></span>
                        Upload Excel File
                    </h2>
                    <div class="upload-area" id="uploadArea">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <p style="margin-top: 15px; color: #667eea; font-weight: 600;">Click or drag Excel file here</p>
                        <p style="margin-top: 5px; color: #6c757d; font-size: 0.9em;">Supports .xlsx files</p>
                        <input type="file" id="fileInput" accept=".xlsx">
                    </div>
                    <button class="btn" id="uploadBtn" disabled>Upload File</button>

                    <div class="file-info" id="fileInfo">
                        <h3>Uploaded File</h3>
                        <ul id="fileDetails"></ul>
                    </div>
                </div>

                <div class="section">
                    <h2>Generate Insights</h2>
                    <div class="insights-form">
                        <div class="form-group">
                            <label>Insight Type</label>
                            <select id="insightSpec">
                                <option value="">General Overview</option>
                                <option value="correlations">Correlations</option>
                                <option value="outliers">Outlier Detection</option>
                                <option value="trends">Trend Analysis</option>
                            </select>
                        </div>
                        <button class="btn" id="insightsBtn">Generate Insights</button>
                    </div>
                </div>
            </div>

            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message system">
                        Welcome! Upload an Excel file to get started. You can then ask questions about your data in natural language.
                    </div>
                </div>
                <div class="chat-input-area">
                    <div class="input-group">
                        <input type="text" id="queryInput" placeholder="Ask a question about your data..." disabled>
                        <button class="btn" id="queryBtn" style="width: auto; padding: 12px 30px;" disabled>Send</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');
        const fileInfo = document.getElementById('fileInfo');
        const fileDetails = document.getElementById('fileDetails');
        const chatMessages = document.getElementById('chatMessages');
        const queryInput = document.getElementById('queryInput');
        const queryBtn = document.getElementById('queryBtn');
        const insightsBtn = document.getElementById('insightsBtn');
        const insightSpec = document.getElementById('insightSpec');

        let currentFile = null;
        let isUploaded = false;

        // File upload handling
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect();
            }
        });

        fileInput.addEventListener('change', handleFileSelect);

        function handleFileSelect() {
            if (fileInput.files.length > 0) {
                currentFile = fileInput.files[0];
                uploadBtn.disabled = false;
                uploadArea.querySelector('p').textContent = `Selected: ${currentFile.name}`;
            }
        }

        uploadBtn.addEventListener('click', async () => {
            if (!currentFile) return;

            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="spinner"></span> Uploading...';

            try {
                const formData = new FormData();
                formData.append('file', currentFile);

                const response = await fetch('/ingest', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.status === 'success') {
                    addMessage('system', `Successfully uploaded: ${result.file_name}\\nIngested ${result.sheets_ingested} sheet(s)`);

                    fileInfo.classList.add('visible');
                    fileDetails.innerHTML = result.table_names.map(table =>
                        `<li>${table}</li>`
                    ).join('');

                    isUploaded = true;
                    queryInput.disabled = false;
                    queryBtn.disabled = false;

                    uploadBtn.innerHTML = 'Upload Another File';
                    uploadBtn.disabled = false;
                } else {
                    addMessage('error', `Upload failed: ${result.message}`);
                    uploadBtn.innerHTML = 'Upload File';
                    uploadBtn.disabled = false;
                }
            } catch (error) {
                addMessage('error', `Error: ${error.message}`);
                uploadBtn.innerHTML = 'Upload File';
                uploadBtn.disabled = false;
            }
        });

        // Chat functionality
        queryBtn.addEventListener('click', sendQuery);
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !queryBtn.disabled) {
                sendQuery();
            }
        });

        async function sendQuery() {
            const query = queryInput.value.trim();
            if (!query) return;

            addMessage('user', query);
            queryInput.value = '';
            queryBtn.disabled = true;
            queryInput.disabled = true;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `message=${encodeURIComponent(query)}`
                });

                const result = await response.json();

                if (result.status === 'success') {
                    addMessage('assistant', result.response);
                } else {
                    addMessage('error', result.message || 'Failed to process query');
                }
            } catch (error) {
                addMessage('error', `Error: ${error.message}`);
            }

            queryBtn.disabled = false;
            queryInput.disabled = false;
            queryInput.focus();
        }

        // Insights functionality
        insightsBtn.addEventListener('click', async () => {
            if (!isUploaded) {
                addMessage('error', 'Please upload a file first');
                return;
            }

            const spec = insightSpec.value;
            insightsBtn.disabled = true;
            insightsBtn.innerHTML = '<span class="spinner"></span> Generating...';

            addMessage('system', `Generating ${spec || 'general'} insights...`);

            try {
                const response = await fetch('/insights', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `spec=${encodeURIComponent(spec)}`
                });

                const result = await response.json();

                if (result.status === 'success') {
                    addMessage('assistant', formatInsights(result));
                } else {
                    addMessage('error', result.message || 'Failed to generate insights');
                }
            } catch (error) {
                addMessage('error', `Error: ${error.message}`);
            }

            insightsBtn.disabled = false;
            insightsBtn.innerHTML = 'Generate Insights';
        });

        function addMessage(type, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.textContent = content;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function formatInsights(result) {
            let formatted = `Insights Analysis\\n\\n`;
            formatted += `Tables Analyzed: ${result.tables_analyzed}\\n\\n`;
            formatted += `Executive Summary:\\n${result.executive_summary}\\n\\n`;
            formatted += `View full details in the response.`;
            return formatted;
        }

        // Check server status on load
        async function checkServerStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                if (data.mcp_server === 'connected') {
                    document.querySelector('.status-indicator').classList.add('online');
                }
            } catch (error) {
                document.querySelector('.status-indicator').classList.remove('online');
                document.querySelector('.status-indicator').classList.add('offline');
            }
        }

        checkServerStatus();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/ingest")
async def ingest_excel(file: UploadFile = File(...)):
    """
    Upload and ingest Excel file via MCP server
    """
    try:
        # Read file content
        content = await file.read()

        # Encode to base64
        base64_content = base64.b64encode(content).decode('utf-8')

        # Call MCP ingest_excel tool
        result = await call_mcp_tool("ingest_excel", {
            "file_name": file.filename,
            "base64_content": base64_content
        })

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/chat")
async def chat(message: str = Form(...)):
    """
    Handle natural language queries using Azure OpenAI with MCP tools
    """
    try:
        # First, try to answer with query_data tool
        result = await call_mcp_tool("query_data", {
            "nl_query": message,
            "max_attempts": 3
        })

        if result.get("status") == "success":
            # Format the response
            response_text = f"Query: {result['query']}\\n\\n"

            if result.get('sql'):
                response_text += f"SQL: {result['sql']}\\n\\n"

            response_text += f"Results ({result['row_count']} rows):\\n"

            # Display results in a readable format
            if result['row_count'] > 0:
                results = result['results'][:10]  # Limit to first 10 rows
                for i, row in enumerate(results, 1):
                    response_text += f"\\n{i}. "
                    response_text += ", ".join([f"{k}: {v}" for k, v in row.items()])

                if result['row_count'] > 10:
                    response_text += f"\\n\\n... and {result['row_count'] - 10} more rows"
            else:
                response_text += "No results found"

            return JSONResponse(content={
                "status": "success",
                "response": response_text
            })
        else:
            # If query failed, use Azure OpenAI to provide context
            return JSONResponse(content={
                "status": "error",
                "message": result.get("message", "Failed to process query")
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/insights")
async def generate_insights(spec: str = Form("")):
    """
    Generate insights from ingested data
    """
    try:
        # Call MCP generate_insights tool
        result = await call_mcp_tool("generate_insights", {
            "tables": None,  # Analyze all tables
            "spec": spec
        })
        print(f"Insights generation result: {result}")

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/health")
async def health_check():
    """Health check endpoint (checks MCP session readiness)."""
    mcp_status = "disconnected"
    try:
        session: ClientSession = app.state.mcp_session
        # Prefer ping if available; fall back to listing tools.
        if hasattr(session, 'send_ping'):
            await session.send_ping()
        else:
            await session.list_tools()
        mcp_status = "connected"
    except Exception:
        mcp_status = "disconnected"
    return {
        "status": "healthy",
        "mcp_server": mcp_status,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Client on http://127.0.0.1:8000")
    print("Make sure MCP server is running on http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8000)
