#!/usr/bin/env python3
"""
BigQuery Natural Language API - Multi-Model Support
Supports Claude, OpenAI GPT, and Google Gemini
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google.cloud import bigquery
from google.oauth2 import service_account
import anthropic
import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="BigQuery Natural Language API",
    description="Multi-model AI for BigQuery queries (Claude, OpenAI, Gemini)",
    version="2.0.0",
    docs_url="/docs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
bq_client: Optional[bigquery.Client] = None
db_pool: Optional[asyncpg.Pool] = None

# Configuration
API_KEY = os.getenv("API_KEY", "")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")


class ModelProvider(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"


class DatasetInfo(BaseModel):
    """Dataset information"""
    project_id: str
    dataset_id: str
    location: Optional[str] = None
    created: Optional[str] = None
    modified: Optional[str] = None
    table_count: int = 0


class TableInfo(BaseModel):
    """Table information"""
    table_id: str
    full_name: str
    num_rows: Optional[int] = None
    size_bytes: Optional[int] = None
    created: Optional[str] = None


class QueryRequest(BaseModel):
    """Request for natural language query"""
    query: str = Field(..., min_length=1)
    dataset_id: Optional[str] = None
    project_id: Optional[str] = None
    limit: int = Field(100, ge=1, le=10000)
    model_provider: Optional[ModelProvider] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None  # For user's own API keys


class QueryResponse(BaseModel):
    """Response with query results"""
    success: bool
    query: str
    sql: Optional[str] = None
    data: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    timestamp: str
    message: str
    model_used: str


class ModelConfig(BaseModel):
    """Model configuration for user"""
    user_id: str
    provider: ModelProvider
    model_name: str
    api_key: str  # Encrypted in DB


async def init_db():
    """Initialize database connection and tables"""
    global db_pool
    
    if db_pool is None:
        logger.info("Initializing database connection...")
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        
        # Create tables
        async with db_pool.acquire() as conn:
            # Conversations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) NOT NULL,
                    dataset_id VARCHAR(255),
                    project_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Messages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    sql_query TEXT,
                    result_count INTEGER,
                    model_used VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Model configurations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_configs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) UNIQUE NOT NULL,
                    provider VARCHAR(50) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    api_key_encrypted TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user 
                ON conversations(user_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            
            logger.info("✓ Database tables initialized")


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key"""
    if not API_KEY:
        return
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def init_bigquery():
    """Initialize BigQuery client"""
    global bq_client
    
    if bq_client is None:
        logger.info("Initializing BigQuery client...")
        
        gcp_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
        if gcp_json:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                f.write(gcp_json)
                credentials = service_account.Credentials.from_service_account_file(f.name)
                bq_client = bigquery.Client(
                    credentials=credentials,
                    project=GCP_PROJECT_ID
                )
                os.unlink(f.name)
        else:
            bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        
        logger.info("✓ BigQuery client initialized")


async def get_user_model_config(user_id: str) -> Optional[Dict]:
    """Get user's model configuration from DB"""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT provider, model_name, api_key_encrypted FROM model_configs WHERE user_id = $1",
            user_id
        )
        if row:
            return {
                "provider": row["provider"],
                "model_name": row["model_name"],
                "api_key": row["api_key_encrypted"]  # TODO: Decrypt
            }
    return None


async def convert_nl_to_sql_claude(
    query: str,
    schema_info: str,
    project_id: str,
    dataset_id: str,
    api_key: Optional[str] = None
) -> str:
    """Convert natural language to SQL using Claude"""
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
    
    system_prompt = f"""You are a SQL expert. Convert natural language queries to BigQuery SQL.

Available tables and schemas:
{schema_info}

Rules:
1. Use standard BigQuery SQL syntax
2. Always use fully qualified table names: `{project_id}.{dataset_id}.table_name`
3. Use appropriate date functions for time-based queries
4. Add LIMIT clause if not specified (default 100)
5. Return ONLY the SQL query, no explanations
6. For "last week": WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
7. For "this month": WHERE date >= DATE_TRUNC(CURRENT_DATE(), MONTH)
8. Format SQL on one line for execution"""

    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        temperature=0.1,
        system=system_prompt,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    
    sql = message.content[0].text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    logger.info(f"Generated SQL (Claude): {sql}")
    return sql


async def convert_nl_to_sql_openai(
    query: str,
    schema_info: str,
    project_id: str,
    dataset_id: str,
    api_key: str,
    model_name: str = "gpt-4"
) -> str:
    """Convert natural language to SQL using OpenAI"""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    system_prompt = f"""You are a SQL expert. Convert natural language queries to BigQuery SQL.

Available tables and schemas:
{schema_info}

Rules:
1. Use standard BigQuery SQL syntax
2. Always use fully qualified table names: `{project_id}.{dataset_id}.table_name`
3. Use appropriate date functions
4. Return ONLY the SQL query, no explanations"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    logger.info(f"Generated SQL (OpenAI): {sql}")
    return sql


async def convert_nl_to_sql_gemini(
    query: str,
    schema_info: str,
    project_id: str,
    dataset_id: str,
    api_key: str,
    model_name: str = "gemini-pro"
) -> str:
    """Convert natural language to SQL using Gemini"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"""You are a SQL expert. Convert this natural language query to BigQuery SQL.

Available tables and schemas:
{schema_info}

Query: {query}

Rules:
1. Use fully qualified table names: `{project_id}.{dataset_id}.table_name`
2. Return ONLY the SQL query, no explanations

SQL:"""

    response = model.generate_content(prompt)
    sql = response.text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    logger.info(f"Generated SQL (Gemini): {sql}")
    return sql


async def get_dataset_schemas(project_id: str, dataset_id: str) -> str:
    """Get schemas of all tables in dataset"""
    init_bigquery()
    
    try:
        dataset_ref = bq_client.dataset(dataset_id, project=project_id)
        tables = list(bq_client.list_tables(dataset_ref))
        
        if not tables:
            return "No tables found in the dataset."
        
        schema_info = []
        for table in tables[:15]:  # Limit to 15 tables
            table_ref = dataset_ref.table(table.table_id)
            table_obj = bq_client.get_table(table_ref)
            
            columns = []
            for field in table_obj.schema[:20]:  # Limit columns
                columns.append(f"  - {field.name} ({field.field_type})")
            
            schema_info.append(
                f"Table: {table.table_id}\n" + "\n".join(columns)
            )
        
        return "\n\n".join(schema_info)
    
    except Exception as e:
        logger.error(f"Error getting schemas: {e}")
        return f"Error retrieving schemas: {str(e)}"


async def execute_sql(sql: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Execute SQL on BigQuery"""
    init_bigquery()
    
    try:
        if "LIMIT" not in sql.upper():
            sql = f"{sql} LIMIT {limit}"
        
        logger.info(f"Executing SQL: {sql}")
        query_job = bq_client.query(sql)
        results = query_job.result()
        
        rows = []
        for row in results:
            rows.append(dict(row))
        
        logger.info(f"Query returned {len(rows)} rows")
        return rows
    
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.on_event("startup")
async def startup():
    """Startup tasks"""
    await init_db()
    init_bigquery()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup"""
    if db_pool:
        await db_pool.close()


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "BigQuery Natural Language API",
        "version": "2.0.0",
        "status": "running",
        "models_supported": ["claude", "openai", "gemini"],
        "endpoints": {
            "health": "GET /health",
            "datasets": "GET /api/datasets",
            "tables": "GET /api/tables",
            "query": "POST /api/query",
            "model_config": "POST /api/model/config"
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "bigquery": bq_client is not None,
        "database": db_pool is not None
    }


@app.get("/api/datasets")
async def list_datasets(
    project_id: Optional[str] = None,
    authorization: Optional[str] = Header(None)
):
    """List all datasets in project"""
    await verify_api_key(authorization)
    init_bigquery()
    
    try:
        proj_id = project_id or GCP_PROJECT_ID
        datasets = list(bq_client.list_datasets(project=proj_id))
        
        result = []
        for dataset in datasets:
            dataset_ref = bq_client.dataset(dataset.dataset_id, project=proj_id)
            dataset_obj = bq_client.get_dataset(dataset_ref)
            
            # Count tables
            tables = list(bq_client.list_tables(dataset_ref))
            
            result.append({
                "project_id": proj_id,
                "dataset_id": dataset.dataset_id,
                "location": dataset_obj.location,
                "created": dataset_obj.created.isoformat() if dataset_obj.created else None,
                "modified": dataset_obj.modified.isoformat() if dataset_obj.modified else None,
                "table_count": len(tables)
            })
        
        return {"datasets": result, "count": len(result)}
    
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tables")
async def list_tables(
    dataset_id: str,
    project_id: Optional[str] = None,
    authorization: Optional[str] = Header(None)
):
    """List tables in dataset"""
    await verify_api_key(authorization)
    init_bigquery()
    
    try:
        proj_id = project_id or GCP_PROJECT_ID
        dataset_ref = bq_client.dataset(dataset_id, project=proj_id)
        tables = list(bq_client.list_tables(dataset_ref))
        
        result = []
        for table in tables:
            table_ref = dataset_ref.table(table.table_id)
            table_obj = bq_client.get_table(table_ref)
            
            result.append({
                "table_id": table.table_id,
                "full_name": f"{proj_id}.{dataset_id}.{table.table_id}",
                "num_rows": table_obj.num_rows,
                "size_bytes": table_obj.num_bytes,
                "created": table_obj.created.isoformat() if table_obj.created else None
            })
        
        return {"tables": result, "count": len(result)}
    
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_natural_language(
    request: QueryRequest,
    user_id: str = Header(None, alias="X-User-ID"),
    authorization: Optional[str] = Header(None)
):
    """Main query endpoint with multi-model support"""
    await verify_api_key(authorization)
    
    start_time = datetime.now()
    
    try:
        # Determine project and dataset
        project_id = request.project_id or GCP_PROJECT_ID
        
        if not request.dataset_id:
            raise HTTPException(
                status_code=400,
                detail="dataset_id is required"
            )
        
        # Get user's model config or use default
        model_config = None
        if user_id:
            model_config = await get_user_model_config(user_id)
        
        # Determine which model to use
        provider = request.model_provider or (
            model_config["provider"] if model_config else ModelProvider.CLAUDE
        )
        
        # Get schemas
        schema_info = await get_dataset_schemas(project_id, request.dataset_id)
        
        # Convert NL to SQL based on provider
        if provider == ModelProvider.CLAUDE:
            api_key = request.api_key or (
                model_config["api_key"] if model_config else ANTHROPIC_API_KEY
            )
            sql = await convert_nl_to_sql_claude(
                request.query,
                schema_info,
                project_id,
                request.dataset_id,
                api_key
            )
            model_used = request.model_name or DEFAULT_MODEL
            
        elif provider == ModelProvider.OPENAI:
            if not request.api_key and not (model_config and model_config["api_key"]):
                raise HTTPException(400, "OpenAI API key required")
            
            api_key = request.api_key or model_config["api_key"]
            model_name = request.model_name or (
                model_config["model_name"] if model_config else "gpt-4"
            )
            sql = await convert_nl_to_sql_openai(
                request.query,
                schema_info,
                project_id,
                request.dataset_id,
                api_key,
                model_name
            )
            model_used = model_name
            
        elif provider == ModelProvider.GEMINI:
            if not request.api_key and not (model_config and model_config["api_key"]):
                raise HTTPException(400, "Gemini API key required")
            
            api_key = request.api_key or model_config["api_key"]
            model_name = request.model_name or (
                model_config["model_name"] if model_config else "gemini-pro"
            )
            sql = await convert_nl_to_sql_gemini(
                request.query,
                schema_info,
                project_id,
                request.dataset_id,
                api_key,
                model_name
            )
            model_used = model_name
        
        # Execute query
        data = await execute_sql(sql, request.limit)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save to conversation history
        if user_id and db_pool:
            async with db_pool.acquire() as conn:
                # Get or create conversation
                conv = await conn.fetchrow(
                    """INSERT INTO conversations (user_id, dataset_id, project_id)
                       VALUES ($1, $2, $3)
                       RETURNING id""",
                    user_id, request.dataset_id, project_id
                )
                
                # Save message
                await conn.execute(
                    """INSERT INTO messages 
                       (conversation_id, role, content, sql_query, result_count, model_used)
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    conv["id"], "user", request.query, sql, len(data), model_used
                )
        
        message = f"Found {len(data)} results" if data else "No results found"
        
        return QueryResponse(
            success=True,
            query=request.query,
            sql=sql,
            data=data,
            row_count=len(data),
            execution_time_ms=execution_time,
            timestamp=datetime.utcnow().isoformat(),
            message=message,
            model_used=model_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/config")
async def save_model_config(
    config: ModelConfig,
    authorization: Optional[str] = Header(None)
):
    """Save user's model configuration"""
    await verify_api_key(authorization)
    
    try:
        # TODO: Encrypt api_key before storing
        encrypted_key = config.api_key  # Placeholder
        
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO model_configs (user_id, provider, model_name, api_key_encrypted)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (user_id) 
                   DO UPDATE SET 
                       provider = EXCLUDED.provider,
                       model_name = EXCLUDED.model_name,
                       api_key_encrypted = EXCLUDED.api_key_encrypted,
                       updated_at = NOW()""",
                config.user_id,
                config.provider.value,
                config.model_name,
                encrypted_key
            )
        
        return {"success": True, "message": "Model configuration saved"}
    
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"Project: {GCP_PROJECT_ID}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
