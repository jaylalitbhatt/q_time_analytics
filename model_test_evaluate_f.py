import os
import json
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import quote_plus
import mysql.connector
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from fastapi.responses import PlainTextResponse

# =====================================================
# Load Environment & Secrets
# =====================================================
def load_secrets():
    """Load secrets from AWS Secrets Manager"""
    secret_name = "Q2O_time_analytics"
    region_name = "us-east-1"

    try:
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)
        secret_value = client.get_secret_value(SecretId=secret_name)
        return json.loads(secret_value["SecretString"])
    except Exception as e:
        raise RuntimeError(f"⚠️ Could not load secrets from AWS: {e}")

secrets = load_secrets()

# Azure OpenAI config
endpoint = secrets.get("AZURE_OPENAI_API_BASE")
key = secrets.get("AZURE_OPENAI_API_KEY")
version = "2024-02-01"

# MySQL config
host = secrets.get("MYSQL_HOST")
user = secrets.get("MYSQL_USER")
password = secrets.get("MYSQL_PASSWORD")
database = secrets.get("MYSQL_DB")

encoded_user = quote_plus(user)
encoded_password = quote_plus(password)
db_uri = f"mysql+pymysql://{encoded_user}:{encoded_password}@{host}:3306/{database}"

# =====================================================
# Test MySQL Connection
# =====================================================
try:
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    connection.close()
except Exception as e:
    raise RuntimeError(f"❌ MySQL Connection failed: {e}")

# =====================================================
# Security Middleware for SQL Tools
# =====================================================
def safe_sql_tool(tool_func):
    """Wrapper to only allow SELECT queries."""
    def wrapper(query: str):
        if not query.strip().lower().startswith("select"):
            raise ValueError("❌ Only SELECT queries are allowed!")
        return tool_func(query)
    return wrapper

# =====================================================
# Setup Agent
# =====================================================
db = SQLDatabase.from_uri(db_uri, include_tables=["time_model"])
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version=version,
    temperature=0,
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Wrap SQL execution tool for security
for tool in tools:
    if tool.name.lower() == "sql_db_query":
        tool._run = safe_sql_tool(tool._run)

agent = create_react_agent(llm, tools)

# =====================================================
# FastAPI Setup
# =====================================================
app = FastAPI(title="Time Insights Query API")

class QueryRequest(BaseModel):
    question: str

@app.get("/query", response_class=PlainTextResponse)
async def query_agent(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Get final result instead of streaming intermediate steps
        result = agent.invoke(
            {"messages": [{"role": "user", "content": req.question}]}
        )

        # Extract the final answer (last message content)
        final_answer = result["messages"][-1].content
        return final_answer.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =====================================================
# Entry Point for Local Dev
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)
