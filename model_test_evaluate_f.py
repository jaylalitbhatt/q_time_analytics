import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
import mysql.connector
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

# ------------------ Load Environment ------------------
load_dotenv()

# Azure OpenAI config
api_type = "azure"
endpoint = os.getenv("AZURE_OPENAI_API_BASE")
key = os.getenv("AZURE_OPENAI_API_KEY")
version = "2024-02-01"

# MySQL config
host = os.getenv('MYSQL_HOST')
user = os.getenv('MYSQL_USER')
password = os.getenv('MYSQL_PASSWORD')
database = os.getenv('MYSQL_DB')
encoded_user = quote_plus(user)
encoded_password = quote_plus(password)
db_uri = f"mysql+pymysql://{encoded_user}:{encoded_password}@{host}:3306/{database}"

# ------------------ Test DB Connection ------------------
try:
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    print("✅ MySQL connection successful!")
    connection.close()
except Exception as e:
    print(f"❌ MySQL Connection failed: {e}")
    raise

# ------------------ Setup Agent ------------------
db = SQLDatabase.from_uri(db_uri, include_tables=["time_model"])

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version=version,
    temperature=0,
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
agent = create_react_agent(llm, tools)

# ------------------ Chat Loop ------------------
print("\n🧠 Ask any question about the time_model table (type 'exit' to quit):")
while True:
    question = input("\n🔹 Your question: ")
    if question.strip().lower() in ["exit", "quit", "q"]:
        print("👋 Exiting. Goodbye!")
        break

    try:
        for step in agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
    except Exception as e:
        print(f"❌ Error during response generation: {e}")
