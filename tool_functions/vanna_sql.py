import os
from pydantic import BaseModel, Field, field_validator
from vanna.openai import OpenAI_Chat
from openai import AzureOpenAI
from vanna.chromadb import ChromaDB_VectorStore

from typing import Optional
from dotenv import load_dotenv

import json

load_dotenv('.env')

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=AzureOpenAI(
                                 api_version="2024-12-01-preview",
                                 azure_endpoint="https://llmcoechangemateopenai2.openai.azure.com/",
                                 api_key=os.getenv("AZURE_OPENAI_KEY"),
                              ), config=config) # Make sure to put your AzureOpenAI client here
        
# Define a Pydantic model for PostgreSQL configuration
class PostgresConfig(BaseModel):
    host: str
    dbname: str
    user: str
    password: str
    port: int = Field(..., gt=0, lt=65536)

    @field_validator('host', 'dbname', 'user', 'password')
    def not_empty(cls, v):
        if not v:
            raise ValueError('must not be empty')
        return v

def create_vanna_client(postgres_config: PostgresConfig) -> MyVanna:
    vn = MyVanna(config={'model': 'gpt-4o-dev', 'path': './chroma_langchain_db'})
    vn.connect_to_postgres(**postgres_config.model_dump())

    return vn

def train_vanna_client(vn: MyVanna, sql_training_data_path: str) -> MyVanna:
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

    plan = vn.get_training_plan_generic(df_information_schema)

    vn.train(plan=plan)

    # Sometimes you may want to add documentation about your business terminology or definitions.
    # vn.train(documentation="The names of company, product, cycle and product version of company Accenture, product Transformation GPS and cycle Cycle 1")

    # You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.

    with open(sql_training_data_path, 'r') as f:
        sqlFile = f.read()
    
    vn.train(sql=sqlFile)

    return vn

if __name__=="__main__":
    # Validate postgres_config using Pydantic
    postgres_config = PostgresConfig(
        host='localhost',
        dbname='ctre_unstable',
        user='orgplatform',
        password='orgplatform',
        port=5432
    )
    vn = create_vanna_client(postgres_config)

    # vn = train_vanna_client(vn=vn, sql_training_data_path='tool_functions/product_version.sql')

    # ## Asking the AI
    # Whenever you ask a new question, it will find the 10 most relevant pieces of training data and use it as part of the LLM prompt to generate the SQL.

    # Ask the AI
    sql_query, result_df, _ = vn.ask(
        question="What are the product drivers for company Accenture?",
        print_results=False,
        visualize=False,
        allow_llm_to_see_data=True
    )

    # Print the SQL query
    print("Generated SQL Query:")
    print(sql_query)

    # Print the result of the SQL query
    print("\nQuery Result:")
    if result_df is not None:
        print(json.loads(result_df.to_json()))
    else:
        print("No results returned.")

