import sys
import os

# Add the current workspace directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tool_functions.vanna_sql import PostgresConfig, create_vanna_client
from models import OPENAI_MODEL

from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, Union, Optional, List
from annotated_types import MinLen
from pydantic_ai import Agent, ModelRetry, RunContext
import pandas as pd
import json

from dotenv import load_dotenv

load_dotenv('.env')

class SQLSuccess(BaseModel):
    sql_query: Annotated[str, MinLen(1)] = Field(alias='sql_query', description='SQL query to run')
    query_results: Optional[str] = Field(None, alias='query_results', description='JSON string of the query results')
    answer: str = Field(alias='answer', description='Answer from the SQL query')

class InvalidRequest(BaseModel):
    error_message: str

SQLResponse: TypeAlias = Union[SQLSuccess, InvalidRequest]

@dataclass
class Dependencies:
    connection_string: PostgresConfig

sql_query_result_agent = Agent(
    model=OPENAI_MODEL,
    output_type=SQLResponse,
    retries=3,
    model_settings={'temperature': 0.1}
)

@sql_query_result_agent.tool
def get_sql_query_and_result(ctx: RunContext[Dependencies], question:str):
    vn = create_vanna_client(ctx.deps.connection_string)
    sql_query, result_df, _ = vn.ask(
        question=question,
        print_results=False,
        visualize=False,
        allow_llm_to_see_data=True
    )
    return sql_query, result_df.to_json()

@sql_query_result_agent.system_prompt
def system_prompt(ctx: RunContext[Dependencies]) -> str:
    return f"""
    You are an intelligent SQL query agent. Your task is to process any given question by passing it to the 'get_sql_query_and_result' tool. 
    This tool will return the SQL query executed and the results in JSON format. Your job is to interpret these results and provide a clear, 
    concise answer in natural language. Ensure that your response is accurate and easy to understand, summarizing the key findings from the 
    SQL query results.
    """

@sql_query_result_agent.output_validator
def sql_query_result_agent_output_validator(ctx: RunContext[Dependencies], output: SQLResponse) -> SQLResponse:
    """
    Validates the parsed output object from the SQLAgent.
    """
    if isinstance(output, InvalidRequest):
        print(f"SQLAgent Result Validator: Received InvalidRequest: {output.error_message}")
        return output
    
    if isinstance(output, SQLSuccess):
        if not output.sql_query:
            print("SQLAgent Result Validator: SQLSuccess object has an empty sql_query.")
            return InvalidRequest(error_message="SQLSuccess object has an empty sql_query.")
        
        if not output.query_results:
            print("SQLAgent Result Validator: SQLSuccess object has an empty or missing query_results_json.")
            return InvalidRequest(error_message="SQLSuccess object has an empty or missing query_results. It must be a valid JSON string, even if empty (e.g., '[]').")

        print("SQLAgent Result Validator: SQLSuccess object passed custom validation.")
        return output
    
def main():
    postgres_config = PostgresConfig(
        host='localhost',
        dbname='ctre_unstable',
        user='orgplatform',
        password='orgplatform',
        port=5432
    )

    deps = Dependencies(connection_string=postgres_config)

    user_query = "How many product drivers for company Accenture in Cycle 1?"

    result = sql_query_result_agent.run_sync(user_query, deps=deps)

    print(result)

if __name__=="__main__":
    main()