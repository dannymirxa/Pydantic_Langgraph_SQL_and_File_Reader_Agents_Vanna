import sys
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from typing_extensions import Annotated, TypeAlias, Union, Optional
from annotated_types import MinLen
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from sqlalchemy import create_engine, Engine

from models import OPENAI_MODEL
from agents import  file_reader, sql_query_creator_insights_curator  
from agents.sql_query_result import sql_query_creator_agent, SQLResponse
from agents.file_reader import file_reader_agent,  FileResponse
from tool_functions.file_operations import list_files

# define which agent
SQL_AGENT = "sql_agent"
FILE_AGENT = "file_agent"
BOTH_AGENT = "both_agent"
NONE = 'none'

class MasterAgentResponse(BaseModel):
    agent: str

@dataclass
class MasterDependencies:
    db_engine: Engine
    available_files: list[str]

master_agent = Agent(
    model=OPENAI_MODEL,
    output_type=MasterAgentResponse,
    result_tool_description="To decide which agent to use.",
    result_retries=3,
  )

@master_agent.system_prompt
def master_system_prompt(ctx: RunContext[MasterDependencies]) -> str:
    return f"""\
    You are a master AI agent designed to coordinate between a SQL query creator agent and a file reader agent.
    Your primary goal is to understand the user's request and delegate it to the appropriate sub-agent.

    Here are the available sub-agents and their capabilities:
    - **SQL Query Creator Agent:** Use this agent when the user's request involves querying a database.
      It can list tables, describe table schemas, and run SQL queries.
      Use the `run_sql_query_creator_agent` tool to interact with it.
      The database engine is already configured.

    - **File Reader Agent:** Use this agent when the user's request involves reading content from files.
      It can read JSON, CSV, text, and PDF files.
      Use the `run_file_reader_agent` tool to interact with it.
      The available files are dynamically retrieved from the 'files/' directory.

    **Instructions:**
    1. Carefully analyze the user's request to determine the primary intent.
    2. If the request involves querying a database (e.g., "how many genres", "list tables", "describe table", "run query", "database", "SQL", "db"), output `{SQL_AGENT}`.
    3. If the request involves reading or summarizing content from files (e.g., "read file", "summarize file", "content of", "information from", "PDF", "CSV", "TXT", "JSON", "file", "document", "report"), output `{FILE_AGENT}`.
    4. If the request clearly involves both database interaction AND file content reading (e.g., "genres and summary of file", "SQL query and read document", "database and file content"), output `{BOTH_AGENT}`.
    5. If the request is ambiguous, prioritize the most direct interpretation.
    6. The output of the chosen sub-agent will be wrapped in a `MasterAgentResponse`.

    When `{SQL_AGENT}`, `{FILE_AGENT}`, `{BOTH_AGENT}` are not viable options, output `{NONE}`.

    Only choose from these options: `{SQL_AGENT}`, `{FILE_AGENT}`, `{BOTH_AGENT}` or `{NONE}`"
    """

@master_agent.output_validator
def validate_result(ctx: RunContext[None], response: MasterAgentResponse) -> MasterAgentResponse:
    if response.agent not in [SQL_AGENT, FILE_AGENT, BOTH_AGENT, NONE]:
        raise ModelRetry(
            f"Invalid action. Please choose from `{SQL_AGENT}`, `{FILE_AGENT}`, `{BOTH_AGENT}` or `{NONE}`"
        )

    return response