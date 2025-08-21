import sys

# adding Folder_2 to the system path
# sys.path.insert(0, 'utils')
from util_functions.file_operations import list_files, read_json, read_csv, read_pdf, read_txt
from models import OPENAI_MODEL

from dotenv import load_dotenv
from dataclasses import dataclass
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, Union, Optional
from annotated_types import MinLen
from pydantic_ai import Agent, ModelRetry, RunContext


load_dotenv("/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/.env")

class FileSuccess(BaseModel):
    file_content: Annotated[str, MinLen(1)] = Field(alias='file_content', description='text content of the file')
    summary: str = Field(alias='summary', description='Summary of the file content')

class InvalidRequest(BaseModel):
    error_message: str

FileResponse: TypeAlias = Union[FileSuccess, InvalidRequest]

@dataclass
class Dependencies:
    files: list[str]

file_reader_agent = Agent(
    model=OPENAI_MODEL,
    output_type=FileResponse,
    result_retries=3,
)

@file_reader_agent.system_prompt
def system_prompt(files: list[str]) -> str:
    return f"""\
    You are a file reader agent. Your task is to read the content of a file based on the user's request.

    Here is a list of available files: {files}

    When the user asks for a file, you must:
    1. **CRITICAL:** First, check if the exact file mentioned in the user's request exists in the `available_files` list.
    2. If the exact file is NOT found in `available_files`, you MUST immediately output an `InvalidRequest` with a clear `error_message` stating that the file was not found. For example: `InvalidRequest(error_message="File 'non_existent_file.txt' not found in available files.")`. DO NOT attempt to read or infer any other file.
    3. If the exact file IS found, then determine its type (e.g., .json, .csv, .txt, .pdf).
    4. Call the appropriate tool to read the file, passing the full file path as the 'file_path' argument.

    Available tools:
    - read_json_tool(file_path: str): Use this for .json files.
    - read_csv_tool(file_path: str): Use this for .csv files.
    - read_text_tool(file_path: str): Use this for .txt files.
    - read_pdf_tool(file_path: str): Use this for .pdf files.

    Example 1: If the user says "I want the content of the bike data", and 'files/structured_bike_data_cleaned.json' is in the list, you should call read_json_tool('files/structured_bike_data_cleaned.json').
    Example 2: If the user says "I need to know more about risc", and 'files/risc.txt' is in the list, you should call read_txt_tool('files/risc.txt').
    """

@file_reader_agent.tool
def read_json_tool(ctx: RunContext[Dependencies], file_path: str):
    print(f"read_json_tool called, {file_path}")
    file_data = read_json(file_path)
    return FileSuccess(file_content=file_data["file_content"], summary=file_data["summary"])

@file_reader_agent.tool
def read_csv_tool(ctx: RunContext[Dependencies], file_path: str):
    print(f"read_csv_tool called, {file_path}")
    file_data = read_csv(file_path)
    return FileSuccess(file_content=file_data["file_content"], summary=file_data["summary"])

@file_reader_agent.tool
def read_text_tool(ctx: RunContext[Dependencies], file_path: str):
    print(f"read_text_tool called, {file_path}")
    file_data = read_txt(file_path)
    return FileSuccess(file_content=file_data["file_content"], summary=file_data["summary"])

@file_reader_agent.tool
def read_pdf_tool(ctx: RunContext[Dependencies], file_path: str):
    print(f"read_pdf_tool called, {file_path}")
    file_data = read_pdf(file_path)
    return FileSuccess(file_content=file_data["file_content"], summary=file_data["summary"])

@file_reader_agent.output_validator
def file_reader_agent_output_validator(ctx: RunContext[Dependencies], output: FileResponse) -> FileResponse:
    """
    Validates the parsed output object from the FileReaderAgent.
    This function is called by pydantic-ai after it attempts to parse the LLM's raw output
    into the specified output_type (FileResponse).
    """
    if isinstance(output, InvalidRequest):
        # If pydantic-ai already determined it's an InvalidRequest, just return it.
        print(f"FileReaderAgent Result Validator: Received InvalidRequest: {output.error_message}")
        return output
    
    if isinstance(output, FileSuccess):
        # Perform additional validation on the FileSuccess object if needed.
        # For example, ensure critical fields are not empty or have expected formats.
        if not output.file_content:
            print("FileReaderAgent Result Validator: No file in the directory can be found.")
            return InvalidRequest(error_message="No file in the directory can be found.")
        
        print("FileReaderAgent Result Validator: FileSuccess object passed custom validation.")
        return output
    