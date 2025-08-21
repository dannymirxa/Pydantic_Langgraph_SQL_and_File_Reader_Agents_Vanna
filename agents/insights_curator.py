import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import OPENAI_MODEL

from dataclasses import dataclass
from typing_extensions import List, TypeAlias, Union, Dict, Any
import pandas as pd
import json
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import io


from tool_functions.chromadb_operations import get_content_from_collection

@dataclass
class Dependencies:
    query_results: str

class DataframeSuccess(BaseModel):
    data_insights: List[str] = Field(default_factory=list, description="List of key insights derived from the Pandas DataFrame.")

class InsightsError(BaseModel):
    error_message: str

class DataFrameMetadata(BaseModel):
    num_rows: int
    num_columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]

DataframeResponse: TypeAlias = Union[
                                     DataframeSuccess,
                                     InsightsError,
                                     DataFrameMetadata,
                                     ]

data_insights_agent = Agent(
    model=OPENAI_MODEL,
    output_type=DataframeResponse,
    retries=3,
)

@data_insights_agent.tool
def get_documentation_tool(ctx: RunContext[Dependencies], query: str) -> List[str]:
    result = get_content_from_collection(
        query=query,
        db_folder='chroma_langchain_db',
        collection_name='documentation',
        n_results=3
    )
    return result

@data_insights_agent.tool
def create_dataframe_pd_tool(ctx: RunContext[Dependencies]):
    try:
        # Create DataFrame from the provided JSON string
        df = pd.read_json(io.StringIO(ctx.deps.query_results))
        
        # Return DataFrame metadata instead of the DataFrame itself
        return DataFrameMetadata(
            num_rows=len(df),
            num_columns=len(df.columns),
            column_names=df.columns.tolist(),
            data_types={str(col): str(dtype) for col, dtype in df.dtypes.items()},
            missing_values={str(col): int(count) for col, count in df.isnull().sum().items()}
        )
    except Exception as e:
        # It's good practice to handle potential errors
        # and return a consistent error format.
        return InsightsError(error_message=f"Error processing query results: {str(e)}")



@data_insights_agent.system_prompt
def system_prompt() -> str:
    return """
    You are a data insights agent designed to process user queries and extract insights from data. Your tasks are as follows:

    1. **Receive Input**: You will receive a user query and a dictionary string containing data.

    2. **Fetch Documentation**: Use the `get_documentation_tool` to retrieve relevant documentation that may assist in understanding the context of the data. You need to passes the user's query to it.

    3. **Extract Metadata**: Utilize the `create_dataframe_pd_tool` to parse the dictionary string into a Pandas DataFrame and extract metadata such as the number of rows, columns, column names, data types, and missing values:
        - The total number of rows (num_rows).
        - The total number of columns (num_columns).
        - The names of all columns (column_names).
        - The data types of each column (data_types).
        - Any missing values per column (missing_values).

    4. **Generate Insights**: Based on the extracted metadata, analyze the data to generate key insights that are relevant to the user's query. These insights should be concise, informative, and directly address the user's query.

    5. **Return Results**: 
       - If insights are successfully generated, return them in a `DataframeSuccess` object.
       - If an error occurs at any stage, return an error message in an `InsightsError` object.

    Ensure that your responses are clear and adhere to the expected output format. Handle any exceptions gracefully and provide meaningful error messages when necessary. Always aim to transform metadata into actionable insights that fulfill the user's query.

    **Important**: The `create_dataframe_pd_tool` will return a `DataFrameMetadata` object. You must then use the information from this `DataFrameMetadata` object to formulate the `data_insights` and return a `DataframeSuccess` object. Do NOT return the `DataFrameMetadata` object as your final output.
    """

@data_insights_agent.output_validator
def data_insights_agent_output_validator(ctx: RunContext[Dependencies], output: Union[DataframeSuccess, DataFrameMetadata]) -> DataframeResponse:
    """
    Validates the parsed output object from the FileReaderAgent.
    This function is called by pydantic-ai after it attempts to parse the LLM's raw output
    into the specified output_type (FileResponse).
    """
    if isinstance(output, InsightsError):
        # If pydantic-ai already determined it's an InsightsError, just return it.
        print(f"data_insights_agent Result Validator: Received InsightsError: {output.error_message}")
        return output
    
    if isinstance(output, DataFrameMetadata):
        # If the output is DataFrameMetadata, the agent needs to continue to generate insights.
        # This part of the validator should not be reached if the agent is correctly following the prompt.
        # However, as a safeguard, we can return an error or prompt the agent to continue.
        return InsightsError(error_message="Agent returned DataFrameMetadata as final output, but DataframeSuccess was expected. The agent must generate insights based on the metadata and return DataframeSuccess.")
    elif isinstance(output, DataframeSuccess):
        if not output.data_insights:
            print("data_insights_agent Result Validator: No insights can be made")
            return InsightsError(error_message="No insights can be made")
        
        print("data_insights_agent Result Validator: DataframeSuccess object passed custom validation.")
        return output

    
# def main():
#     sql_result = """
#                     {"driver_count":{"0":16}}
#                  """

#     user_query = "How many drivers for company Accenture, product Transformation GPS in Cycle 1? Visualize in table"

#     deps = Dependencies(query_results=sql_result)

#     result = data_insights_agent.run_sync(user_query, deps=deps)

#     print(result)


# if __name__=="__main__":
#     main()

  