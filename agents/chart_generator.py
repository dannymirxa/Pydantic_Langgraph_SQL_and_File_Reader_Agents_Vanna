import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import OPENAI_MODEL

from dataclasses import dataclass
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, Union, Optional
from annotated_types import MinLen
from pydantic_ai import Agent, ModelRetry, RunContext
from dotenv import load_dotenv

import io

load_dotenv(".env")

@dataclass
class Dependencies:
    query_results: Optional[str]

class ChartSuccess(BaseModel):
    python_codes: Annotated[list[str], MinLen(1)] = Field(alias='python_codes', description='List of Plotly Python code strings to run, each generating a chart')
    
class ChartError(BaseModel):
    error_message: str

ChartResponse: TypeAlias = Union[
                                ChartSuccess,     
                                ChartError
                            ]

chart_creator_agent = Agent(
    model=OPENAI_MODEL,
    output_type=ChartResponse,
    retries=3,
    model_settings={'temperature': 0.1}
)

chartOptions = (
                    "Scatter Plots",
                    "Line Charts",
                    "Bar Charts",
                    "Pie Charts",
                    "Bubble Charts",
                    "Dot Plots",
                    "Filled Area Plots",
                    "Horizontal Bar Charts",
                    "Gantt Charts",
                    "Sunburst Charts",
                    "Tables",
                    "Sankey Diagram",
                    "Treemap Charts",
                    "High Performance Visualization",
                    "Figure Factory Tables",
                    "Categorical Axes",
                    "Icicle Charts",
                    "Patterns, Hatching, Texture",
                    "Dumbbell Plots"
                )

@chart_creator_agent.system_prompt
def system_prompt(ctx: RunContext[Dependencies]) -> str:
    df = pd.read_json(io.StringIO(ctx.deps.query_results))
    if not ctx.deps.query_results:
        raise ModelRetry("""
            system: Error - No data available to generate a chart. The DataFrame is missing or empty.
            Please ensure data is loaded correctly before requesting a chart.
            """)
    return \
    f"""
    system: You are an AI assistant specialized in creating insightful graphs and generating Python code for them.
    The data is already available in a pandas DataFrame named `df`.
    DataFrame columns: {list(df.columns)}
    Sample of DataFrame (first 5 rows):
    {df.head().to_markdown()}
    Your task:
    1.  Analyze the user's request and the provided DataFrame.
    2.  **Adhere to Chart Type Request:**
        a.  The user's instruction (provided as `user_input` to this agent) may specify a preferred chart type (e.g., "generate a scatter plot", "show a bar graph").
        b.  If a specific chart type is requested by the user and it is available in {chartOptions} (e.g., 'Scatter', 'Line', 'Bar', etc.) and is appropriate for the data, **you MUST generate the Python code for that specific chart type.**
        c.  If the user does not specify a chart type, or if the specified type is not in {chartOptions} or is clearly unsuitable for the data, then you may choose the most fitting chart type from {chartOptions}. In such cases, briefly explain your choice of chart type in the 'insights'.
    3.  Generate valuable insights based on the data and the chosen chart.
    4.  Produce concise and correct Python code (using `plotly.express`) to plot the graph. The code should assume `df` (the pandas DataFrame) is pre-loaded.
    5.  The Python code should be a complete, executable script.
    6.  **Crucially, for each chart, the generated Python code MUST include `fig.write_html(f'templates/chart_{{i}}.html')` to save the chart, where `i` is a unique index for each chart (e.g., `chart_0.html`, `chart_1.html`).** This allows the application to display multiple charts.
    7.  **Do NOT include `fig.show()` in the Python code**, as the chart display is handled by saving to HTML.
    8.  Return the insights and the Python code as per the `ChartResponses` model. If multiple charts are requested or appropriate, provide a list of Python code strings.

    **Important**: The DataFrame `df` provided to you is the result of a pre-filtered SQL query. It already contains the data relevant to the user's request (e.g., sales of albums by artists with at least one rock genre). Do not attempt to re-filter or re-process the data based on criteria that have already been applied upstream. Focus solely on visualizing the provided `df`.

    If the request is unclear or cannot be fulfilled with the given data, return an `ChartError` with an explanation.

    Example of Python code structure for multiple charts:
    ```python
    import plotly.express as px
    # df is assumed to be pre-loaded with the data

    # Chart 1
    fig1 = px.bar(df, x='column_x', y='column_y', title='Chart 1 Title')
    fig1.write_html('templates/chart_0.html')

    # Chart 2
    fig2 = px.line(df, x='column_a', y='column_b', title='Chart 2 Title')
    fig2.write_html('templates/chart_1.html')
    ```

    When returning the results in the `ChartSuccess` object, the `python_codes` field must be a list of Python markdown code blocks.
    """

@chart_creator_agent.output_validator
def chart_creator_agent_output_validator(ctx: RunContext[Dependencies], output: ChartResponse) -> ChartResponse:
    """
    Validates the parsed output object from the ChartAgent.
    This function is called by pydantic-ai after it attempts to parse the LLM's raw output
    into the specified output_type (SQLResponse).
    """
    if isinstance(output, ChartError):
        # If pydantic-ai already determined it's an ChartError, just return it.
        print(f"ChartAgent Result Validator: Received ChartError: {output.error_message}")
        return output
    
    if isinstance(output, ChartSuccess):
        # Perform additional validation on the ChartSuccess object if needed.
        # For example, ensure critical fields are not empty or have expected formats.
        if not output.python_codes:
            print("ChartAgent Result Validator: ChartSuccess object has an empty python_codes list.")
            return ChartError(error_message="ChartSuccess object has an empty python_codes list.")
        
        for i, code in enumerate(output.python_codes):
            if not code:
                print(f"ChartAgent Result Validator: ChartSuccess object has an empty python_code at index {i}.")
                return ChartError(error_message=f"ChartSuccess object has an empty python_code at index {i}.")
        
        print("ChartAgent Result Validator: ChartSuccess object passed custom validation.")
        return output
    
def main():
    sql_result = """
                {
                    "name": {
                        "0": "Business Performance",
                        "1": "Benefits Realization",
                        "2": "Risks & Roadblocks",
                        "3": "Amount of Change",
                        "4": "Pace of Change",
                        "5": "Stage of Change",
                        "6": "Vision & Direction",
                        "7": "Communication",
                        "8": "Business Leadership",
                        "9": "Team Leadership",
                        "10": "Skills & Staffing",
                        "11": "Systems & Processes",
                        "12": "Teamwork",
                        "13": "Accountability",
                        "14": "Passion & Drive",
                        "15": "Fear & Frustration",
                        "16": "Issues & Obstacles",
                        "17": "sbp|bp less now",
                        "18": "sbp|bp same",
                        "19": "sbp|bp more now",
                        "20": "Changes Taking Place",
                        "21": "Benefits All",
                        "22": "Confidence Success",
                        "23": "New Benefits Realization",
                        "24": "New Benefits Realization",
                        "25": "Benefits",
                        "26": "ABCD TEst"
                    },
                    "cluster_id": {
                        "0": 7,
                        "1": 18,
                        "2": 41,
                        "3": 30,
                        "4": 596,
                        "5": 597,
                        "6": 59,
                        "7": 64,
                        "8": 69,
                        "9": 77,
                        "10": 52,
                        "11": 55,
                        "12": 87,
                        "13": 95,
                        "14": 101,
                        "15": 104,
                        "16": 42,
                        "17": 8,
                        "18": 9,
                        "19": 10,
                        "20": 29,
                        "21": 648,
                        "22": 663,
                        "23": 668,
                        "24": 669,
                        "25": 693,
                        "26": 1768
                    },
                    "product_id": {
                        "0": 3,
                        "1": 3,
                        "2": 3,
                        "3": 3,
                        "4": 3,
                        "5": 3,
                        "6": 3,
                        "7": 3,
                        "8": 3,
                        "9": 3,
                        "10": 3,
                        "11": 3,
                        "12": 3,
                        "13": 3,
                        "14": 3,
                        "15": 3,
                        "16": 3,
                        "17": 3,
                        "18": 3,
                        "19": 3,
                        "20": 3,
                        "21": 3,
                        "22": 3,
                        "23": 3,
                        "24": 3,
                        "25": 3,
                        "26": 3
                    }
                }
                """

    user_query = "What are the names visualized in bar chart"

    deps = Dependencies(query_results=sql_result)

    result = chart_creator_agent.run_sync(user_query, deps=deps)

    print(result)


if __name__=="__main__":
    main()