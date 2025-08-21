import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict, Annotated, Optional
from sqlalchemy import Engine, create_engine
from langchain_core.messages import HumanMessage
import re, io
from textwrap import dedent
import pandas as pd
import plotly.express as px


from agents import sql_query_result, insights_curator, chart_generator
from agents.sql_query_result import sql_query_result_agent, PostgresConfig, SQLSuccess
from agents.insights_curator import data_insights_agent, DataframeSuccess
from agents.chart_generator import chart_creator_agent, ChartSuccess


load_dotenv()

# import logfire
# logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
# logfire.instrument_pydantic_ai() 

# db_engine = create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment')
# files = list_files(dir ="/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files")

class AllState(TypedDict):
    # request: str
    request: Annotated[list[AnyMessage], add_messages]

    connection_string: PostgresConfig
    sql_query: Optional[str]
    answer: Optional[str]
    query_results: Optional[str]
    sql_error_message: Optional[str]

    data_insights: Optional[list[str]]
    insights_error: Optional[str]

    python_codes: Optional[list[str]]
    chart_error: Optional[str] 
    code_error: Optional[str]

    insights_error: Optional[str]
    file_content: Optional[str]

def sql_query_creator_node(state: AllState):
    sql_query_agent_response = sql_query_result_agent.run_sync(
                    user_prompt=state["request"][-1].content,
                    deps=sql_query_result.Dependencies(
                                state["connection_string"]
                ))
    
    if isinstance(sql_query_agent_response.output, SQLSuccess):
        # Modify the SQL query to include additional columns for context
        # print("isinstance(sql_query_agent_response.output, SQLSuccess)")
        # print(sql_query_agent_response.output.sql_query)
        # print(sql_query_agent_response.output.answer)
        # print(sql_query_agent_response.output.query_results)
        return {
            "sql_query": sql_query_agent_response.output.sql_query,
            "answer": sql_query_agent_response.output.answer,
            "query_results": sql_query_agent_response.output.query_results,
        }
    else: # InvalidRequest
        return {
            "sql_error_message": sql_query_agent_response.output.error_message
        }
    
def data_insights_node(state: AllState):
    data_insights_agent_response = data_insights_agent.run_sync(
            user_prompt=state["request"][-1].content,
            deps=insights_curator.Dependencies( 
                                                query_results= state["query_results"]
                                                )
            )
    
    if isinstance(data_insights_agent_response.output, DataframeSuccess):
        # print(state["request"][-1].content)
        # print(data_insights_agent_response.output.data_insights)
        return {
            "data_insights": data_insights_agent_response.output.data_insights,
        }
    else: # InvalidRequest
        return {
            "insights_error": data_insights_agent_response.output.error_message
        }
    
def chart_generator_node(state: AllState):
    chart_creator_response = chart_creator_agent.run_sync(
                user_prompt=state["request"][-1].content,
                deps=chart_generator.Dependencies(query_results= state["query_results"]
                )
    )
    
    if isinstance(chart_creator_response.output, ChartSuccess):
        return {
            "python_codes": chart_creator_response.output.python_codes,
        }
    else: # InvalidRequest
        return {
            "chart_error": chart_creator_response.output.error_message
        }
    
def run_code(state: AllState):
    try:
        for python_code in state["python_codes"]:
            code_blocks = re.findall(r"```python\n(.*?)```", python_code, re.DOTALL)
            full_code = "\n".join(code_blocks)
            full_code = dedent(full_code)
            exec_globals = {"df": pd.read_json(io.StringIO(state["query_results"])), "px": px, "pd": pd}
            exec(full_code, exec_globals)
            # print(pd.read_json(io.StringIO(state["query_results"])))
            # print(full_code)
    except Exception as e:
        return {
            "code_error": str(e)
        }
    return state

def output(state: AllState):
    import json
    output_state = state.copy()
    if isinstance(output_state.get("connection_string"), PostgresConfig):
        output_state["connection_string"] = output_state["connection_string"].to_dict()
    
    # Convert HumanMessage objects to their content strings for JSON serialization
    if "request" in output_state and isinstance(output_state["request"], list):
        output_state["request"] = [
            msg.content if hasattr(msg, 'content') else str(msg)
            for msg in output_state["request"]
        ]
    # print(output_state)
    with open("graphs/checking_output_sql_insights_charts.json", "w") as f:
        json.dump(output_state, f, indent=4)
    return state

def after_chart_router(state: AllState):
    # Always proceed to run_code after chart generation if successful
    if state.get("python_codes"): # Check if python_codes were generated
        # print("run_code")
        return "run_code"
    else: # Otherwise, go to output (e.g., if chart generation failed or no codes were expected)
        return "output"

def create_sql_insights_charts_graph():
    graph = StateGraph(AllState)

    graph.add_node("sql_query_create_agent", sql_query_creator_node)
    graph.add_node("data_insights_agent", data_insights_node)
    graph.add_node("chart_generator_agent", chart_generator_node)
    graph.add_node("run_code", run_code)
    graph.add_node("output", output)

    graph.add_conditional_edges("chart_generator_agent",
                                after_chart_router,
                                {
                                    "run_code": "run_code",
                                    "output": "output"
                                }
    )
    graph.add_edge("sql_query_create_agent", "data_insights_agent")
    graph.add_edge("data_insights_agent", "chart_generator_agent")
    graph.add_edge("run_code", "output")
    graph.add_edge("output", END)

    graph.set_entry_point("sql_query_create_agent")

    return graph.compile()

graph = create_sql_insights_charts_graph()

def main():

    from langchain_core.runnables.graph import MermaidDrawMethod

    graph_png = graph.get_graph().draw_mermaid_png(output_file_path="graphs/sql_query_create_agent.png",
        draw_method=MermaidDrawMethod.PYPPETEER,
    )

    initial_state = {
                        "request":
                            [HumanMessage(content="How many drivers of company Accenture for each cycle? visualize in bar chart and line chart")],
                        "connection_string":
                            PostgresConfig( host='localhost',
                                            dbname='ctre_unstable',
                                            user='orgplatform',
                                            password='orgplatform',
                                            port=5432 ),
                        "files": 
                            "/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files"
                    }

    # with open("graph.png", "wb") as f:
    #     f.write(graph_png)

    for event in graph.stream(initial_state):
        for key in event:
            print("\n-----------------------------------")
            print("Done with " + key)
            print("\n*******************************************\n")
    # graph.invoke(initial_state)


if  __name__ == "__main__":
    main()