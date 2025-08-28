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

from graphs.sql_insights_charts import create_sql_insights_charts_graph

from agents import master, file_reader
from agents.file_reader import file_reader_agent, FileSuccess
from agents.chart_generator import chartOptions, chart_creator_agent, ChartSuccess

from agents.master import (
                        master_agent, MasterAgentResponse,
                        SQL_AGENT,
                        FILE_AGENT,
                        BOTH_AGENT,
                        NONE
                        )
from tool_functions.file_operations import list_files


load_dotenv('env')

# connection_string = create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment')
# files = list_files(dir ="/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files")

class AllState(TypedDict):
    request: Annotated[list[AnyMessage], add_messages]

    connection_string: str
    files: str
    agent: str
    
    sql_query: Optional[str]
    detail: Optional[str]
    query_results: Optional[str]
    sql_error_message: Optional[str]

    data_insights: Optional[list[str]]
    insights_error: Optional[str]

    python_codes: Optional[list[str]]
    chart_error: Optional[str]
    code_error: Optional[str]

    file_content: Optional[str]
    summary: Optional[str]
    file_error_message: Optional[str]

def master_agent_node(state = AllState):
    master_agent_response = master_agent.run_sync(
        user_prompt=state["request"][-1].content,
        deps=master.MasterDependencies(connection_string=state["connection_string"], available_files=list_files(dir =state["files"]))
        )
    return {
        "agent": master_agent_response.output.agent
    }

def sql_query_creator_node(state: AllState):
    graph = create_sql_insights_charts_graph()
    outputs = graph.invoke(state)
    return outputs

def file_reader_node(state: AllState):
    file_reader_agent_response = file_reader_agent.run_sync(
        user_prompt=state["request"][-1].content,
        deps= file_reader.Dependencies(files=list_files(state["files"]))
        )
    
    if isinstance(file_reader_agent_response.output, FileSuccess):
        return {
            "file_content": file_reader_agent_response.output.file_content,
            "summary": file_reader_agent_response.output.summary,
        }
    else:
        return {
            "file_error_message": file_reader_agent_response.output.error_message
        }


def output(state: AllState):
    import json
    output_state = state.copy()
    if "connection_string" in output_state:
        del output_state["connection_string"]
    
    # Convert HumanMessage objects to their content strings for JSON serialization
    if "request" in output_state and isinstance(output_state["request"], list):
        output_state["request"] = [
            msg.content if hasattr(msg, 'content') else str(msg)
            for msg in output_state["request"]
        ]

    with open("checking_output.json", "w") as f:
        json.dump(output_state, f, indent=4)
    return state

def sql_or_file_router(state: AllState):
    if state["agent"] == BOTH_AGENT:
        return ["sql_insights_charts_graph", "file_reader_agent"]
    elif state["agent"] == SQL_AGENT:
        return ["sql_insights_charts_graph"]
    elif state["agent"] == FILE_AGENT:
        return ["file_reader_agent"]
    elif state["agent"] == NONE:
        return ["output"]
    else:
        return ["output"]

def create_graph():
    graph = StateGraph(AllState)

    graph.add_node("master", master_agent_node)
    graph.add_node("sql_insights_charts_graph", sql_query_creator_node)
    graph.add_node("file_reader_agent", file_reader_node)
    # graph.add_node("parallel_execution_node", lambda x: x) # A pass-through node
    graph.add_node("output", output)

    graph.add_conditional_edges("master",
                                sql_or_file_router,
                                {
                                    "sql_insights_charts_graph": "sql_insights_charts_graph",
                                    "file_reader_agent": "file_reader_agent",
                                    "output": "output"
                                }
    )
    
    graph.add_edge("sql_insights_charts_graph", "output")
    graph.add_edge("file_reader_agent", "output")

    graph.set_entry_point("master")

    return graph.compile()

graph = create_graph()

def main():

    from langchain_core.runnables.graph import MermaidDrawMethod

    graph_png = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER,
    )

    initial_state = {
                        "request":
                            [HumanMessage(content="What are in the sql db, the product drivers for ctre company Accenture, ctre program Transformation GPS and ctre cycle named \"Cycle 1\"? Visualize in table.")],
                        # "connection_string":
                        #     create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment'),
                        # "files": 
                        #     list_files(dir ="/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files")
                        "connection_string":
                                            """{
                                                    "host":"localhost",
                                                    "dbname":"ctre_unstable",
                                                    "user":"orgplatform",
                                                    "password":"orgplatform",
                                                    "port":5432
                                            }""",
                        "files": 
                            "/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents_Vanna/files"
                    }

    with open("graph.png", "wb") as f:
        f.write(graph_png)

    for event in graph.stream(initial_state):
        for key in event:
            print("\n-----------------------------------")
            print("Done with " + key)
            print("\n*******************************************\n")

if  __name__ == "__main__":
    main()