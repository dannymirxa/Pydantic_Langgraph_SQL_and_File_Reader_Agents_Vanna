import os
import json

from pypdf import PdfReader
import pandas as pd

def list_files(dir: str) -> list[str]:
    # return os.listdir(dir)
    
    full_paths = [os.path.join(dir, f) for f in os.listdir(dir)]

    return full_paths


MAX_FILE_CONTENT_LENGTH = 100  # Max characters for file content

def read_pdf(file_path: str) -> dict:
    with PdfReader(file_path) as file:
        number_of_pages = len(file.pages)
        texts = [file.pages[page].extract_text() for page in range(number_of_pages)]
        full_content = "\n".join(texts)
        
        summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
        
        return {
            "file_content": full_content[:MAX_FILE_CONTENT_LENGTH],
            "summary": summary
        }
    
def read_txt(file_path: str) -> dict:
    with open(file_path, "r") as file:
        full_content = file.read()
        summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
        return {
            "file_content": full_content[:MAX_FILE_CONTENT_LENGTH],
            "summary": summary
        }

def read_csv(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    full_content = df.to_json()
    summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
    return {
        "file_content": full_content[:MAX_FILE_CONTENT_LENGTH],
        "summary": summary
    }

def read_json(file_path: str) -> dict:
    with open(file_path, "r") as file:
        full_content = json.dumps(json.load(file))
        summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
        return {
            "file_content": full_content[:MAX_FILE_CONTENT_LENGTH],
            "summary": summary
        }

if __name__=="__main__":
    print(list_files(dir ="/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files"))
    # print(read_pdf("/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files/Specific Agents Versus Generality.pdf"))
    # print(read_txt(file= "/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files/risc.txt"))
    # print(read_csv(file="/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files/ai_job_dataset.csv"))
    # print(read_json(file="/mnt/c/Projects/Pydantic_Langgraph_SQL_and_File_Reader_Agents/files/structured_bike_data_cleaned.json"))