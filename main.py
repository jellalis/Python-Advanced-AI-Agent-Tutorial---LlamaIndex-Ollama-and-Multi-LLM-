from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool,ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context,code_parser_template
from code_reader import code_reader
import ast
import os
load_dotenv()

llm=Ollama(
    model='mistral',
    request_timeout=300.0
)



parser = LlamaParse(result_type='markdown')

file_extractor={'.pdf':parser}
documents=SimpleDirectoryReader('./data',file_extractor=file_extractor).load_data()
embed_model=resolve_embed_model('local:BAAI/bge-m3')
vector_index=VectorStoreIndex.from_documents(documents,embed_model=embed_model)
query_engine=vector_index.as_query_engine(llm=llm)




tools=[
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
        name='api_documentation',
        description='this gives documentation about code for an API.Use this for reading docs for the API',


    )
    ),
    code_reader,

]
code_llm=Ollama(model='codellama',
    request_timeout=300.0)
agent=ReActAgent.from_tools(tools,llm=code_llm,verbose=True,context=context)

class CodeOutput(BaseModel):
    code:str
    description:str
    filename:str
parser=PydanticOutputParser(CodeOutput)
json_prompt_str=parser.format(code_parser_template)
json_prompt_tmpl=PromptTemplate(json_prompt_str)
output_pipeline=QueryPipeline(chain=[json_prompt_tmpl,llm])
while(prompt:=input('Enter a prompt(q to quit :'))!='q':
    retries = 0#retry hanler (part (5)

    while retries <3 :#retry hanler (part (5)
        try:#retry hanler (part (5)
            result=agent.query(prompt)
            next_result=output_pipeline.run(response=result)
            cleaned_json=ast.literal_eval(str(next_result).replace("assistant:",""))
            break#retry hanler (part (5)
        except Exception as e:#retry hanler (part (5)
            retries+=1#retry hanler (part (5)
            print(f'Error occured, retry #{retries}',e)#retry hanler (part (5)
    if retries>=3:#retry hanler (part (5)
        print ('unable to process request,try again...')#retry hanler (part (5)
        continue#retry hanler (part (5)


    print('code generated')
    print(cleaned_json['code'])#

    print('n\nDescritption',cleaned_json['description'])
    filename=cleaned_json['filename']
    try :#saving to a file (part(6))
        with open(os.path.join('output',filename),'w') as f:#helps not to overwite something cause i created a file output and there i will put everything
            f.write(cleaned_json['code'])
            print('saved file',filename)
    except:
        print('error saving file')#



