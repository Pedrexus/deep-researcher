import json
import re
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from agents.configuration import Configuration
from agents.enums import SearchAPI
from agents.utils import deduplicate_and_format_sources, tavily_search, format_sources
from agents.states import SummaryState, SummaryStateInput, SummaryStateOutput
from langgraph.store.memory import InMemoryStore


def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """
    c = Configuration.from_store()
    prompt = c.prompts["query_writing"].format(research_topic=state.research_topic)
    
    llm_json_mode = ChatOllama(model=c.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate a query for web search:")
    ])
    query = json.loads(result.content)
    
    return {"search_query": query['query']}

def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """
    c = Configuration.from_store()

    if c.search_api == SearchAPI.TAVILY:
        search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    else:
        raise ValueError(f"Unsupported search API: {c.search_api}")
        
    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str]
    }

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """
    c = Configuration.from_store()
    
    # we format a message with all the current data for the model to summarize
    #  if this is the first iteration, we don't have a running summary yet
    search_and_summary_results = c.prompts["search_and_summary_results"].format(
             research_topic=state.research_topic, 
             existing_summary=state.running_summary,
             most_recent_web_research=state.web_research_results[-1]
        ).replace("\n", "").replace("<Existing Summary><Existing Summary>", "")

    llm = ChatOllama(model=c.local_llm, temperature=0)
    result = llm.invoke(
        [SystemMessage(content=c.prompts["summarization"]),
         HumanMessage(content=search_and_summary_results)]
    )

    running_summary = re.sub(r'<think>.*?</think>', '', result.content, flags=re.DOTALL)
    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """
    c = Configuration.from_store()
    
    llm_json_mode = ChatOllama(model=c.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=c.prompts["reflection_system"].format(research_topic=state.research_topic)),
         HumanMessage(content=c.prompts["reflection_human"].format(running_summary=state.running_summary))]
    )

    if query := json.loads(result.content).get('follow_up_query'):
        return {"search_query": query}
    return {"search_query": f"Tell me more about {state.research_topic}"}


def finalize_summary(state: SummaryState):
    """ Finalize the summary """
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """
    c = Configuration.from_store()
    return "web_research" if state.research_loop_count <= c.max_web_research_loops else "finalize_summary"
    
# Build the graph using the base configuration from config.yml
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

store = InMemoryStore()
graph = builder.compile(store=store)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    for event in graph.stream(
        input=SummaryStateInput(research_topic="Quantum Computing"), 
        config={"configurable": {"thread_id": "1"}}, 
        stream_mode="values"
    ):
        print(event)
        # event["messages"][-1].pretty_print()