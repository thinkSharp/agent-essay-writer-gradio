from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from Utils import AgentState, Queries, Prompts, NodeNames
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ChatMessage
import sqlite3

class EssayWriter():
    def __init__(self, model, tavily_api_key):
        self.model = model
        
        self.prompts = Prompts
          
        self.tavily = TavilyClient(api_key= tavily_api_key)

        self.graph = self.build_agent_graph()

        

    def build_agent_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node(NodeNames.Planner, self.plan_node)
        builder.add_node(NodeNames.Research_Plan, self.research_plan_node)
        builder.add_node(NodeNames.Generate, self.generation_node)
        builder.add_node(NodeNames.Reflect, self.reflection_node)
        builder.add_node(NodeNames.Research_Critique, self.research_critique_node)

        builder.set_entry_point(NodeNames.Planner)
    
        builder.add_edge(NodeNames.Planner, NodeNames.Research_Plan)
        builder.add_edge(NodeNames.Research_Plan, NodeNames.Generate)
        builder.add_edge(NodeNames.Reflect, NodeNames.Research_Critique)
        builder.add_edge(NodeNames.Research_Critique, NodeNames.Generate)

        builder.add_conditional_edges(
            NodeNames.Generate, self.should_continue, 
            {END: END, NodeNames.Reflect: NodeNames.Reflect}
        )

        memory = SqliteSaver(conn=sqlite3.connect(':memory:', check_same_thread=False))
        graph = builder.compile( checkpointer=memory,
                                interrupt_after= [NodeNames.Planner, NodeNames.Generate,
                                                  NodeNames.Reflect, NodeNames.Research_Plan,
                                                  NodeNames.Research_Critique])
        
        return graph
    

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.prompts.PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        return {
            'plan': response.content,
            'lnode': NodeNames.Planner,
            'count':1
        }

    def research_plan_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.prompts.RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {
            'content': content,
            'queries': queries.queries,
            'lnode': NodeNames.Research_Plan,
            'count': 1
        }

    def generation_node(self, state: AgentState):
        content = '\n\n'.join(state['content'] or [])
        user_message = f"{state['task']}\n\n Here is my plan:\n\n{state['plan']}"
        messages = [
            SystemMessage(
                content=self.prompts.WRITER_PROMPT.format(content=content)
            ),
            user_message
        ]
        response = self.model.invoke(messages)
        return {
            'draft': response.content,
            'revision_number': state.get('revision_number', 1) + 1,
            'lnode' : NodeNames.Generate,
            'count' : 1
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content = self.prompts.REFLECTION_PROMPT),
            HumanMessage(content=state['draft'])
        ]
        response = self.model.invoke(messages)
        return {
            'critique': response.content,
            'lnode': NodeNames.Reflect,
            'count': 1
        }

    def research_critique_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.prompts.RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {
            'content': content,
            'lnode': NodeNames.Research_Critique,
            'count': 1
        }
    
    def should_continue(self, state: AgentState):
        if state['revision_number'] > state['max_revisions']:
            return END
        return NodeNames.Reflect
    


