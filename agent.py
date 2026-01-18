"""
LangGraph Pitch Generation Agent Workflow
A multi-agent system for generating, critiquing, and refining startup pitches
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
import operator
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ===========================
# DEFINE STATE
# ===========================

class PitchState(TypedDict):
    """State that flows through the workflow"""
    mvp_description: str
    context: str
    pitch: str
    critique: dict
    critique_count: int
    human_feedback: str
    human_approved: bool
    final_pitch: str
    messages: Annotated[list, operator.add]


# ===========================
# DEFINE TOOLS (as simple functions)
# ===========================

def web_search(query: str) -> str:
    """Search the web for information about competitors, market trends, or industry insights"""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results
    except Exception as e:
        return f"Search failed: {str(e)}"


def pitch_template_tool(pitch_type: str = "elevator") -> str:
    """Get a proven pitch template structure"""
    templates = {
        "elevator": """
        ELEVATOR PITCH TEMPLATE:
        1. Hook (1 sentence): Grab attention with the problem
        2. Solution (1-2 sentences): What you built
        3. Unique Value (1 sentence): Why you're different
        4. Traction (1 sentence): Evidence it works
        5. Ask (1 sentence): What you need
        """,
        "investor": """
        INVESTOR PITCH TEMPLATE:
        1. Problem: What pain point exists?
        2. Solution: Your product/MVP
        3. Market Size: TAM/SAM/SOM
        4. Business Model: How you make money
        5. Traction: Metrics, users, revenue
        6. Competition: Landscape and differentiation
        7. Team: Why you'll win
        8. Ask: Funding amount and use
        """,
        "demo_day": """
        DEMO DAY PITCH TEMPLATE:
        1. Opening Hook: Surprising stat or story
        2. Problem: Relatable pain point
        3. Solution Demo: Show the product
        4. Market Opportunity: Size and timing
        5. Traction: Key metrics
        6. Vision: Where you're headed
        7. Team: Quick credibility
        8. The Ask: Clear and specific
        """
    }
    return templates.get(pitch_type, templates["elevator"])


def pitch_analyzer(pitch_text: str) -> dict:
    """Analyze pitch structure and provide metrics"""
    words = pitch_text.split()
    sentences = pitch_text.split('.')
    
    analysis = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": len(words) / max(len(sentences), 1),
        "has_problem_statement": any(word in pitch_text.lower() for word in ['problem', 'challenge', 'issue', 'pain']),
        "has_solution": any(word in pitch_text.lower() for word in ['solution', 'solve', 'built', 'created']),
        "has_market": any(word in pitch_text.lower() for word in ['market', 'customers', 'users', 'billion']),
        "has_traction": any(word in pitch_text.lower() for word in ['users', 'revenue', 'growth', 'customers']),
    }
    return analysis


# ===========================
# INITIALIZE LLMs
# ===========================

# Get Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file")

# Using Groq with openai/gpt-oss-120b model for all agents
# Different temperatures for different purposes
llm_context = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    groq_api_key=groq_api_key
)

llm_generator = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.8,
    groq_api_key=groq_api_key
)

llm_critic = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    groq_api_key=groq_api_key
)

llm_refiner = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    groq_api_key=groq_api_key
)

llm_readiness = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    groq_api_key=groq_api_key
)


# ===========================
# AGENT NODES
# ===========================

def pitch_context_agent(state: PitchState) -> PitchState:
    """Research and gather context about the MVP and market"""
    
    # Use web search to gather market context
    print("[INFO] Gathering market context...")
    search_query = f"{state['mvp_description'][:100]} market analysis competitors"
    search_results = web_search(search_query)
    
    # Get pitch template
    template = pitch_template_tool("elevator")
    
    system_prompt = """You are a startup research expert. Analyze the MVP description and search results to provide comprehensive context for creating a compelling pitch.

    Provide context including:
    - Key market insights from the search results
    - Competitive landscape understanding
    - Target audience identification
    - Recommended pitch approach
    - Key value propositions to emphasize
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""MVP Description: {state['mvp_description']}

Market Research Results:
{search_results}

Pitch Template to Follow:
{template}

Based on this information, provide comprehensive context for creating a compelling pitch.""")
    ]
    
    response = llm_context.invoke(messages)
    context = response.content
    
    return {
        **state,
        "context": context,
        "messages": [AIMessage(content=f"Context gathered successfully")]
    }


def pitch_generator_agent(state: PitchState) -> PitchState:
    """Generate the initial pitch based on context"""
    system_prompt = """You are an expert pitch writer. Create a compelling, concise pitch that:
    - Clearly articulates the problem and solution
    - Highlights unique value proposition
    - Includes specific, measurable outcomes
    - Is engaging and memorable
    - Follows proven pitch structure
    
    Keep it concise (150-250 words for elevator pitch).
    Be specific, avoid jargon, and focus on impact.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
        MVP Description: {state['mvp_description']}
        
        Research Context: {state['context']}
        
        Generate a compelling pitch.
        """)
    ]
    
    response = llm_generator.invoke(messages)
    
    return {
        **state,
        "pitch": response.content,
        "messages": [AIMessage(content=f"Generated pitch")]
    }


def pitch_critic_agent(state: PitchState) -> PitchState:
    """Critically evaluate the pitch"""
    system_prompt = """You are a tough but fair pitch critic (think YC partner or top VC).
    
    Evaluate the pitch on:
    1. CLARITY (10/10): Is it immediately clear what they do?
    2. PROBLEM (10/10): Is the problem compelling and relatable?
    3. SOLUTION (10/10): Is the solution clearly explained?
    4. UNIQUENESS (10/10): What makes this different/better?
    5. TRACTION (10/10): Any proof it works?
    6. ENGAGEMENT (10/10): Is it memorable and compelling?
    
    Provide:
    - Scores for each criterion (out of 10)
    - Overall score (average)
    - Specific feedback on what's weak
    - PASS/FAIL decision (PASS if overall >= 7.5)
    
    Return in JSON format:
    {
        "scores": {"clarity": X, "problem": X, ...},
        "overall_score": X,
        "decision": "PASS" or "FAIL",
        "feedback": "detailed feedback...",
        "strengths": ["strength 1", ...],
        "weaknesses": ["weakness 1", ...]
    }
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Critique this pitch:\n\n{state['pitch']}")
    ]
    
    response = llm_critic.invoke(messages)
    
    # Parse JSON response
    try:
        critique = json.loads(response.content)
    except:
        # Fallback if not proper JSON
        critique = {
            "overall_score": 5.0,
            "decision": "FAIL",
            "feedback": response.content
        }
    
    return {
        **state,
        "critique": critique,
        "critique_count": state.get("critique_count", 0) + 1,
        "messages": [AIMessage(content=f"Critique complete: {critique['decision']}")]
    }


def pitch_refiner_agent(state: PitchState) -> PitchState:
    """Refine the pitch based on critique"""
    system_prompt = """You are a pitch refinement expert. 
    
    Take the original pitch and the critique, then create an improved version that:
    - Addresses all weaknesses mentioned
    - Maintains the strengths
    - Incorporates the feedback precisely
    - Stays concise and impactful
    
    Make substantial improvements, don't just tweak words.
    """
    
    feedback = state['critique'].get('feedback', 'No specific feedback')
    weaknesses = state['critique'].get('weaknesses', [])
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
        Original Pitch:
        {state['pitch']}
        
        Critique Feedback:
        {feedback}
        
        Weaknesses to address:
        {', '.join(weaknesses)}
        
        Create an improved version.
        """)
    ]
    
    response = llm_refiner.invoke(messages)
    
    return {
        **state,
        "pitch": response.content,
        "messages": [AIMessage(content=f"Pitch refined (iteration {state['critique_count']})")]
    }


def safe_print(text):
    """Safely print text with encoding fallback"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters with ASCII equivalents
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


def human_review_node(state: PitchState) -> PitchState:
    """Simulate human review (in production, this would wait for actual human input)"""
    safe_print("\n" + "="*60)
    safe_print("HUMAN REVIEW REQUIRED")
    safe_print("="*60)
    safe_print(f"\nCurrent Pitch:\n{state['pitch']}\n")
    safe_print(f"Critique Score: {state['critique'].get('overall_score', 'N/A')}/10")
    safe_print(f"Feedback: {state['critique'].get('feedback', 'N/A')}\n")
    
    # In production, you'd wait for actual input
    # For demo, we'll auto-approve if score > 8
    auto_approve = state['critique'].get('overall_score', 0) >= 8.0
    
    safe_print("Options: [A]pprove or [R]eject")
    user_input = input("Your decision (A/R): ").strip().upper()
    
    if user_input == 'A' or (user_input == '' and auto_approve):
        approved = True
        feedback = "Approved for final preparation"
    else:
        approved = False
        feedback = input("What should be improved? ")
    
    return {
        **state,
        "human_approved": approved,
        "human_feedback": feedback,
        "messages": [AIMessage(content=f"Human review: {'Approved' if approved else 'Rejected'}")]
    }


def pitch_readiness_agent(state: PitchState) -> PitchState:
    """Prepare final pitch with delivery notes"""
    system_prompt = """You are a pitch coach preparing the final deliverable.
    
    Create a polished final pitch package including:
    1. The final pitch (clean, ready to use)
    2. Delivery tips (tone, pacing, emphasis)
    3. Anticipated questions and suggested answers
    4. Key talking points to remember
    5. One-liner version (for quick intros)
    
    Format professionally and make it presentation-ready.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
        Approved Pitch:
        {state['pitch']}
        
        Human Feedback:
        {state['human_feedback']}
        
        Prepare the final pitch package.
        """)
    ]
    
    response = llm_readiness.invoke(messages)
    
    return {
        **state,
        "final_pitch": response.content,
        "messages": [AIMessage(content="Final pitch package ready!")]
    }


# ===========================
# ROUTING FUNCTIONS
# ===========================

def route_after_critic(state: PitchState) -> Literal["human_review", "refiner"]:
    """Route based on critic's decision"""
    decision = state['critique'].get('decision', 'FAIL')
    
    # Safety: max 5 refinement iterations
    if state['critique_count'] >= 5:
        print(f"\n[WARNING] Max iterations (5) reached. Moving to human review...")
        return "human_review"
    
    if decision == "PASS":
        return "human_review"
    else:
        print(f"\n[INFO] Refinement iteration {state['critique_count']}/5 - Sending to refiner...")
        return "refiner"


def route_after_human(state: PitchState) -> Literal["readiness", "refiner"]:
    """Route based on human decision"""
    if state.get('human_approved', False):
        return "readiness"
    else:
        return "refiner"


# ===========================
# BUILD GRAPH
# ===========================

def create_pitch_workflow():
    """Create and compile the LangGraph workflow"""
    
    workflow = StateGraph(PitchState)
    
    # Add nodes
    workflow.add_node("context", pitch_context_agent)
    workflow.add_node("generator", pitch_generator_agent)
    workflow.add_node("critic", pitch_critic_agent)
    workflow.add_node("refiner", pitch_refiner_agent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("readiness", pitch_readiness_agent)
    
    # Add edges
    workflow.set_entry_point("context")
    workflow.add_edge("context", "generator")
    workflow.add_edge("generator", "critic")
    
    # Conditional routing after critic
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "human_review": "human_review",
            "refiner": "refiner"
        }
    )
    
    # After refiner, go back to critic
    workflow.add_edge("refiner", "critic")
    
    # Conditional routing after human review
    workflow.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "readiness": "readiness",
            "refiner": "refiner"
        }
    )
    
    # End after readiness
    workflow.add_edge("readiness", END)
    
    return workflow.compile()


# ===========================
# MAIN EXECUTION
# ===========================

def run_pitch_workflow(mvp_description: str):
    """Run the complete pitch generation workflow"""
    
    # Initialize state
    initial_state = {
        "mvp_description": mvp_description,
        "context": "",
        "pitch": "",
        "critique": {},
        "critique_count": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_pitch": "",
        "messages": []
    }
    
    # Create and run workflow
    app = create_pitch_workflow()
    
    safe_print("\n[START] Starting Pitch Generation Workflow...\n")
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    safe_print("\n" + "="*60)
    safe_print("FINAL PITCH PACKAGE")
    safe_print("="*60)
    safe_print(final_state["final_pitch"])
    safe_print("\n" + "="*60)
    
    return final_state


# ===========================
# EXAMPLE USAGE
# ===========================

if __name__ == "__main__":
    # Example input
    user_input = {
        "mvp_description": """We built a Chrome extension that automatically summarizes 
        lengthy academic papers using AI. Users install the extension, navigate to any PDF 
        research paper, and click our icon. Within seconds, they get a structured summary 
        with key findings, methodology, and conclusions. We also extract and explain complex 
        figures and tables. Currently used by 2,000 graduate students across 15 universities."""
    }
    
    # Run workflow
    result = run_pitch_workflow(user_input["mvp_description"])