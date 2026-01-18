"""
FastAPI Deployment for Pitch Generation Agent
Step-by-step execution with manual approval at each stage
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime

# LangGraph and LLM imports
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
import operator
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ===========================
# FASTAPI APP SETUP
# ===========================

app = FastAPI(
    title="Pitch Generation Agent API",
    description="AI-powered pitch generation with step-by-step approval",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# PYDANTIC MODELS
# ===========================

class PitchInput(BaseModel):
    mvp_description: str

class ApprovalDecision(BaseModel):
    approved: bool  # True = approve, False = reject and refine
    feedback: Optional[str] = ""

class SessionStatus(str, Enum):
    INITIALIZED = "initialized"
    CONTEXT_GATHERED = "context_gathered"
    PITCH_GENERATED = "pitch_generated"
    PITCH_CRITIQUED = "pitch_critiqued"
    AWAITING_APPROVAL = "awaiting_approval"
    REFINING = "refining"
    APPROVED = "approved"
    COMPLETED = "completed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

# ===========================
# IN-MEMORY SESSION STORE
# ===========================

sessions: Dict[str, Dict[str, Any]] = {}

# ===========================
# TOOLS
# ===========================

def web_search(query: str) -> str:
    """Search the web for information"""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results[:1000]  # Limit results
    except Exception as e:
        return f"Market research for similar products and competitors"

def pitch_template_tool() -> str:
    """Get pitch template"""
    return """
    PITCH STRUCTURE:
    1. Hook: Grab attention with the problem
    2. Solution: What you built and how it works
    3. Unique Value: Why you're different
    4. Traction: Evidence it works
    5. Ask: What you need
    """

# ===========================
# INITIALIZE LLMs
# ===========================

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

llm_context = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, groq_api_key=groq_api_key)
llm_generator = ChatGroq(model="openai/gpt-oss-120b", temperature=0.8, groq_api_key=groq_api_key)
llm_critic = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, groq_api_key=groq_api_key)
llm_refiner = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, groq_api_key=groq_api_key)
llm_readiness = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5, groq_api_key=groq_api_key)

# ===========================
# CORE WORKFLOW FUNCTIONS
# ===========================

def gather_context(mvp_description: str) -> str:
    """Step 1: Gather context"""
    search_query = f"{mvp_description[:100]} market analysis"
    search_results = web_search(search_query)
    template = pitch_template_tool()
    
    system_prompt = """You are a startup research expert. Analyze the MVP and provide context for creating a compelling pitch."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""MVP: {mvp_description}

Search Results: {search_results}

Template: {template}

Provide comprehensive context including market insights, target audience, and key value propositions.""")
    ]
    
    response = llm_context.invoke(messages)
    return response.content

def generate_pitch(mvp_description: str, context: str) -> str:
    """Step 2: Generate pitch"""
    system_prompt = """Create a compelling, concise pitch (150-250 words) that:
    - Clearly articulates the problem and solution
    - Highlights unique value proposition
    - Includes specific outcomes
    - Is engaging and memorable"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"MVP: {mvp_description}\n\nContext: {context}\n\nGenerate a compelling pitch.")
    ]
    
    response = llm_generator.invoke(messages)
    return response.content

def critique_pitch(pitch: str) -> dict:
    """Step 3: Critique pitch"""
    system_prompt = """Evaluate the pitch on 6 criteria (each out of 10):
    1. CLARITY: Is it immediately clear what they do?
    2. PROBLEM: Is the problem compelling?
    3. SOLUTION: Is the solution clearly explained?
    4. UNIQUENESS: What makes this different?
    5. TRACTION: Any proof it works?
    6. ENGAGEMENT: Is it memorable?
    
    Return ONLY valid JSON (no markdown, no backticks):
    {
        "scores": {"clarity": X, "problem": X, "solution": X, "uniqueness": X, "traction": X, "engagement": X},
        "overall_score": X.X,
        "decision": "PASS or FAIL",
        "feedback": "detailed feedback",
        "strengths": ["strength 1", "strength 2"],
        "weaknesses": ["weakness 1", "weakness 2"]
    }
    
    PASS if overall_score >= 7.5, otherwise FAIL."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Critique this pitch:\n\n{pitch}")
    ]
    
    response = llm_critic.invoke(messages)
    
    try:
        # Clean up response
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        critique = json.loads(content)
        return critique
    except Exception as e:
        print(f"Error parsing critique: {e}")
        print(f"Response: {response.content}")
        return {
            "overall_score": 6.0,
            "decision": "FAIL",
            "feedback": f"Could not parse critique properly. Raw response: {response.content[:200]}",
            "scores": {},
            "strengths": [],
            "weaknesses": ["Needs improvement"]
        }

def refine_pitch(original_pitch: str, critique: dict, user_feedback: str = "") -> str:
    """Step 4: Refine pitch based on critique"""
    system_prompt = """You are a pitch refinement expert. Improve the pitch by addressing all weaknesses."""
    
    feedback = critique.get('feedback', '')
    weaknesses = critique.get('weaknesses', [])
    
    user_note = f"\n\nUser Feedback: {user_feedback}" if user_feedback else ""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Original Pitch:
{original_pitch}

Critique Feedback:
{feedback}

Weaknesses to address:
{', '.join(weaknesses)}{user_note}

Create a substantially improved version.""")
    ]
    
    response = llm_refiner.invoke(messages)
    return response.content

def prepare_final_pitch(pitch: str, user_feedback: str = "") -> dict:
    """Step 5: Prepare final pitch package in structured JSON format"""
    system_prompt = """Create a comprehensive final pitch package. Return ONLY valid JSON (no markdown, no backticks) with this structure:

{
  "elevator_pitch": "One sentence pitch (30-40 words)",
  "executive_summary": "2-3 paragraph overview",
  "problem_statement": "Clear description of the problem",
  "solution": "How your product solves it",
  "unique_value_proposition": "What makes you different",
  "traction_metrics": {
    "users": "number",
    "revenue": "amount",
    "growth": "percentage",
    "other_metrics": ["metric1", "metric2"]
  },
  "market_opportunity": {
    "tam": "Total addressable market",
    "sam": "Serviceable addressable market",
    "target_segment": "Who you're targeting"
  },
  "business_model": {
    "revenue_streams": ["stream1", "stream2"],
    "pricing": "pricing strategy",
    "unit_economics": "CAC, LTV, margins"
  },
  "competitive_advantage": ["advantage1", "advantage2", "advantage3"],
  "team_highlights": "Brief team credentials",
  "funding_ask": {
    "amount": "how much",
    "use_of_funds": {
      "category1": "percentage",
      "category2": "percentage"
    },
    "milestones": ["milestone1", "milestone2"]
  },
  "key_talking_points": ["point1", "point2", "point3"],
  "anticipated_questions": [
    {
      "question": "question text",
      "answer": "concise answer"
    }
  ],
  "delivery_tips": {
    "tone": "recommended tone",
    "pacing": "timing guidance",
    "emphasis_points": ["what to emphasize"]
  }
}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Approved Pitch:\n{pitch}\n\nUser Notes: {user_feedback}\n\nCreate structured final pitch package in JSON format.")
    ]
    
    response = llm_readiness.invoke(messages)
    
    try:
        # Clean up response
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        final_pitch_json = json.loads(content)
        return final_pitch_json
    except Exception as e:
        print(f"Error parsing final pitch JSON: {e}")
        # Fallback structure
        return {
            "elevator_pitch": pitch[:200],
            "executive_summary": pitch,
            "problem_statement": "See full pitch content",
            "solution": "See full pitch content",
            "unique_value_proposition": "See full pitch content",
            "traction_metrics": {},
            "market_opportunity": {},
            "business_model": {},
            "competitive_advantage": [],
            "team_highlights": "",
            "funding_ask": {},
            "key_talking_points": [],
            "anticipated_questions": [],
            "delivery_tips": {},
            "raw_pitch": pitch
        }

# ===========================
# API ENDPOINTS
# ===========================

@app.post("/api/pitch/start")
async def start_pitch_workflow(pitch_input: PitchInput):
    """Step 1: Start workflow and generate initial pitch"""
    session_id = str(uuid.uuid4())
    
    # Gather context
    print(f"[{session_id}] Gathering context...")
    context = gather_context(pitch_input.mvp_description)
    
    # Generate initial pitch
    print(f"[{session_id}] Generating pitch...")
    pitch = generate_pitch(pitch_input.mvp_description, context)
    
    # Store session
    sessions[session_id] = {
        'mvp_description': pitch_input.mvp_description,
        'context': context,
        'pitch': pitch,
        'critique': {},
        'iteration_count': 0,
        'critic_fail_count': 0,
        'status': SessionStatus.PITCH_GENERATED,
        'created_at': datetime.now().isoformat()
    }
    
    # Enter the critique-refine loop
    return await _run_critique_refine_loop(session_id)


async def _run_critique_refine_loop(session_id: str):
    """Internal function to handle critic-refiner loop until PASS or max attempts"""
    session = sessions[session_id]
    max_auto_refine_attempts = 3
    
    while session['critic_fail_count'] < max_auto_refine_attempts:
        # Critique the current pitch
        print(f"[{session_id}] Critiquing pitch (attempt {session['critic_fail_count'] + 1})...")
        critique = critique_pitch(session['pitch'])
        
        session['critique'] = critique
        session['iteration_count'] += 1
        
        # Check if critic passed
        if critique.get('decision') == 'PASS':
            session['status'] = SessionStatus.AWAITING_APPROVAL
            print(f"[{session_id}] Critic PASSED! Sending to human for approval.")
            
            return {
                "session_id": session_id,
                "status": session['status'],
                "pitch": session['pitch'],
                "critique": critique,
                "iteration_count": session['iteration_count'],
                "critic_decision": "PASS",
                "message": f"Pitch PASSED critic review (iteration {session['iteration_count']})! Please review and approve."
            }
        
        # Critic failed
        session['critic_fail_count'] += 1
        print(f"[{session_id}] Critic FAILED (attempt {session['critic_fail_count']}/{max_auto_refine_attempts})")
        
        # If we've hit max auto-refine attempts, send to human
        if session['critic_fail_count'] >= max_auto_refine_attempts:
            session['status'] = SessionStatus.AWAITING_APPROVAL
            print(f"[{session_id}] Max auto-refine attempts reached. Sending to human for decision.")
            
            return {
                "session_id": session_id,
                "status": session['status'],
                "pitch": session['pitch'],
                "critique": critique,
                "iteration_count": session['iteration_count'],
                "critic_decision": "FAIL",
                "critic_fail_count": session['critic_fail_count'],
                "message": f"Pitch failed critic review {session['critic_fail_count']} times. Auto-refinement complete. Please review and decide whether to approve or manually refine."
            }
        
        # Auto-refine and loop back
        print(f"[{session_id}] Auto-refining pitch...")
        session['status'] = SessionStatus.REFINING
        refined_pitch = refine_pitch(
            session['pitch'],
            critique,
            f"Auto-refinement attempt {session['critic_fail_count']}"
        )
        session['pitch'] = refined_pitch
        
    # This should never be reached due to the check above, but just in case
    session['status'] = SessionStatus.AWAITING_APPROVAL
    return {
        "session_id": session_id,
        "status": session['status'],
        "pitch": session['pitch'],
        "critique": session['critique'],
        "iteration_count": session['iteration_count'],
        "critic_decision": "FAIL",
        "message": "Max refinement attempts reached. Please review."
    }

@app.post("/api/pitch/approve/{session_id}")
async def approve_pitch(session_id: str, decision: ApprovalDecision):
    """
    Step 2: User approves or rejects the pitch
    - If approved=True: Generate final pitch package
    - If approved=False: Manually refine with user feedback, then re-enter critic loop
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Check if status is awaiting approval
    if session['status'] != SessionStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve/reject. Current status: {session['status']}"
        )
    
    # Check max total iterations (including manual refinements)
    if session['iteration_count'] >= 10 and not decision.approved:
        return {
            "session_id": session_id,
            "status": SessionStatus.MAX_ITERATIONS_REACHED,
            "message": "Maximum 10 total iterations reached. Please approve current pitch or start a new session.",
            "pitch": session['pitch'],
            "critique": session['critique']
        }
    
    # User APPROVED - prepare final pitch
    if decision.approved:
        print(f"[{session_id}] User approved! Preparing final pitch...")
        session['status'] = SessionStatus.APPROVED
        
        final_pitch = prepare_final_pitch(session['pitch'], decision.feedback)
        
        session['final_pitch'] = final_pitch
        session['status'] = SessionStatus.COMPLETED
        
        return {
            "session_id": session_id,
            "status": SessionStatus.COMPLETED,
            "final_pitch_package": final_pitch,
            "total_iterations": session['iteration_count'],
            "message": "Pitch approved! Final package ready."
        }
    
    # User REJECTED - manually refine with their feedback, then re-enter critic loop
    else:
        print(f"[{session_id}] User rejected. Manual refinement with feedback...")
        session['status'] = SessionStatus.REFINING
        
        # Refine pitch with user's specific feedback
        refined_pitch = refine_pitch(
            session['pitch'],
            session['critique'],
            decision.feedback
        )
        
        # Update session
        session['pitch'] = refined_pitch
        session['critic_fail_count'] = 0  # Reset auto-refine counter for new manual iteration
        
        # Re-enter the critic-refiner loop
        return await _run_critique_refine_loop(session_id)

@app.get("/api/pitch/status/{session_id}")
async def get_status(session_id: str):
    """Get current session status"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": session['status'],
        "iteration_count": session['iteration_count'],
        "critic_fail_count": session.get('critic_fail_count', 0),
        "current_pitch": session.get('pitch'),
        "critique": session.get('critique'),
        "final_pitch": session.get('final_pitch'),
        "created_at": session['created_at']
    }

@app.get("/api/pitch/final/{session_id}")
async def get_final_pitch(session_id: str):
    """Get final pitch package in structured JSON format (only available after approval)"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session['status'] != SessionStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Final pitch not ready. Current status: {session['status']}"
        )
    
    return {
        "session_id": session_id,
        "final_pitch_package": session['final_pitch'],
        "total_iterations": session['iteration_count'],
        "metadata": {
            "mvp_description": session['mvp_description'],
            "created_at": session['created_at'],
            "final_critique_score": session.get('critique', {}).get('overall_score', 0)
        }
    }

@app.delete("/api/pitch/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/")
async def root():
    """API Documentation"""
    return {
        "message": "Pitch Generation Agent API v2.0",
        "workflow": {
            "step_1": "POST /api/pitch/start - Submit MVP description, get pitch + critique",
            "step_2": "POST /api/pitch/approve/{session_id} - Approve (get final) or Reject (refine)",
            "step_3": "GET /api/pitch/final/{session_id} - Get final pitch package"
        },
        "endpoints": {
            "start": "POST /api/pitch/start",
            "approve_reject": "POST /api/pitch/approve/{session_id}",
            "status": "GET /api/pitch/status/{session_id}",
            "final": "GET /api/pitch/final/{session_id}",
            "delete": "DELETE /api/pitch/session/{session_id}"
        },
        "example_flow": {
            "1": "POST /api/pitch/start with {'mvp_description': '...'}",
            "2": "Review pitch and critique from response",
            "3a": "POST /api/pitch/approve/{session_id} with {'approved': true} to get final",
            "3b": "POST /api/pitch/approve/{session_id} with {'approved': false, 'feedback': '...'} to refine",
            "4": "Repeat step 2-3 up to 5 times",
            "5": "GET /api/pitch/final/{session_id} when status is 'completed'"
        }
    }

@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "status": session['status'],
                "iteration_count": session['iteration_count'],
                "created_at": session['created_at']
            }
            for sid, session in sessions.items()
        ]
    }

# ===========================
# RUN SERVER
# ===========================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("PITCH GENERATION AGENT API")
    print("="*60)
    print("\nStarting server on http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("\nWorkflow:")
    print("1. POST /api/pitch/start - Generate pitch + get critique")
    print("2. POST /api/pitch/approve/{session_id} - Approve or Reject")
    print("3. GET /api/pitch/final/{session_id} - Get final pitch\n")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)