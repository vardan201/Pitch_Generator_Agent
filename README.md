# Pitch Generation Agent

An AI-powered multi-agent system for generating, critiquing, and refining startup pitches using LangGraph and LLM agents.

## Overview

This project provides two implementations:
1. **CLI Version** (`pitch_agent.py`) - Interactive command-line workflow with manual review
2. **API Version** (`pitch_api.py`) - FastAPI-based REST API with step-by-step approval

The system uses multiple specialized AI agents that work together to create compelling startup pitches through an iterative refinement process.

## Features

- **Multi-Agent Architecture**: Specialized agents for research, generation, critique, refinement, and finalization
- **Automated Critique Loop**: Pitch automatically refined up to 3 times based on AI critic feedback
- **Human-in-the-Loop**: Manual review and approval at key decision points
- **Web Research Integration**: Automatic market and competitor research using DuckDuckGo
- **Structured Output**: Comprehensive final pitch package with delivery tips, Q&A, and talking points
- **Session Management**: Track multiple pitch generation workflows simultaneously (API version)

## Architecture

### Agent Workflow

```
Start → Context Agent → Generator Agent → Critic Agent
                                              ↓
                                    ┌─── PASS? ───┐
                                    ↓             ↓
                              Human Review    Refiner Agent
                                    ↓             ↓
                              Approved?      (Back to Critic)
                                    ↓
                            Readiness Agent → Final Pitch
```

### Specialized Agents

1. **Context Agent** - Researches market, competitors, and gathers insights
2. **Generator Agent** - Creates initial pitch based on MVP description and context
3. **Critic Agent** - Evaluates pitch on 6 criteria (clarity, problem, solution, uniqueness, traction, engagement)
4. **Refiner Agent** - Improves pitch based on critique feedback
5. **Readiness Agent** - Prepares final pitch package with delivery notes

## Installation

### Prerequisites

- Python 3.8+
- Groq API Key (free tier available at [console.groq.com](https://console.groq.com))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pitch-generation-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file in project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Requirements.txt

```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
langgraph==0.2.0
langchain-groq==0.1.0
langchain-core==0.3.0
langchain-community==0.3.0
duckduckgo-search==6.0.0
python-dotenv==1.0.0
```

## Usage

### CLI Version

Run the interactive command-line version:

```bash
python pitch_agent.py
```

The workflow will:
1. Gather market context automatically
2. Generate initial pitch
3. Critique and auto-refine up to 5 times
4. Prompt for human approval
5. Prepare final pitch package

**Example Input:**
```python
mvp_description = """We built a Chrome extension that automatically summarizes 
lengthy academic papers using AI. Users install the extension, navigate to any PDF 
research paper, and click our icon. Within seconds, they get a structured summary 
with key findings, methodology, and conclusions. We also extract and explain complex 
figures and tables. Currently used by 2,000 graduate students across 15 universities."""
```

### API Version

Start the FastAPI server:

```bash
python pitch_api.py
```

Server runs on `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

#### API Workflow

**Step 1: Start Pitch Generation**
```bash
curl -X POST http://localhost:8000/api/pitch/start \
  -H "Content-Type: application/json" \
  -d '{
    "mvp_description": "Your MVP description here..."
  }'
```

Response includes:
- `session_id` - Track your workflow
- `pitch` - Generated pitch
- `critique` - AI critic evaluation
- `critic_decision` - PASS or FAIL
- `iteration_count` - Current iteration number

**Step 2: Approve or Reject**

If satisfied with pitch:
```bash
curl -X POST http://localhost:8000/api/pitch/approve/{session_id} \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "feedback": "Looks great!"
  }'
```

If you want refinement:
```bash
curl -X POST http://localhost:8000/api/pitch/approve/{session_id} \
  -H "Content-Type: application/json" \
  -d '{
    "approved": false,
    "feedback": "Focus more on the market opportunity and add specific revenue projections"
  }'
```

**Step 3: Get Final Pitch Package**
```bash
curl http://localhost:8000/api/pitch/final/{session_id}
```

#### Additional API Endpoints

```bash
# Check session status
GET /api/pitch/status/{session_id}

# List all sessions
GET /api/sessions

# Delete session
DELETE /api/pitch/session/{session_id}
```

## API Response Structure

### Final Pitch Package (JSON)

```json
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
  "competitive_advantage": ["advantage1", "advantage2"],
  "team_highlights": "Brief team credentials",
  "funding_ask": {
    "amount": "how much",
    "use_of_funds": {
      "category1": "percentage",
      "category2": "percentage"
    },
    "milestones": ["milestone1", "milestone2"]
  },
  "key_talking_points": ["point1", "point2"],
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
}
```

## Critique Scoring System

Each pitch is evaluated on 6 criteria (scored 0-10):

- **Clarity** - Is it immediately clear what the product does?
- **Problem** - Is the problem compelling and relatable?
- **Solution** - Is the solution clearly explained?
- **Uniqueness** - What makes this different/better than alternatives?
- **Traction** - Is there proof that it works?
- **Engagement** - Is it memorable and compelling?

**Decision Logic:**
- **PASS**: Overall score ≥ 7.5/10 → Sends to human review
- **FAIL**: Overall score < 7.5/10 → Auto-refines (up to 3 times)

## Workflow Limits

- **Auto-refinement**: Maximum 3 automatic iterations
- **Total iterations**: Maximum 10 (including manual refinements)
- **Purpose**: Prevent infinite loops while ensuring quality

## Configuration

### LLM Models

The system uses Groq's `openai/gpt-oss-120b` model with different temperature settings for each agent:

```python
llm_context = ChatGroq(temperature=0.7)      # Balanced research
llm_generator = ChatGroq(temperature=0.8)    # Creative generation
llm_critic = ChatGroq(temperature=0.3)       # Precise evaluation
llm_refiner = ChatGroq(temperature=0.7)      # Thoughtful refinement
llm_readiness = ChatGroq(temperature=0.5)    # Structured output
```

### Customization

Modify pitch templates in `pitch_template_tool()` function to customize for:
- Elevator pitches
- Investor presentations
- Demo day pitches
- Custom formats

## Troubleshooting

### Common Issues

**"GROQ_API_KEY not found"**
- Ensure `.env` file exists in project root
- Verify API key is correctly formatted

**Unicode/Encoding errors (Windows)**
- The CLI version includes automatic UTF-8 encoding fixes
- If issues persist, run in PowerShell instead of CMD

**Web search failures**
- DuckDuckGo search may occasionally timeout
- System includes fallback handling
- Consider rate limiting for production use

**JSON parsing errors**
- LLM responses are cleaned automatically
- Check `pitch_api.py` for fallback handling examples

## Development

### Session Storage

Current implementation uses in-memory dictionary storage:
```python
sessions: Dict[str, Dict[str, Any]] = {}
```

For production, consider:
- Redis for distributed systems
- PostgreSQL for persistence
- MongoDB for document storage

### Adding New Agents

1. Create agent function following pattern:
```python
def new_agent(state: PitchState) -> PitchState:
    # Agent logic here
    return {**state, "new_field": value}
```

2. Add to workflow graph:
```python
workflow.add_node("new_agent", new_agent)
workflow.add_edge("previous_node", "new_agent")
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional pitch templates
- More sophisticated critique metrics
- Integration with presentation tools
- A/B testing framework
- Analytics dashboard

## License

MIT License - feel free to use and modify for your projects.

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [Groq](https://groq.com) - Fast LLM inference
- [FastAPI](https://fastapi.tiangolo.com) - API framework

## Support

For issues or questions:
1. Check API documentation at `/docs`
2. Review error messages in console output
3. Verify environment variables are set correctly
4. Ensure all dependencies are installed

---

**Happy Pitching! 🚀**
