# Two-Phase Reflective Agent Architecture

## Overview

This document describes the enhanced architecture for the RAG-MCP agent system that incorporates self-reflection and reasoning capabilities to meet capstone project requirements.

## Background & Motivation

### Current System Limitations

The existing RAG-MCP agent follows a simple pipeline:
```
User Query → MCP Search → Build Context → LLM → Output
```

**What it does well:**
- ✅ Semantic search via MCP tools
- ✅ LLM-based code recommendations
- ✅ PostgreSQL vector database integration

**What's missing:**
- ❌ No explicit reasoning/thinking process
- ❌ No reflection on its own decisions
- ❌ No self-inspection of its own code
- ❌ Limited transparency in decision-making

### Capstone Requirements

Per project feedback, the agent must demonstrate:
1. **Reasoning** - Show explicit thinking and planning
2. **Tool-based Actions** - Execute meaningful operations (we have this via MCP)
3. **Reflection** - Critique its own performance and decisions
4. **Self-Inspection** - Analyze its own source code and suggest improvements

## Proposed Architecture: Two-Phase Reflective Agent

The agent operates in two distinct phases that emphasize different cognitive capabilities:

### Phase 1: Execution (Reasoning → Action → Analysis)

This phase handles the core task execution with explicit reasoning.

```
User Query
    ↓
[1.1 REASONING]
    - Understand the user's intent
    - Plan approach and tool usage
    - Identify potential challenges
    - Define success criteria
    ↓
[1.2 ACTION]
    - Execute planned MCP tool calls
    - Log each action with rationale
    - Collect search results
    - Gather relevant context
    ↓
[1.3 ANALYSIS]
    - Evaluate quality of results
    - Assess relevance to query
    - Identify information gaps
    - Synthesize findings
```

### Phase 2: Meta-Reflection (Reflection → Self-Inspection)

This phase adds self-awareness and continuous improvement capability.

```
[2.1 REFLECTION]
    - Critique the execution phase
    - Evaluate decision quality
    - Rate confidence level
    - Identify what could be better
    - Decide if retry is needed
    ↓
[2.2 SELF-INSPECTION] (on-demand)
    - Read own source code
    - Analyze implementation patterns
    - Identify potential bugs
    - Suggest architectural improvements
    - Recommend optimizations
    ↓
Final Output (with full transparency)
```

## Detailed Phase Descriptions

### Phase 1.1: Reasoning

**Purpose:** Make the agent's thinking process explicit and deliberate.

**Implementation:**
```python
async def reasoning_phase(self, query: str) -> ReasoningOutput:
    """
    Agent thinks through the problem before taking action.
    Returns structured reasoning output.
    """
    reasoning_prompt = f"""
    Given user query: "{query}"

    Think step-by-step:
    1. What is the user really asking for?
    2. What information do I need to gather?
    3. Which MCP tools should I use and in what order?
    4. What are potential challenges or edge cases?
    5. How will I evaluate if my answer is good?
    6. What would success look like?

    Provide structured reasoning with clear decision rationale.
    """

    reasoning = await self.llm.generate(reasoning_prompt)

    return ReasoningOutput(
        user_intent=...,
        planned_actions=[...],
        tool_sequence=[...],
        expected_challenges=[...],
        success_criteria=[...]
    )
```

**Key Benefits:**
- Makes decision-making transparent
- Helps identify the right tools before execution
- Provides audit trail of agent's thought process
- Enables better error handling

### Phase 1.2: Action

**Purpose:** Execute planned actions with explicit logging and rationale.

**Implementation:**
```python
async def action_phase(self, reasoning: ReasoningOutput) -> ActionOutput:
    """
    Execute planned actions with detailed tracking.
    Each action is logged with its purpose and outcome.
    """
    actions_taken = []

    # Action 1: Search relevant modules
    if "search_modules" in reasoning.tool_sequence:
        modules = await self.mcp.search_modules(
            query=reasoning.user_intent,
            top_k=5
        )
        actions_taken.append(Action(
            name="search_modules",
            parameters={"top_k": 5},
            rationale="Find high-level code organization",
            result_count=len(modules),
            quality_score=self._assess_results(modules)
        ))

    # Action 2: Search specific files
    if "search_files" in reasoning.tool_sequence:
        files = await self.mcp.search_files(
            query=reasoning.user_intent,
            top_k=10
        )
        actions_taken.append(Action(
            name="search_files",
            parameters={"top_k": 10},
            rationale="Locate specific implementation files",
            result_count=len(files),
            quality_score=self._assess_results(files)
        ))

    # Action 3: Search historical tasks
    if "search_similar_tasks" in reasoning.tool_sequence:
        tasks = await self.mcp.search_similar_tasks(
            query=reasoning.user_intent,
            top_k=5
        )
        actions_taken.append(Action(
            name="search_similar_tasks",
            parameters={"top_k": 5},
            rationale="Learn from past solutions",
            result_count=len(tasks),
            quality_score=self._assess_results(tasks)
        ))

    return ActionOutput(
        actions=actions_taken,
        total_results=sum(a.result_count for a in actions_taken),
        execution_time=...,
        encountered_errors=[...]
    )
```

**Key Benefits:**
- Clear traceability of all operations
- Each action has documented purpose
- Quality assessment per action
- Easy to debug and optimize

### Phase 1.3: Analysis

**Purpose:** Evaluate the quality and relevance of gathered information.

**Implementation:**
```python
async def analysis_phase(
    self,
    reasoning: ReasoningOutput,
    actions: ActionOutput
) -> AnalysisOutput:
    """
    Analyze search results and assess their quality.
    """
    analysis_prompt = f"""
    Original Intent: {reasoning.user_intent}
    Actions Taken: {actions.summary()}
    Results Gathered: {actions.total_results} items

    Analyze:
    1. Are the results relevant to the user's query?
    2. What's the quality of information gathered?
    3. Are there any gaps in the information?
    4. Do we have enough context to provide a good answer?
    5. Should we perform additional searches?

    Rate relevance (1-10) and identify gaps.
    """

    analysis = await self.llm.generate(analysis_prompt)

    return AnalysisOutput(
        relevance_score=...,
        coverage_assessment=...,
        identified_gaps=[...],
        quality_rating=...,
        needs_additional_search=...,
        synthesized_context=...
    )
```

**Key Benefits:**
- Quality control before generating response
- Identifies when more information is needed
- Prevents low-quality responses
- Enables iterative refinement

### Phase 2.1: Reflection

**Purpose:** Self-critique and performance evaluation.

**Implementation:**
```python
async def reflection_phase(
    self,
    query: str,
    reasoning: ReasoningOutput,
    actions: ActionOutput,
    analysis: AnalysisOutput
) -> ReflectionOutput:
    """
    Agent reflects on its own performance and decisions.
    Critical self-assessment phase.
    """
    reflection_prompt = f"""
    You just processed this query: "{query}"

    Your reasoning: {reasoning.summary()}
    Your actions: {actions.summary()}
    Your analysis: {analysis.summary()}

    Now reflect critically:
    1. Did you choose the right approach?
    2. Were your tool selections optimal?
    3. What went well in this execution?
    4. What could have been done better?
    5. How confident are you in your results? (0-100%)
    6. Should you retry with a different strategy?
    7. What would you change if you could redo this?

    Be honest and critical. Identify weaknesses.
    """

    reflection = await self.llm.generate(reflection_prompt)

    # Parse reflection into structured format
    return ReflectionOutput(
        approach_evaluation=...,
        what_went_well=[...],
        what_could_improve=[...],
        confidence_score=...,
        should_retry=...,
        alternative_approaches=[...],
        lessons_learned=[...]
    )
```

**Key Benefits:**
- Builds self-awareness into the agent
- Identifies room for improvement
- Confidence scoring helps users trust results
- Enables retry logic with different strategies

### Phase 2.2: Self-Inspection

**Purpose:** Agent analyzes its own source code and suggests improvements.

**Implementation:**
```python
async def self_inspection_phase(
    self,
    query: str,
    trigger_reason: str = "code-related query"
) -> SelfInspectionOutput:
    """
    Agent reads and analyzes its own implementation.
    Triggered when query is about the agent itself or periodically.
    """
    # Determine which source files to inspect
    relevant_files = self._identify_relevant_source_files(query)

    inspection_results = []

    for file_path in relevant_files:
        # Read own source code
        source_code = await self._read_source_file(file_path)

        inspection_prompt = f"""
        You are analyzing your own source code.
        File: {file_path}

        Code:
        ```python
        {source_code}
        ```

        Analyze this code:
        1. What does this component do?
        2. Are there any potential bugs or issues?
        3. Is the code following best practices?
        4. Are there performance bottlenecks?
        5. How could this be improved?
        6. Are there edge cases not handled?
        7. Is error handling adequate?

        Provide constructive self-critique.
        """

        inspection = await self.llm.generate(inspection_prompt)

        inspection_results.append(FileInspection(
            file_path=file_path,
            purpose=...,
            issues_found=[...],
            improvement_suggestions=[...],
            code_quality_score=...,
            technical_debt_notes=[...]
        ))

    return SelfInspectionOutput(
        files_inspected=len(relevant_files),
        inspections=inspection_results,
        overall_health_assessment=...,
        priority_improvements=[...],
        architectural_recommendations=[...]
    )
```

**Triggers for Self-Inspection:**
- User asks about the agent's implementation
- User queries "how does this agent work?"
- Periodic health checks (every N queries)
- After performance degradation is detected
- When reflection phase identifies systemic issues

**Key Benefits:**
- Demonstrates meta-cognitive capability
- Identifies technical debt
- Self-improvement suggestions
- Transparency about implementation
- Educational for users understanding the system

## Agent Class Structure

```python
class TwoPhaseReflectiveAgent:
    """
    RAG-MCP Agent with explicit reasoning and self-reflection.

    Capabilities:
    - Phase 1: Reasoning → Action → Analysis
    - Phase 2: Reflection → Self-Inspection
    """

    def __init__(self, mcp_client, llm_client):
        self.mcp = mcp_client
        self.llm = llm_client
        self.execution_history = []

    async def process_query(
        self,
        query: str,
        enable_self_inspection: bool = None
    ) -> AgentResponse:
        """
        Main entry point for query processing.
        Returns complete response with all phases.
        """
        # PHASE 1: EXECUTION
        # 1.1 Reasoning
        reasoning = await self.reasoning_phase(query)

        # 1.2 Action
        actions = await self.action_phase(reasoning)

        # 1.3 Analysis
        analysis = await self.analysis_phase(reasoning, actions)

        # Check if retry is needed based on analysis
        if analysis.needs_additional_search:
            # Adaptive retry with refined strategy
            reasoning = await self.refine_reasoning(reasoning, analysis)
            actions = await self.action_phase(reasoning)
            analysis = await self.analysis_phase(reasoning, actions)

        # PHASE 2: META-REFLECTION
        # 2.1 Reflection
        reflection = await self.reflection_phase(
            query, reasoning, actions, analysis
        )

        # 2.2 Self-Inspection (conditional)
        self_inspection = None
        should_inspect = (
            enable_self_inspection or
            self._is_code_introspection_query(query) or
            reflection.suggests_systemic_issues
        )

        if should_inspect:
            self_inspection = await self.self_inspection_phase(
                query,
                trigger_reason="query requires self-analysis"
            )

        # Generate final response
        final_response = await self.generate_response(
            query=query,
            reasoning=reasoning,
            actions=actions,
            analysis=analysis,
            reflection=reflection,
            self_inspection=self_inspection
        )

        # Store in execution history for learning
        self.execution_history.append(ExecutionRecord(
            query=query,
            reasoning=reasoning,
            actions=actions,
            analysis=analysis,
            reflection=reflection,
            self_inspection=self_inspection,
            final_response=final_response,
            timestamp=datetime.now()
        ))

        return final_response

    async def reasoning_phase(self, query: str) -> ReasoningOutput:
        """Phase 1.1: Explicit reasoning"""
        # Implementation shown above
        pass

    async def action_phase(self, reasoning: ReasoningOutput) -> ActionOutput:
        """Phase 1.2: Deliberate action execution"""
        # Implementation shown above
        pass

    async def analysis_phase(
        self,
        reasoning: ReasoningOutput,
        actions: ActionOutput
    ) -> AnalysisOutput:
        """Phase 1.3: Result quality assessment"""
        # Implementation shown above
        pass

    async def reflection_phase(
        self,
        query: str,
        reasoning: ReasoningOutput,
        actions: ActionOutput,
        analysis: AnalysisOutput
    ) -> ReflectionOutput:
        """Phase 2.1: Self-critique"""
        # Implementation shown above
        pass

    async def self_inspection_phase(
        self,
        query: str,
        trigger_reason: str
    ) -> SelfInspectionOutput:
        """Phase 2.2: Source code self-analysis"""
        # Implementation shown above
        pass

    def _is_code_introspection_query(self, query: str) -> bool:
        """Detect if query is asking about agent's implementation"""
        introspection_keywords = [
            "how do you work",
            "how does this agent",
            "your implementation",
            "your source code",
            "how are you built",
            "explain your architecture"
        ]
        return any(kw in query.lower() for kw in introspection_keywords)
```

## Response Format

The agent returns a comprehensive, transparent response:

```python
class AgentResponse:
    """Complete agent response with full transparency"""

    # User's original query
    query: str

    # PHASE 1 outputs
    reasoning: ReasoningOutput
    actions: ActionOutput
    analysis: AnalysisOutput

    # PHASE 2 outputs
    reflection: ReflectionOutput
    self_inspection: Optional[SelfInspectionOutput]

    # Final synthesized answer
    answer: str

    # Metadata
    confidence_score: float  # 0-100
    processing_time: float
    total_tokens_used: int

    def format_for_display(self) -> str:
        """Format response for user display"""
        return f"""
        {'='*80}
        QUERY: {self.query}
        {'='*80}

        REASONING:
        {self.reasoning.format()}

        ACTIONS TAKEN:
        {self.actions.format()}

        ANALYSIS:
        {self.analysis.format()}

        REFLECTION:
        {self.reflection.format()}

        {self.self_inspection.format() if self.self_inspection else ''}

        {'='*80}
        FINAL ANSWER:
        {self.answer}

        Confidence: {self.confidence_score}%
        Processing Time: {self.processing_time:.2f}s
        {'='*80}
        """
```

## Impact on Ollama Models with Built-in Thinking

### Question: What if Ollama models support thinking from the box?

If future Ollama models include native "thinking" or "reasoning" tokens (similar to OpenAI's o1), the architecture would adapt but not fundamentally change:

### With Native Thinking Support

**Advantages:**
- Reasoning phase becomes more natural and efficient
- Better quality internal reasoning
- Potentially faster processing
- More coherent thought chains

**What changes:**
```python
async def reasoning_phase(self, query: str) -> ReasoningOutput:
    """
    With native thinking: Let model think internally,
    then extract structured reasoning.
    """
    # Model does internal reasoning (hidden thinking tokens)
    response = await self.llm.generate(
        prompt=query,
        enable_thinking=True,  # New parameter
        thinking_budget=1000   # Tokens for internal reasoning
    )

    # Extract reasoning from model's internal thoughts
    reasoning = ReasoningOutput(
        user_intent=response.thinking.user_intent,
        planned_actions=response.thinking.planned_actions,
        # ... extracted from thinking process
    )

    return reasoning
```

**What stays the same:**
- Two-phase architecture remains
- Reflection and self-inspection still needed
- Explicit action logging unchanged
- Analysis phase unchanged
- Overall transparency maintained

**Key insight:** Native thinking would enhance Phase 1.1 (Reasoning) but wouldn't eliminate the need for:
- Phase 1.2 (Action) - still need explicit tool use
- Phase 1.3 (Analysis) - still need quality assessment
- Phase 2.1 (Reflection) - still need self-critique
- Phase 2.2 (Self-Inspection) - still need code analysis

The architecture is designed to be **thinking-mode agnostic** - it works with both:
1. Current approach: Explicit prompting for reasoning
2. Future approach: Native thinking tokens

## Comparison: One Agent vs. Multiple Agents

### Option A: Single Two-Phase Agent (RECOMMENDED for Capstone)

**Structure:**
```
TwoPhaseReflectiveAgent
    ├── Phase 1: Execution
    │   ├── Reasoning
    │   ├── Action
    │   └── Analysis
    └── Phase 2: Meta-Reflection
        ├── Reflection
        └── Self-Inspection
```

**Pros:**
- Simpler architecture
- Easier to debug and maintain
- All logic in one coherent system
- Clear progression through phases
- Sufficient for demonstrating all required capabilities
- Easier to explain and document

**Cons:**
- Single class can become large
- All logic in one component
- Less modular for future extension

**Best for:** Capstone project, proof of concept, educational purposes

### Option B: Three Specialized Agents

**Structure:**
```
Orchestrator
    ├── ReasoningAgent (plans and thinks)
    ├── ActionAgent (executes MCP tools)
    └── MetaReflectionAgent (critiques and inspects)
```

**Pros:**
- True multi-agent architecture
- Separation of concerns
- Each agent has focused responsibility
- More impressive architecturally
- Easier to extend individual components
- Could enable parallel processing

**Cons:**
- More complex orchestration needed
- Inter-agent communication overhead
- More potential failure points
- Harder to debug
- Overkill for current requirements

**Best for:** Production systems, research into multi-agent systems, future scalability

## Recommendation

For the capstone project: **Use Option A (Single Two-Phase Agent)**

**Rationale:**
1. Meets all stated requirements
2. Demonstrates reasoning, action, and reflection
3. Includes self-inspection capability
4. Simpler to implement and validate
5. Easier to explain in documentation
6. Sufficient complexity for academic evaluation
7. Can be extended to Option B later if needed

## Implementation Roadmap

### Minimal Changes Required

To transform current system into two-phase reflective agent:

**Step 1: Add Reasoning Phase**
- Create `reasoning_phase()` method
- Add structured prompts for explicit thinking
- Parse reasoning into actionable plan

**Step 2: Enhance Action Phase**
- Add action logging with rationale
- Track each MCP tool call explicitly
- Include quality metrics per action

**Step 3: Add Analysis Phase**
- Evaluate search result quality
- Assess coverage and relevance
- Identify gaps

**Step 4: Implement Reflection Phase**
- Self-critique prompt design
- Confidence scoring
- Alternative approach generation

**Step 5: Implement Self-Inspection Phase**
- Source code reading capability
- Code analysis prompts
- Improvement suggestion generation

**Step 6: Update Response Format**
- Include all phase outputs
- Transparent display of reasoning
- Confidence metrics

### Files to Modify

Primary changes in:
- `ragmcp/local_agent.py` - Add two-phase logic
- `ragmcp/local_agent_web.py` - Update UI to show phases
- New file: `ragmcp/reflective_agent.py` - Core implementation

Minimal changes to:
- `ragmcp/mcp_server_postgres.py` - No changes needed
- `ragmcp/llm_integration.py` - Minor prompt enhancements

## Success Criteria

The enhanced agent will be successful when it can:

1. **Demonstrate Reasoning**
   - Show explicit thinking before acting
   - Explain tool selection rationale
   - Identify potential challenges upfront

2. **Execute Tool-Based Actions**
   - Use MCP tools deliberately
   - Log each action with purpose
   - Provide quality metrics

3. **Reflect on Performance**
   - Critique its own decisions
   - Rate confidence in results
   - Suggest improvements

4. **Perform Self-Inspection**
   - Read its own source code
   - Identify potential issues
   - Recommend optimizations

5. **Maintain Transparency**
   - Show all phases in output
   - Explain decision-making process
   - Provide confidence metrics

## Conclusion

The two-phase reflective agent architecture enhances the existing RAG-MCP system with:

- **Phase 1 (Execution):** Reasoning → Action → Analysis
- **Phase 2 (Meta-Reflection):** Reflection → Self-Inspection

This design:
- Meets all capstone requirements
- Maintains simplicity and clarity
- Adds transparency and self-awareness
- Remains compatible with both current and future LLMs
- Provides a foundation for future multi-agent evolution

The architecture is **thinking-mode agnostic** and will benefit from but not require native reasoning capabilities in future Ollama models.

## Next Steps

1. Review this architecture with project stakeholders
2. Implement core `TwoPhaseReflectiveAgent` class
3. Add phase-specific methods
4. Update UI to display all phases
5. Test with sample queries
6. Document examples of each phase in action
7. Prepare for capstone submission
