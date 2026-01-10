#!/usr/bin/env python3
"""
Two-Phase RAG Agent with Comprehensive Recommendations
Implements a three-phase approach:
  Phase 1: Reasoning & File Selection (using dual collections + similar tasks)
  Phase 2: Deep Analysis with File Content (loads actual code and diffs)
  Phase 3: Final Comprehensive Recommendation (synthesizes all information)
"""

import asyncio
import sys
import json
import re
from typing import Optional, List, Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
import config


@dataclass
class FileScore:
    """Represents a file with relevance score"""
    path: str
    similarity: float
    relevance: str


@dataclass
class Phase1Output:
    """Output from Phase 1: Reasoning & File Selection

    IMPORTANT DISTINCTION:
    - module_scores: FOLDER-LEVEL semantic recommendations (e.g., "src/main/java/auth/")
      Purpose: Provide high-level structural context about which root directories are relevant
      Use: Help validate that selected files are in the right areas of the codebase

    - file_scores: INDIVIDUAL FILE-LEVEL semantic recommendations (e.g., "LoginService.java")
      Purpose: Provide ranked list of specific files to examine
      Use: LLM selects from these for detailed analysis

    - selected_files: The actual files chosen by LLM from file_scores (NOT from modules)
    """
    task_description: str
    reasoning: str
    similar_tasks: List[Dict]
    module_scores: List[Dict]  # Folder-level context
    file_scores: List[FileScore]  # Individual files ranked by similarity
    selected_files: List[str]  # Files chosen by LLM for detailed analysis
    selection_rationale: str


@dataclass
class Phase2Output:
    """Output from Phase 2: Deep Analysis"""
    file_contents: Dict[str, str]
    file_diffs: Dict[str, str]
    analysis: str
    solution_assessment: str
    additional_files_needed: List[str]
    additional_files_rationale: str


@dataclass
class Phase3Output:
    """Output from Phase 3: Final Comprehensive Recommendation

    Synthesizes all information from Phase 1 and Phase 2 into a cohesive recommendation:
    - Similar historical tasks with their patterns and file changes
    - Module/folder structure showing code organization
    - Selected file contents and implementations
    - Deep analysis and solution assessment

    The final_recommendation provides:
    1. Executive summary of the task
    2. Key findings from similar tasks
    3. Current code structure analysis
    4. Detailed step-by-step implementation recommendation
    5. Additional context (testing, risks, edge cases)
    """
    final_recommendation: str
    confidence_score: float  # Deprecated - kept for backward compatibility
    strengths: List[str]  # Deprecated - kept for backward compatibility
    weaknesses: List[str]  # Deprecated - kept for backward compatibility
    alternative_approaches: List[str]  # Deprecated - kept for backward compatibility
    lessons_learned: List[str]  # Deprecated - kept for backward compatibility


class TwoPhaseRAGAgent:
    """Two-Phase Reflective RAG Agent with self-reflection capabilities"""

    def __init__(
        self,
        mcp_server_path: str = "mcp_server_dual.py",
        ollama_url: str = None,
        model: str = None,
        use_dual_search: bool = True,
        top_k_tasks_recent: int = 3,
        top_k_tasks_all: int = 2,
        top_k_modules_recent: int = 5,
        top_k_modules_all: int = 5,
        top_k_files_recent: int = 10,
        top_k_files_all: int = 10,
        show_task_details: bool = True,
        show_task_files: bool = True,
        show_task_diffs: bool = True,
        max_diffs_per_task: int = 3
    ):
        self.mcp_server_path = mcp_server_path
        self.ollama_url = ollama_url if ollama_url else config.OLLAMA_URL
        self.model = model if model else config.OLLAMA_MODEL
        self.use_dual_search = use_dual_search
        self.session = None
        self.execution_history = []

        # Collection result count configuration
        self.top_k_tasks_recent = top_k_tasks_recent
        self.top_k_tasks_all = top_k_tasks_all
        self.top_k_modules_recent = top_k_modules_recent
        self.top_k_modules_all = top_k_modules_all
        self.top_k_files_recent = top_k_files_recent
        self.top_k_files_all = top_k_files_all

        # Task detail visibility toggles
        self.show_task_details = show_task_details
        self.show_task_files = show_task_files
        self.show_task_diffs = show_task_diffs
        self.max_diffs_per_task = max_diffs_per_task

        # LLM input tracking
        self.llm_inputs = {
            'phase1': None,
            'phase2': None,
            'phase3': None
        }

    async def initialize(self):
        """Initialize MCP connection and LLM"""
        print("\n" + "="*80)
        print("TWO-PHASE REFLECTIVE RAG AGENT")
        print("="*80)
        print(f"[INIT] MCP Server: {self.mcp_server_path}")
        print(f"[INIT] LLM Model: {self.model}")
        print(f"[INIT] Ollama URL: {self.ollama_url}")
        print()

        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code != 200:
                print("[ERROR] Ollama is not running! Start it with: 'ollama serve'")
                return False
        except Exception as e:
            print(f"[ERROR] Cannot connect to Ollama: {e}")
            return False

        print("[OK] Ollama is running")

        # Connect to MCP server
        server_params = StdioServerParameters(
            command="python",
            args=[self.mcp_server_path],
            env=None
        )

        try:
            self._client_context = stdio_client(server_params)
            read, write = await self._client_context.__aenter__()
            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()
            await self.session.initialize()

            tools_response = await self.session.list_tools()
            print(f"[OK] MCP Server connected ({len(tools_response.tools)} tools available)")
            print()
            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize MCP server: {e}")
            return False

    async def cleanup(self):
        """Cleanup connections"""
        try:
            if hasattr(self, '_session_context'):
                await self._session_context.__aexit__(None, None, None)
        except (RuntimeError, GeneratorExit, Exception):
            pass

        try:
            if hasattr(self, '_client_context'):
                await self._client_context.__aexit__(None, None, None)
        except (RuntimeError, GeneratorExit, Exception):
            pass

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        """Call Ollama LLM"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 3000
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=600
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"[ERROR] Ollama error: {e}"

    async def phase1_reasoning_and_selection(self, user_query: str) -> Phase1Output:
        """
        Phase 1: Reasoning, Search, and File Selection
        - Understand user intent
        - Search for similar tasks, modules, and files
        - Score and rank files
        - Ask LLM to select most important files
        """
        from datetime import datetime

        phase1_start = datetime.now()
        print("\n" + "="*80)
        print("PHASE 1: REASONING & FILE SELECTION")
        print(f"Started at: {phase1_start.strftime('%H:%M:%S')}")
        print("="*80)

        # Use dual search if enabled
        if self.use_dual_search:
            # Step 1: Search for similar tasks (dual)
            step_start = datetime.now()
            print(f"\n[1.1] Searching for similar tasks (RECENT: {self.top_k_tasks_recent}, ALL: {self.top_k_tasks_all})...")
            print(f"      ⏱️  Started at: {step_start.strftime('%H:%M:%S.%f')[:-3]}")

            tasks_result = await self.session.call_tool(
                "search_tasks_dual",
                arguments={
                    "task_description": user_query,
                    "top_k_recent": self.top_k_tasks_recent,
                    "top_k_all": self.top_k_tasks_all
                }
            )
            tasks_text = tasks_result.content[0].text

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()
            print(f"      ✓ Completed in {duration:.2f}s")

            # Enhance with task details if enabled
            if self.show_task_details or self.show_task_files or self.show_task_diffs:
                enhance_start = datetime.now()
                print(f"\n[1.1+] Enhancing tasks with details (files={self.show_task_files}, diffs={self.show_task_diffs})...")
                print(f"       ⏱️  Started at: {enhance_start.strftime('%H:%M:%S.%f')[:-3]}")

                tasks_text = await self._enhance_tasks_with_details(tasks_text)

                enhance_end = datetime.now()
                enhance_duration = (enhance_end - enhance_start).total_seconds()
                print(f"       ✓ Completed in {enhance_duration:.2f}s")

            # Step 2: Search for relevant modules (dual)
            # IMPORTANT: Modules are FOLDER-LEVEL semantic embeddings
            # They represent high-level directories/packages (e.g., "src/main/java/auth/")
            # Purpose: Provide structural context to understand which areas of codebase are relevant
            # This helps validate that selected files are in the right folder structure
            step_start = datetime.now()
            print(f"\n[1.2] Searching for relevant modules (RECENT: {self.top_k_modules_recent}, ALL: {self.top_k_modules_all})...")
            print(f"      ⏱️  Started at: {step_start.strftime('%H:%M:%S.%f')[:-3]}")

            modules_result = await self.session.call_tool(
                "search_modules_dual",
                arguments={
                    "task_description": user_query,
                    "top_k_recent": self.top_k_modules_recent,
                    "top_k_all": self.top_k_modules_all
                }
            )
            modules_text = modules_result.content[0].text

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()
            print(f"      ✓ Completed in {duration:.2f}s")

            # Step 3: Search for relevant files (dual)
            # IMPORTANT: Files are INDIVIDUAL FILE-LEVEL semantic embeddings
            # They represent specific source files (e.g., "src/main/java/auth/LoginService.java")
            # Purpose: Provide ranked list of specific files that LLM can select for detailed analysis
            # The LLM will choose which of these files to actually load and examine
            step_start = datetime.now()
            print(f"\n[1.3] Searching for relevant files (RECENT: {self.top_k_files_recent}, ALL: {self.top_k_files_all})...")
            print(f"      ⏱️  Started at: {step_start.strftime('%H:%M:%S.%f')[:-3]}")

            files_result = await self.session.call_tool(
                "search_files_dual",
                arguments={
                    "task_description": user_query,
                    "top_k_recent": self.top_k_files_recent,
                    "top_k_all": self.top_k_files_all
                }
            )
            files_text = files_result.content[0].text

            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()
            print(f"      ✓ Completed in {duration:.2f}s")
        else:
            # Original single search
            # Step 1: Search for similar tasks
            print("\n[1.1] Searching for similar tasks...")
            tasks_result = await self.session.call_tool(
                "search_similar_tasks",
                arguments={"task_description": user_query, "top_k": 5}
            )
            tasks_text = tasks_result.content[0].text

            # Step 2: Search for relevant modules
            print("[1.2] Searching for relevant modules...")
            modules_result = await self.session.call_tool(
                "search_modules",
                arguments={"task_description": user_query, "top_k": 10}
            )
            modules_text = modules_result.content[0].text

            # Step 3: Search for relevant files
            print("[1.3] Searching for relevant files...")
            files_result = await self.session.call_tool(
                "search_files",
                arguments={"task_description": user_query, "top_k": 20}
            )
            files_text = files_result.content[0].text

        # Step 4: Ask LLM to reason and select files
        step_start = datetime.now()
        print(f"\n[1.4] LLM reasoning and file selection...")
        print(f"      ⏱️  Started at: {step_start.strftime('%H:%M:%S.%f')[:-3]}")

        reasoning_prompt = f"""# Task Description
{user_query}

# Similar Historical Tasks
{tasks_text}

# Relevant Modules (FOLDER-LEVEL context - these are ROOT DIRECTORIES)
# Purpose: Shows which high-level folders/packages are semantically relevant
# This helps validate if recommended files are in the right area of the codebase
{modules_text}

# Relevant Files (SPECIFIC FILES to potentially examine)
# Purpose: Individual file paths ranked by semantic similarity
# These are the actual files you can select for detailed analysis
{files_text}

Your task is to:
1. Understand what the user is asking for
2. Analyze the similar tasks to learn from past solutions
3. Review the MODULE rankings to understand which root folders/packages are relevant
   - Modules show folder structure (e.g., "src/main/java/auth/", "server/api/")
   - Use this to validate if recommended files align with relevant areas of codebase
4. Review the FILE rankings to see specific files ranked by similarity
5. Select the TOP 5-10 MOST IMPORTANT FILES to examine in detail
   - IMPORTANT: Select INDIVIDUAL FILES, not modules/folders
   - Choose files that are in the relevant modules AND most likely to contain implementation

Please provide:
1. Your reasoning about this task
2. A JSON array of INDIVIDUAL FILE PATHS you want to examine, formatted exactly as:
   {{"selected_files": ["path/to/file1.java", "path/to/file2.java"]}}
3. Rationale for why you selected these specific files

Be strategic:
- Use modules to understand the overall structure
- Select individual files (not folders) that are in relevant modules
- Prioritize files most likely to contain the actual implementation details
"""

        system_prompt = """You are an expert software engineer analyzing codebases.

IMPORTANT DISTINCTION:
- MODULES = Folder-level context showing which root directories/packages are relevant
- FILES = Individual file paths that you can select for detailed examination

Your job is to:
1. Use module recommendations to understand the overall folder structure
2. Select INDIVIDUAL FILES (not folders) for detailed examination
3. Ensure selected files align with relevant modules

Always output a valid JSON object with "selected_files" key containing an array of INDIVIDUAL FILE PATHS (not folder paths)."""

        # Store the full LLM input for visibility
        self.llm_inputs['phase1'] = {
            'system_prompt': system_prompt,
            'user_prompt': reasoning_prompt,
            'temperature': 0.3
        }

        llm_response = self.call_llm(reasoning_prompt, system_prompt, temperature=0.3)

        step_end = datetime.now()
        duration = (step_end - step_start).total_seconds()
        print(f"      ✓ Completed in {duration:.2f}s")

        # Extract reasoning and selected files
        reasoning = llm_response
        selected_files = self._extract_json_files(llm_response)

        if not selected_files:
            print("\n[WARNING] LLM didn't select files in proper format. Using top 5 from search.")
            # Fallback: extract top files from search results
            selected_files = self._extract_top_files_from_search(files_text, 5)

        print(f"\n[✓] Selected {len(selected_files)} files for detailed analysis")

        # Parse file scores from search results
        file_scores = self._parse_file_scores(files_text)

        # Phase 1 Summary
        phase1_end = datetime.now()
        phase1_duration = (phase1_end - phase1_start).total_seconds()
        print(f"\n" + "="*80)
        print(f"PHASE 1 COMPLETE - Total time: {phase1_duration:.2f}s")
        print(f"Finished at: {phase1_end.strftime('%H:%M:%S')}")
        print("="*80)

        return Phase1Output(
            task_description=user_query,
            reasoning=reasoning,
            similar_tasks=self._parse_tasks(tasks_text),
            module_scores=self._parse_modules(modules_text),
            file_scores=file_scores,
            selected_files=selected_files,
            selection_rationale=reasoning
        )

    async def phase2_deep_analysis(self, phase1: Phase1Output) -> Phase2Output:
        """
        Phase 2: Deep Analysis with File Content
        - Fetch content and diffs for selected files
        - Analyze the actual code
        - Check if solution is viable
        - Decide if more files are needed
        """
        print("\n" + "="*80)
        print("PHASE 2: DEEP ANALYSIS")
        print("="*80)

        file_contents = {}
        file_diffs = {}

        # Fetch file contents and diffs
        print(f"\n[2.1] Fetching content for {len(phase1.selected_files)} files...")
        for i, file_path in enumerate(phase1.selected_files, 1):
            print(f"  [{i}/{len(phase1.selected_files)}] {file_path}")

            # Try to get file content
            try:
                content_result = await self.session.call_tool(
                    "get_file_content",
                    arguments={"file_path": file_path}
                )
                file_contents[file_path] = content_result.content[0].text
            except Exception as e:
                print(f"      [WARNING] Could not get content: {e}")
                file_contents[file_path] = f"(Content not available: {e})"

        print("[OK] Files fetched")

        # Step 2: LLM analysis
        print("\n[2.2] LLM analyzing file contents...")

        # Build context
        files_context = ""
        for path, content in file_contents.items():
            files_context += f"\n{'='*60}\n"
            files_context += f"FILE: {path}\n"
            files_context += f"{'='*60}\n"
            files_context += content[:5000]  # Limit to avoid token overflow
            if len(content) > 5000:
                files_context += "\n... (truncated)"
            files_context += "\n"

        analysis_prompt = f"""# Original Task
{phase1.task_description}

# Your Initial Reasoning
{phase1.reasoning}

# File Contents
{files_context}

Now that you have examined the actual code, please:

1. **Analyze** the implementation in these files
2. **Assess** if you have enough information to provide a good solution/recommendation
3. **Decide** if you need to examine additional files (respond with JSON if yes)

If you need more files, provide:
{{"additional_files": ["path/to/file.java"], "rationale": "why you need them"}}

If you have enough information, provide your analysis and recommendations.
"""

        system_prompt = """You are an expert code reviewer.
Analyze the provided code carefully and determine if you have enough context.
If you need more files, output valid JSON with "additional_files" and "rationale" keys."""

        # Store the full LLM input for visibility
        self.llm_inputs['phase2'] = {
            'system_prompt': system_prompt,
            'user_prompt': analysis_prompt,
            'temperature': 0.5
        }

        llm_analysis = self.call_llm(analysis_prompt, system_prompt, temperature=0.5)

        # Check if LLM wants more files
        additional_files = self._extract_json_additional_files(llm_analysis)

        print(f"[OK] Analysis complete")
        if additional_files:
            print(f"[INFO] LLM requests {len(additional_files)} additional files")

        return Phase2Output(
            file_contents=file_contents,
            file_diffs=file_diffs,
            analysis=llm_analysis,
            solution_assessment=llm_analysis,
            additional_files_needed=additional_files if additional_files else [],
            additional_files_rationale=llm_analysis
        )

    async def phase3_reflection(
        self,
        phase1: Phase1Output,
        phase2: Phase2Output
    ) -> Phase3Output:
        """
        Phase 3: Final Comprehensive Recommendation
        - Synthesize all information from Phase 1 and Phase 2
        - Extract key insights from similar tasks, modules, files, and diffs
        - Provide actionable recommendations
        """
        print("\n" + "="*80)
        print("PHASE 3: FINAL RECOMMENDATION")
        print("="*80)

        print("\n[3.1] Generating comprehensive recommendation...")

        # Build context sections
        similar_tasks_summary = self._format_similar_tasks(phase1.similar_tasks)
        modules_summary = self._format_modules(phase1.module_scores)
        selected_files_summary = "\n".join([f"  - {f}" for f in phase1.selected_files])

        reflection_prompt = f"""# Task
{phase1.task_description}

# Context from Similar Historical Tasks
{similar_tasks_summary}

# Relevant Code Structure (Modules/Folders)
{modules_summary}

# Selected Files for Analysis
{selected_files_summary}

# File Content Analysis
{phase2.analysis}

# Solution Assessment
{phase2.solution_assessment}

---

Based on ALL the information provided above (similar tasks with their diffs, module structure, file contents, and analysis), provide a COMPREHENSIVE FINAL RECOMMENDATION that:

1. **Executive Summary** (2-3 sentences)
   - What is the task asking for?
   - What is the core challenge?

2. **Key Findings from Similar Tasks**
   - What did similar historical tasks reveal?
   - What patterns or approaches were used before?
   - What files were commonly modified?

3. **Current Code Structure Analysis**
   - Which modules/folders are most relevant?
   - How is the functionality currently organized?
   - What are the key files and their responsibilities?

4. **Detailed Implementation Recommendation**
   - Step-by-step approach to solve the task
   - Which files need to be modified?
   - What specific changes are needed?
   - What existing patterns should be followed?
   - Any important considerations or gotchas?

5. **Additional Context**
   - Related files that might need attention
   - Testing considerations
   - Potential risks or edge cases

Synthesize ALL the data provided - don't just repeat the analysis, but create a cohesive, actionable plan that draws insights from:
- Historical task patterns and diffs
- Module structure context
- Current file implementations
- Your deep analysis

Be concrete and specific. Reference actual file names, classes, and methods where relevant.
"""

        system_prompt = """You are an expert software engineer providing a final comprehensive recommendation.

Your job is to synthesize ALL available information:
- Similar historical tasks and their file changes
- Module/folder structure showing code organization
- Selected file contents and implementations
- Deep analysis of the code

Provide a clear, actionable, well-structured recommendation that a developer can immediately act on.
Be specific with file names, approaches, and implementation steps.
Draw insights from historical patterns while addressing the current task."""

        # Store the full LLM input for visibility
        self.llm_inputs['phase3'] = {
            'system_prompt': system_prompt,
            'user_prompt': reflection_prompt,
            'temperature': 0.4
        }

        final_recommendation = self.call_llm(reflection_prompt, system_prompt, temperature=0.4)

        print(f"[OK] Comprehensive recommendation generated")

        return Phase3Output(
            final_recommendation=final_recommendation,
            confidence_score=0.0,  # Not used anymore
            strengths=[],  # Not used anymore
            weaknesses=[],  # Not used anymore
            alternative_approaches=[],  # Not used anymore
            lessons_learned=[]  # Not used anymore
        )

    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Main entry point - orchestrates all three phases"""
        start_time = datetime.now()

        print(f"\n{'='*80}")
        print(f"QUERY: {user_query}")
        print(f"{'='*80}")

        # Phase 1: Reasoning & File Selection
        phase1 = await self.phase1_reasoning_and_selection(user_query)

        # Phase 2: Deep Analysis
        phase2 = await self.phase2_deep_analysis(phase1)

        # Phase 3: Final Reflection
        phase3 = await self.phase3_reflection(phase1, phase2)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Store in execution history
        execution_record = {
            "query": user_query,
            "phase1": asdict(phase1),
            "phase2": asdict(phase2),
            "phase3": asdict(phase3),
            "llm_inputs": self.llm_inputs,  # Include all LLM inputs for visibility
            "processing_time": processing_time,
            "timestamp": start_time.isoformat()
        }
        self.execution_history.append(execution_record)

        return execution_record

    def display_results(self, result: Dict[str, Any]):
        """Display results in a user-friendly format"""
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)

        print(f"\n📋 QUERY: {result['query']}")
        print(f"⏱️  PROCESSING TIME: {result['processing_time']:.2f}s")

        print(f"\n{'='*80}")
        print("PHASE 1: FILE SELECTION")
        print("="*80)
        print(f"Selected {len(result['phase1']['selected_files'])} files:")
        for i, file in enumerate(result['phase1']['selected_files'], 1):
            print(f"  {i}. {file}")

        print(f"\n{'='*80}")
        print("PHASE 2: ANALYSIS")
        print("="*80)
        print(result['phase2']['analysis'][:500] + "..." if len(result['phase2']['analysis']) > 500 else result['phase2']['analysis'])

        print(f"\n{'='*80}")
        print("PHASE 3: FINAL RECOMMENDATION")
        print("="*80)
        print(result['phase3']['final_recommendation'])

        print(f"\n⏱️  TOTAL TIME: {result['processing_time']:.2f}s")
        print("="*80)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _extract_json_files(self, text: str) -> List[str]:
        """Extract file list from LLM JSON response"""
        try:
            # Look for JSON object with selected_files
            json_match = re.search(r'\{[^{}]*"selected_files"[^{}]*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
            if json_match:
                json_obj = json.loads(json_match.group())
                return json_obj.get("selected_files", [])
        except:
            pass
        return []

    def _extract_json_additional_files(self, text: str) -> Optional[List[str]]:
        """Extract additional files request from LLM response"""
        try:
            json_match = re.search(r'\{[^{}]*"additional_files"[^{}]*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
            if json_match:
                json_obj = json.loads(json_match.group())
                return json_obj.get("additional_files", [])
        except:
            pass
        return None

    def _extract_top_files_from_search(self, search_text: str, count: int) -> List[str]:
        """Extract top N file paths from search results"""
        files = []
        for line in search_text.split('\n'):
            if line.strip().startswith('`') and line.strip().endswith('`'):
                # Extract file path between backticks
                path = line.strip().strip('`').strip()
                if '.' in path:  # Basic check for file extension
                    files.append(path)
                if len(files) >= count:
                    break
        return files

    def _parse_file_scores(self, text: str) -> List[FileScore]:
        """Parse file scores from search results"""
        scores = []
        current_file = None
        current_sim = 0.0

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('`') and line.endswith('`'):
                current_file = line.strip('`').strip()
            elif 'Similarity:' in line and current_file:
                try:
                    sim_str = line.split('Similarity:')[1].strip()
                    current_sim = float(sim_str)
                    relevance = 'High' if current_sim >= 0.7 else 'Medium' if current_sim >= 0.5 else 'Low'
                    scores.append(FileScore(current_file, current_sim, relevance))
                    current_file = None
                except:
                    pass

        return scores

    def _parse_tasks(self, text: str) -> List[Dict]:
        """Parse tasks from search results"""
        return [{"summary": text[:200]}]

    def _parse_modules(self, text: str) -> List[Dict]:
        """Parse modules from search results

        Modules represent FOLDER-LEVEL semantic embeddings.
        These are high-level directories/packages that provide structural context.

        Example module: "src/main/java/org/sonar/server/authentication/"
        NOT individual files like "LoginService.java"

        Use: Help LLM understand which root folders are relevant before selecting files.
        """
        return [{"summary": text[:200]}]

    def _format_similar_tasks(self, tasks: List[Dict]) -> str:
        """Format similar tasks for Phase 3 context"""
        if not tasks or not tasks[0].get('summary'):
            return "No similar historical tasks found."

        formatted = "Similar historical tasks provide patterns and context:\n"
        for idx, task in enumerate(tasks[:5], 1):  # Limit to top 5
            formatted += f"\n  Task {idx}: {task['summary']}\n"

        return formatted

    def _format_modules(self, modules: List[Dict]) -> str:
        """Format modules for Phase 3 context"""
        if not modules or not modules[0].get('summary'):
            return "No module structure information available."

        formatted = "Relevant folders/packages in the codebase:\n"
        formatted += f"  {modules[0]['summary']}\n"

        return formatted

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from reflection"""
        match = re.search(r'confidence[:\s]+(\d+)%?', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 50.0  # Default

    def _extract_list_section(self, text: str, section_name: str) -> List[str]:
        """Extract list items from a section"""
        items = []
        in_section = False

        for line in text.split('\n'):
            if section_name.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    items.append(line.strip().lstrip('-').lstrip('•').strip())
                elif line.strip() and not line.strip().startswith('**'):
                    # End of section
                    break

        return items if items else ["(None specified)"]

    async def _enhance_tasks_with_details(self, tasks_text: str) -> str:
        """Enhance task list with details (title, description, files, diffs)"""
        from datetime import datetime
        import re

        # Extract task names from the search results
        # Pattern matches: "1. **SONAR-12345** (similarity: 0.92)"
        task_pattern = r'\d+\.\s+\*\*([^\*]+)\*\*'
        task_names = re.findall(task_pattern, tasks_text)

        if not task_names:
            # Try alternative pattern for backtick format: "1. `path` (similarity: 0.92)"
            task_pattern_alt = r'\d+\.\s+`([^`]+)`'
            task_names = re.findall(task_pattern_alt, tasks_text)

        if not task_names:
            print(f"       ⚠ No task names found in search results")
            return tasks_text

        enhanced_text = tasks_text + "\n\n" + "="*80 + "\n"
        enhanced_text += "TASK DETAILS\n"
        enhanced_text += "="*80 + "\n\n"

        total_tasks = min(len(task_names), 10)
        print(f"       → Fetching details for {total_tasks} tasks...")

        for idx, task_name in enumerate(task_names[:10], 1):  # Limit to first 10 tasks
            task_start = datetime.now()
            print(f"       → Task {idx}/{total_tasks}: {task_name[:50]}...", end=" ")

            enhanced_text += f"\n{'─'*60}\n"
            enhanced_text += f"Task: {task_name}\n"
            enhanced_text += f"{'─'*60}\n"

            try:
                # Get task files
                if self.show_task_files:
                    files_result = await self.session.call_tool(
                        "get_task_files",
                        arguments={"task_name": task_name}
                    )
                    files_text = files_result.content[0].text
                    enhanced_text += f"\n📁 Changed Files:\n{files_text}\n"

                # Get diffs if enabled
                if self.show_task_diffs and self.show_task_files:
                    # Extract file paths from JSON in files_text
                    # files_text contains: ```json\n{"task_name": ..., "files": [...]}\n```
                    try:
                        import json as json_module
                        # Find JSON block in files_text
                        json_match = re.search(r'```json\n(.*?)\n```', files_text, re.DOTALL)
                        if json_match:
                            files_data = json_module.loads(json_match.group(1))
                            file_list = files_data.get('files', [])

                            if file_list:
                                enhanced_text += f"\n📝 File Diffs (showing top {self.max_diffs_per_task}):\n"
                                for file_info in file_list[:self.max_diffs_per_task]:
                                    file_path = file_info.get('path', '')
                                    if file_path:
                                        try:
                                            diff_result = await self.session.call_tool(
                                                "get_file_diff",
                                                arguments={
                                                    "task_name": task_name,
                                                    "file_path": file_path
                                                }
                                            )
                                            diff_text = diff_result.content[0].text
                                            # Truncate long diffs
                                            if len(diff_text) > 2000:
                                                diff_text = diff_text[:2000] + "\n... (truncated)"
                                            enhanced_text += f"\n{diff_text}\n"
                                        except Exception as e:
                                            enhanced_text += f"\nFile: {file_path} - (diff not available: {e})\n"
                    except Exception as e:
                        enhanced_text += f"\n(Could not parse file list: {e})\n"

                task_end = datetime.now()
                task_duration = (task_end - task_start).total_seconds()
                print(f"({task_duration:.2f}s)")

            except Exception as e:
                enhanced_text += f"(Details not available: {e})\n"
                print(f"FAILED")

        return enhanced_text

    async def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("\nCommands:")
        print("  help  - Show this help")
        print("  exit  - Exit the agent")
        print("\nEnter your task description to begin analysis.")
        print("="*80 + "\n")

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n[BYE] Goodbye!")
                    break

                if user_input.lower() == 'help':
                    print("\nEnter a task description to analyze.")
                    print("Example: 'Fix memory leak in buffer pool'")
                    continue

                # Process query
                result = await self.process_query(user_input)

                # Display results
                self.display_results(result)

            except KeyboardInterrupt:
                print("\n\n[BYE] Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()


async def main():
    """Main entry point"""
    agent = TwoPhaseRAGAgent()

    if not await agent.initialize():
        print("[ERROR] Failed to initialize agent")
        sys.exit(1)

    try:
        await agent.interactive_mode()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Agent stopped by user")
        sys.exit(0)
