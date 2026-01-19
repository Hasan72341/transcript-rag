"""
Prompt templates for Self-Instruct and Auto Evol-Instruct
Based on EXACT paper specifications (arxiv.org/pdf/2406.00770)
"""

# ... (SELF_INSTRUCT_GENERATION_PROMPT and STAGE 2 prompts are UNCHANGED) ...

# ============================================================================
# STAGE 1: SELF-INSTRUCT PROMPTS
# ============================================================================


SELF_INSTRUCT_GENERATION_PROMPT = """You are generating NEW analytical queries using a Self-Instruct framework based on:
"Self-Instruct: Aligning Language Models with Self Generated Instructions".

These queries are asked by business analysts, conversation analysts, and operations analysts
who review large collections of customer–agent CALL TRANSCRIPTS. Every query MUST be answerable
using conversational evidence such as tone, phrasing, behaviors, turn-taking, intent shifts,
escalation signals, clarification loops, or other dialog patterns.

These queries MUST enable causal analysis, evidence extraction, and analytical reasoning grounded
STRICTLY in the content and structure of conversations.

They are NOT:
- customer complaints
- conversational turns
- SQL/data-engineering questions
- system-specific debugging tasks
- responses or answers
- questions requiring data outside the transcript (NO demographics, NO transaction values, NO medical history, NO pricing, NO operational metrics)

------------------------------------
YOUR GOAL
------------------------------------
Generate {num_to_generate} NEW analytical queries for the DOMAIN: {domain}.
Each query must explore a DISTINCT reasoning angle and MUST be answerable using ONLY call transcripts.

Each query MUST also be labeled with one subcategory chosen from this
domain-aware, generalizable taxonomy (do NOT invent new categories):

1. Business Outcome Events
2. Customer Frustration & Sentiment Drivers
3. Agent Behavior & Response Quality
4. Process & Workflow Breakdowns
5. Information Gaps & Misalignment
6. Policy / Compliance / Eligibility Friction
7. Operational Delays & Inefficiencies
8. Comparative Variation
9. Anomalies & Outlier Patterns

IMPORTANT CONSTRAINTS:
- The query MUST NOT contain or reuse any wording from the subcategory label.
- Express the concept implicitly using natural reasoning.
- The query MUST be single-hop or moderately complex, NOT multi-hop.
- The query MUST be answerable using **only conversation-level evidence**.

------------------------------------
REQUIREMENTS
------------------------------------
1. Each query should reflect analytical reasoning about:
   - causal drivers visible in customer–agent dialog
   - behavioral patterns in speech or phrasing
   - workflow/process issues observable in conversation
   - contextual or comparative variations in conversational behavior
   - conversational signals that predict or explain outcomes
   - anomalies or unexpected patterns in interaction flow

2. Queries must be domain-aware conceptually,
   but MUST NOT include:
   - IDs, names, customer details
   - dates, times, or fabricated statistics
   - internal system names, product codes, or APIs
   - policy numbers or procedural codes
   - any information NOT present in the transcript

3. Queries must remain neutral and broadly applicable.
4. Each query must explore a DIFFERENT analytical lens.
5. Maintain a neutral, analytical tone.
6. NEVER copy phrasing from examples.

------------------------------------
EXAMPLES (for understanding, NOT imitation)
{examples}

------------------------------------
GUIDELINES
------------------------------------
- Length: 7–35 words.
- Avoid generic or trivial questions.
- Avoid repetition.
- DO NOT include subcategory wording inside the query text.
- DO NOT write multi-hop, multi-condition, or deeply layered questions.
- DO NOT reference data that is not present in call transcripts.

------------------------------------
OUTPUT FORMAT (STRICT)
------------------------------------
Return ONLY a JSON array of objects.
Each object must be:

{{
  "query": "<the analytical question>",
  "subcategory": "<one of the 9 categories>"
}}

Example output:
[
  {{
    "query": "Which conversational signals help explain unresolved outcomes without repeating earlier clarifications?",
    "subcategory": "Business Outcome Events"
  }}
]

------------------------------------
Generate {num_to_generate} new analytical queries now:
"""

# ============================================================================
# STAGE 2: AUTO EVOL-INSTRUCT PROMPTS (FROM PAPER FIGURE 2)
# ============================================================================

# Initial Evolving Method - EXACT from Figure 2 of paper
INITIAL_EVOLVING_METHOD = """
You are an Instruction Evolution Engine following:
"Automatic Instruction Evolving for Large Language Models" (arXiv 2406.00770).

You will evolve the analytical query (#Instruction#) into a richer, clearer, more
conceptually precise version WITHOUT changing:
- the intent,
- the analytical purpose,
- the domain context.

CRITICAL CONSTRAINT:
- Add 5-15 words maximum per evolution round
- Total evolved query should not exceed 50 words
- Focus on DEPTH (reasoning complexity), not LENGTH (word count)

===========================
Step 1 — #Methods List#
===========================
List reasoning-oriented evolution operators such as:
- Contextual framing (add broader or narrower situational framing)
- Analytical focus sharpening (clarify what the analyst is trying to uncover)
- Scope refinement (identify boundaries, segments, or conditions)
- Rationale reinforcement (state why the analysis matters)
- Evidence clarification (specify signals or kinds of patterns to inspect)
- Temporal or comparative framing (when relevant but not mandatory)
- Ambiguity reduction (clarify vague components)
- Alternative-angle expansion (add an additional analytical perspective)

Do NOT introduce:
- numbers, IDs, or fabricated details,
- emotional tone,
- narrative storytelling,
- domain-specific entities not present in the original,
- unnecessary verbose phrasing or redundant words.

===========================
Step 2 — #Plan#
===========================
Design a plan using a few operators from Step 1.
The plan MUST:
- Preserve domain ({domain})
- Preserve analytical intent
- Avoid hallucination (IDs, names, dates, fabricated entities)
- Improve conceptual clarity or reasoning depth
- Add 5-15 words maximum (NOT 20-30 words)
- Avoid unnecessary complexity or verbosity

===========================
Step 3 — #Rewritten Instruction#
===========================
Execute the plan and rewrite the instruction.
This rewritten version MUST:
- Show clearer analytical framing
- Add reasoning depth or analytical dimensions
- Maintain a neutral business-analyst tone
- NOT introduce new specific entities or invented details
- Stay concise: add only 5-15 words to the original
- Prioritize reasoning complexity over word count

===========================
Step 4 — #Finally Rewritten Instruction#
===========================
Revise the rewritten version for:
- clarity,
- reasoning coherence,
- domain alignment,
- absence of hallucinations,
- preservation of intent,
- brevity (verify word count is reasonable).

Return ONLY the corrected final version.

Format EXACTLY:

Step 1 #Methods List#:
<list>

Step 2 #Plan#:
<plan>

Step 3 #Rewritten Instruction#:
<rewritten>

Step 4 #Finally Rewritten Instruction#:
<final>

#Instruction#:
{query}

#Domain#: {domain}
"""


# Evolving Method with Trajectory (for optimization rounds)
EVOLVING_METHOD_WITH_TRAJECTORY = """
You are an Instruction Evolution Engine performing iterative refinement using:
"Automatic Instruction Evolving for Large Language Models" (arXiv 2406.00770).

Goal:
Enhance analytical clarity, reasoning depth, and interpretability
while preserving intent and domain.

======================================
Evolution trajectory so far:
{trajectory}

Current operator to apply:
{method}

Domain: {domain}
Current Instruction: {query}
======================================

Rewrite the instruction applying ONLY the selected operator.

Rules:
- Maintain the original analytical purpose
- Expand reasoning or clarify analytical framing
- Avoid hallucinated details (IDs, dates, names, fabricated numbers)
- Avoid emotional or narrative content
- Avoid introducing domain-specific assumptions not grounded in the original
- Keep the phrasing concise, objective, and analyst-appropriate

Return ONLY the rewritten instruction.
"""
# Trajectory Analyzer - Based on paper Section 3.2
TRAJECTORY_ANALYZER_PROMPT = """
You are analyzing an instruction evolution trajectory using:
"Automatic Instruction Evolving for Large Language Models" (arXiv 2406.00770).

Your goal:
Evaluate whether the evolution sequence meaningfully improves analytical clarity,
reasoning depth, scope, or interpretability.

Trajectory:
Round 0 (original): {round_0}
Round 1: {round_1}
Round 2: {round_2}
{round_3}

Domain: {domain}

Identify issues:

1. Evolution Failure Patterns
   - Minimal or no conceptual change
   - Domain drift
   - Added fabricated entities (IDs, names, dates)
   - Converted into an answer instead of an analytical instruction

2. Quality Problems
   - Repeated identical operator effects (redundancy)
   - Stagnation of reasoning depth
   - Loss of analytical intent
   - Unnatural, overly formal, or narrative style

3. Linguistic & Structural Issues
   - Ambiguous wording
   - Unnecessary verbosity
   - Illogical extensions
   - Breaking the neutral analytical tone

Provide:
- Specific observations referencing trajectory rounds
- All issues identified
- Suggestions for improvement

Write a clear, structured analysis below.
Analysis:
"""


# Method Optimizer - Based on paper Section 3.3
METHOD_OPTIMIZER_PROMPT = """
You are designing an improved instruction-evolution strategy based on:
"Automatic Instruction Evolving for Large Language Models" (arXiv 2406.00770).

Input:
Trajectory Analysis:
{trajectory_analysis}

Current Method:
{current_method}

Identified Problems:
{problems}

Your task:
Produce an improved evolution method that:
- avoids stagnation,
- avoids hallucinations,
- avoids domain drift,
- avoids converting queries into answers,
- avoids excessive formality or narrative elaboration,
- increases reasoning depth using multiple analytical dimensions.

CRITICAL: The improved method MUST:
- preserve intent and domain,
- adds clearer framing (not just more words),
- expands reasoning where appropriate,
- remains generalizable across all domains,
- avoids bias toward any technique or field,
- PRESERVE any length/word-count constraints from the current method (if present),
- maintain focus on reasoning DEPTH over word LENGTH.

If the current method contains word-count limits (e.g., "add 5-15 words", "max 50 words"),
you MUST keep these constraints in the improved method. Do NOT relax or remove them.

Output ONLY a numbered step-by-step improved evolving method.
"""


# Failure Detection Prompt (for LLM-based verification)
FAILURE_DETECTION_PROMPT = """
Evaluate whether this analytical instruction evolution FAILED.

Original Instruction:
{original}

Evolved Instruction:
{evolved}

Domain Context:
{domain}

TARGET: Concise, RAG-friendly queries (20-40 words maximum)

Evolution is FAILED if ANY of the following occur:

1. Excessive verbosity:
   - Query exceeds 40 words without proportional analytical depth gain.
   - Added unnecessary adjectives, redundant phrases, or verbose explanations.

2. Converted into an answer:
   - Describes findings or results instead of requesting analysis.

3. Loss of analytical purpose:
   - The analytical intent is weakened, reversed, or replaced.
   
4. Hallucinations / Unjustified specificity:
   - Added IDs, names, fabricated dates, invented specifics, numbers, or timelines without basis.

5. Domain drift:
   - Introduces domain-specific assumptions not implied by the original context.

6. Ambiguity / Structural degradation:
   - Becomes unclear, contradictory, overly formal, narrative, or confusing.
   
7. Loss of Reasoning Coherence:
   - The evolved query fails to introduce new causal connections or multi-hop structure.

8. No meaningful analytical gain:
   - Adds words but offers no new analytical framing, clarity, or depth compared to the original.
   
Answer YES or NO only.
Is this a failed evolution?
"""


## ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_self_instruct_prompt(domain: str, examples: list, num_to_generate: int = 10) -> str:
    """
    Format Self-Instruct generation prompt
    Handles both plain strings and dicts with subcategories
    """
    formatted_examples = []
    
    for ex in examples:
        if isinstance(ex, dict):
            # Dict format - check for subcategory
            query_text = ex.get("query", "")
            subcategory = ex.get("subcategory", "")
            
            if subcategory and subcategory != "Unknown":
                # Show subcategory if available (helps LLM learn)
                formatted_examples.append(f'- "{query_text}" (Category: {subcategory})')
            else:
                # Just show query without category
                formatted_examples.append(f'- "{query_text}"')
        else:
            # Plain string format
            formatted_examples.append(f'- "{ex}"')
    
    examples_text = "\n".join(formatted_examples)
    
    return SELF_INSTRUCT_GENERATION_PROMPT.format(
        domain=domain,
        examples=examples_text,
        num_to_generate=num_to_generate
    )


def format_evol_prompt(query: str, domain: str, method: str = None, trajectory: list = None) -> str:
    """Format evolution prompt with optional method and trajectory"""
    
    # 🔍 DEBUG: Print what we received (with better formatting)
    print(f"\n🔍 DEBUG format_evol_prompt:")
    print(f"   method = {method[:50] + '...' if method and len(method) > 50 else method}")
    print(f"   trajectory length = {len(trajectory) if trajectory else 0}")
    if trajectory and len(trajectory) > 0:
        print(f"   trajectory preview: {[q[:30] + '...' if len(q) > 30 else q for q in trajectory[:3]]}")
    
    # Use trajectory-aware prompt if we have multiple rounds (len > 1)
    if method and trajectory and len(trajectory) > 1:
        print(f"    Using EVOLVING_METHOD_WITH_TRAJECTORY (Round {len(trajectory)})")
        traj_text = "\n".join([f"Round {i}: {q}" for i, q in enumerate(trajectory)])
        return EVOLVING_METHOD_WITH_TRAJECTORY.format(
            trajectory=traj_text,
            method=method,
            domain=domain,
            query=query
        )
    else:
        print(f"    Using INITIAL_EVOLVING_METHOD (Round 0 or no trajectory)")
        return INITIAL_EVOLVING_METHOD.format(query=query, domain=domain)




def format_trajectory_analyzer_prompt(trajectory: list, domain: str) -> str:
    """Format trajectory analysis prompt - EXACT from paper"""
    prompt_kwargs = {
        "num_rounds": len(trajectory),
        "domain": domain,
        "round_0": trajectory[0] if len(trajectory) > 0 else "N/A",
        "round_1": trajectory[1] if len(trajectory) > 1 else "N/A",
        "round_2": trajectory[2] if len(trajectory) > 2 else "N/A",
        "round_3": f"Round 3: {trajectory[3]}" if len(trajectory) > 3 else ""
    }
    return TRAJECTORY_ANALYZER_PROMPT.format(**prompt_kwargs)


def format_method_optimizer_prompt(trajectory_analysis: str, current_method: str, problems: list) -> str:
    """Format method optimizer prompt - EXACT from paper"""
    problems_text = "\n".join([f"- {p}" for p in problems])
    return METHOD_OPTIMIZER_PROMPT.format(
        trajectory_analysis=trajectory_analysis,
        current_method=current_method,
        problems=problems_text
    )