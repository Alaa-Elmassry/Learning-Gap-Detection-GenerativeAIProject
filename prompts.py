import json
from typing import Dict, List, Any


def _json_rules() -> str:
    return (
        "IMPORTANT OUTPUT RULES:\n"
        "1) Output MUST be strict valid JSON ONLY.\n"
        "2) No extra text, no markdown, no explanations outside JSON.\n"
        "3) Use double quotes for all JSON strings.\n"
    )


def questions_prompt(context: Dict[str, Any], nonce: str) -> str:
    ctx = json.dumps(context, ensure_ascii=False)
    return f"""
You are an assessment generator for a Learning Gap Detection system.
Generate between 5 and 7 questions to assess the user's knowledge about the topic.

Topic: {context.get('topic')}
Nonce (forces variety): {nonce}

Context (retrieved via RAG, use it heavily): {ctx}

Constraints:
- Total questions: 5 to 7
- Generate ONLY MCQ questions (type = "mcq").
- Each question MUST have exactly 4 options.
- answer_key MUST be one of: "A","B","C","D".
- Keep questions clear and not too long.
- Avoid repeating the same concept.

Return JSON with this schema (JSON array ONLY):
[
  {{
    "id": 1,
    "type": "mcq",
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer_key": "A",
    "expected_answer": "...",
    "difficulty": "easy" | "medium" | "hard",
    "concept": "the key concept tested"
  }}
]

{_json_rules()}
""".strip()


def analysis_prompt(context: Dict[str, Any], questions: List[Dict[str, Any]], user_answers: List[str], nonce: str) -> str:
    ctx = json.dumps(context, ensure_ascii=False)
    qs = json.dumps(questions, ensure_ascii=False)
    ans = json.dumps(user_answers, ensure_ascii=False)
    return f"""
You are an exam grader and learning-gap analyzer.

Nonce: {nonce}

Context (RAG): {ctx}
Questions (JSON): {qs}
User answers (array aligned by index): {ans}

Task:
1) Grade each question (score 0 to 1).
2) Provide short feedback per question.
3) Detect weak concepts and missing prerequisites.
4) Output an overall score (0 to 100) and a list of gaps.

Return strict JSON:
{{
  "overall_score": 0-100,
  "per_question": [
    {{
      "id": 1,
      "score": 0-1,
      "is_correct": true/false,
      "feedback": "...",
      "weak_concepts": ["...", "..."]
    }}
  ],
  "weak_concepts_overall": ["...", "..."],
  "missing_prerequisites": ["...", "..."],
  "summary": "2-4 lines"
}}

{_json_rules()}
""".strip()


def roadmap_prompt(context: Dict[str, Any], analysis: Dict[str, Any], nonce: str) -> str:
    ctx = json.dumps(context, ensure_ascii=False)
    an = json.dumps(analysis, ensure_ascii=False)
    return f"""
You are a learning roadmap generator.

Nonce: {nonce}

Context (RAG): {ctx}
Analysis (gaps): {an}

Create a personalized learning roadmap to close the gaps.

Must:
- Be step-by-step and ordered.
- The "roadmap" array length MUST be >= 3. If fewer, regenerate internally.
- Each step MUST include: step, title, why, estimated_time, what_to_learn, practice, checkpoint, resources.
- Change titles/examples each run, do not reuse previous phrasing
- estimated_time must be a string.
- what_to_learn and practice must be arrays (lists) of strings.
- checkpoint must be a string (one clear measurable task).
- resources MUST be an object with ONLY these keys:
  - free_text: list of links or titles
  - free_video: list of links or titles
  - free_interactive: list of links or titles

Return strict JSON:
{{
  "topic": "{context.get('topic')}",
  "roadmap": [
    {{
      "step": 1,
      "title": "...",
      "why": "...",
      "estimated_time": "...",
      "what_to_learn": ["...", "..."],
      "practice": ["...", "..."],
      "checkpoint": "...",
      "resources": {{
        "free_text": ["...", "..."],
        "free_video": ["...", "..."],
        "free_interactive": ["...", "..."]
      }}
    }},
    {{
      "step": 2,
      "title": "...",
      "why": "...",
      "estimated_time": "...",
      "what_to_learn": ["...", "..."],
      "practice": ["...", "..."],
      "checkpoint": "...",
      "resources": {{
        "free_text": ["...", "..."],
        "free_video": ["...", "..."],
        "free_interactive": ["...", "..."]
      }}
    }},
    {{
      "step": 3,
      "title": "...",
      "why": "...",
      "estimated_time": "...",
      "what_to_learn": ["...", "..."],
      "practice": ["...", "..."],
      "checkpoint": "...",
      "resources": {{
        "free_text": ["...", "..."],
        "free_video": ["...", "..."],
        "free_interactive": ["...", "..."]
      }}
    }}
  ],
  "total_estimated_time": "...",
  "next_action": "1 sentence"
}}

{_json_rules()}
""".strip()