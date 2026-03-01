import os
import json
import time
import uuid
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
load_dotenv(override=True)
import gradio as gr
from rag_pipeline import RAGConfig, build_or_load_index, retrieve_context
from hf_client import HFClient
from prompts import questions_prompt, analysis_prompt, roadmap_prompt
import re
import ast

CFG = RAGConfig()
os.makedirs("outputs", exist_ok=True)
INDEX = build_or_load_index(CFG)
HF = HFClient(model="mistralai/Mistral-7B-Instruct-v0.2")

# TODO----------------------------- Helpers ---------------------------------------------------------------------------------------------------------

def _nonce() -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _normalize_questions(qobj):
    """Accept list OR {'questions': list}."""
    if isinstance(qobj, dict) and "questions" in qobj and isinstance(qobj["questions"], list):
        return qobj["questions"]
    if isinstance(qobj, list):
        return qobj
    return []


def _ensure_mcq_schema(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Force MCQ schema only.
    Ensure:
      - type = mcq
      - options = 4 strings
      - answer_key A/B/C/D exists
    """
    fixed = []
    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue

        q["id"] = q.get("id", i)
        q["type"] = "mcq"
        q["question"] = str(q.get("question", "")).strip() or f"Question {i}"
        q["difficulty"] = q.get("difficulty", "medium")
        q["concept"] = q.get("concept", "general")

        opts = q.get("options", [])
        if not isinstance(opts, list):
            opts = []

        # Normalize options: accept strings OR dicts like {"option": "...", "correct": bool}
        cleaned_opts = []
        for x in opts:
            if isinstance(x, str):
                val = x.strip()
                if val:
                    cleaned_opts.append(val)
            elif isinstance(x, dict):
                # try common keys
                for key in ["option", "text", "label", "value"]:
                    if key in x and isinstance(x[key], str) and x[key].strip():
                        cleaned_opts.append(x[key].strip())
                        break

        # ensure exactly 4 options
        while len(cleaned_opts) < 4:
            cleaned_opts.append(f"Option {len(cleaned_opts)+1}")
        cleaned_opts = cleaned_opts[:4]
        q["options"] = cleaned_opts

        # enforce answer_key
        ak = str(q.get("answer_key", "A")).strip().upper()
        if ak not in ["A", "B", "C", "D"]:
            ak = "A"
        q["answer_key"] = ak

        q.setdefault("expected_answer", "Correct option is the answer key.")
        fixed.append(q)

    # re-id in order
    for idx, q in enumerate(fixed, start=1):
        q["id"] = idx
    return fixed


def _safe_eval_literal(text: str):
    """Safely parse python-like literals: ['a', ...] / {'k': 'v'}"""
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


# compute score + per-question feedback locally
def _compute_mcq_grading(questions: List[Dict[str, Any]], user_answers: List[str]) -> Dict[str, Any]:
    total = len(questions)
    correct = 0
    details = []

    letters = ["A", "B", "C", "D"]

    for i, q in enumerate(questions):
        ak = str(q.get("answer_key", "A")).strip().upper()
        ua = (user_answers[i] if i < len(user_answers) else "") or ""
        ua = ua.strip().upper()

        is_valid = ua in letters
        is_correct = (ua == ak) if is_valid else False
        if is_correct:
            correct += 1

        opts = q.get("options") or ["", "", "", ""]
        opt_map = dict(zip(letters, opts))

        if not ua:
            feedback = "⚠️ Not answered"
        elif is_correct:
            feedback = "✅ Correct"
        else:
            feedback = "❌ Incorrect"

        details.append(
            {
                "id": q.get("id", i + 1),
                "question": q.get("question", ""),
                "user_answer": ua,
                "user_answer_text": opt_map.get(ua, "") if ua else "",
                "correct_answer": ak,
                "correct_answer_text": opt_map.get(ak, ""),
                "explanation": q.get("expected_answer", ""),
                "feedback": feedback,
                "is_correct": is_correct,
            }
        )

    percent = int(round((correct / total) * 100)) if total else 0
    return {"correct": correct, "total": total, "percent": percent, "details": details}


def _render_questions_feedback_md(grading: Dict[str, Any]) -> str:
    details = grading.get("details", []) or []
    if not details:
        return "(No questions found.)"

    lines = []
    for d in details:
        lines.append(f"### Q{d['id']}. {d['question']}")
        lines.append(
            f"- **Your Answer** {d['user_answer'] or '—'}"
            f"{(' — ' + d['user_answer_text']) if d['user_answer_text'] else ''}"
        )
        lines.append(f"- **Correct Answer** {d['correct_answer']} — {d['correct_answer_text']}")
        if d.get("explanation"):
            lines.append(f"- **Explanation:** {d['explanation']}")
        lines.append(f"- **Feedback:** {d['feedback']}")
        lines.append("---")
    return "\n".join(lines)


# TODO----------------------------- Roadmap parsing + rendering ------------------------------------------------------------------------------------

def _parse_roadmap_text_to_steps(txt: str) -> Dict[str, Any]:
    """
    Convert roadmap text to steps dict (best effort).
    """
    if not isinstance(txt, str) or not txt.strip():
        return {"topic": "", "total_estimated_time": "", "roadmap": []}

    topic = ""
    m = re.search(r"Learning Roadmap:\s*([^\n]+)", txt, re.IGNORECASE)
    if m:
        topic = m.group(1).strip()

    txt = re.sub(r"⚠️.*$", "", txt, flags=re.MULTILINE).strip()

    parts = re.split(r"\bstep:\s*(\d+)\b", txt)
    steps = []
    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            step_num = parts[i]
            body = parts[i + 1].strip()

            def pick(pattern, default=""):
                mm = re.search(pattern, body, flags=re.IGNORECASE | re.DOTALL)
                return mm.group(1).strip() if mm else default

            title = pick(r"title:\s*(.*?)\s*why:", "")
            why = pick(r"why:\s*(.*?)\s*estimated_time:", "")
            estimated_time = pick(r"estimated_time:\s*(.*?)\s*what_to_learn:", "")

            what_raw = pick(r"what_to_learn:\s*(\[[\s\S]*?\])\s*practice:", "[]")
            practice_raw = pick(r"practice:\s*(\[[\s\S]*?\])\s*checkpoint:", "[]")

            checkpoint_raw = pick(r"checkpoint:\s*([\s\S]*?)\s*resources:", "")
            resources_raw = pick(r"resources:\s*([\s\S]*?)\s*$", "")

            what_to_learn = _safe_eval_literal(what_raw)
            if not isinstance(what_to_learn, list):
                what_to_learn = [str(what_to_learn)] if what_to_learn else []

            practice = _safe_eval_literal(practice_raw)
            if not isinstance(practice, list):
                practice = [str(practice)] if practice else []

            checkpoint_val = _safe_eval_literal(checkpoint_raw)
            if checkpoint_val is None:
                checkpoint_val = checkpoint_raw.strip()

            # try parse resources as python literal or JSON
            resources_val = _safe_eval_literal(resources_raw)
            if resources_val is None:
                try:
                    resources_val = json.loads(resources_raw)
                except Exception:
                    resources_val = resources_raw.strip()

            steps.append(
                {
                    "step": int(step_num),
                    "title": title,
                    "why": why,
                    "estimated_time": estimated_time,
                    "what_to_learn": what_to_learn,
                    "practice": practice,
                    "checkpoint": checkpoint_val,
                    "resources": resources_val,
                }
            )

    return {"topic": topic, "total_estimated_time": "", "roadmap": steps, "next_action": ""}


def _normalize_roadmap_steps(roadmap: Any) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Normalize roadmap into (topic, steps_list, total_estimated_time)
    Supports:
    - {"roadmap":[...]} / {"roadmap":{...}}
    - {"steps":[...]} / {"steps":{...}}
    - single step dict
    """
    topic = ""
    total = ""

    if isinstance(roadmap, str):
        return topic, [], total

    if not isinstance(roadmap, dict):
        return topic, [], total

    topic = (roadmap.get("topic") or "").strip()
    total = str(roadmap.get("total_estimated_time") or "").strip()

    rm = roadmap.get("roadmap", None)
    if isinstance(rm, list):
        return topic, [x for x in rm if isinstance(x, dict)], total
    if isinstance(rm, dict):
        return topic, [rm], total

    st = roadmap.get("steps", None)
    if isinstance(st, list):
        return topic, [x for x in st if isinstance(x, dict)], total
    if isinstance(st, dict):
        return topic, [st], total

    if any(k in roadmap for k in ["step", "title", "why", "estimated_time", "what_to_learn", "practice", "checkpoint", "resources"]):
        return topic, [roadmap], total

    return topic, [], total


# NEW: force at least 3 steps (if model returns 1 step only)


def _ensure_min_steps_roadmap(
    hf: HFClient,
    context: Dict[str, Any],
    analysis: Dict[str, Any],
    roadmap: Any,
    min_steps: int = 3,
    max_extend_tries: int = 1,
    max_regen_tries: int = 1,
    max_fill_tries: int = 1,
) -> Dict[str, Any]:
    """
    Guarantee roadmap has >= min_steps without using time budget.
    Strategy:
      1) Normalize current output
      2) If < min_steps: EXTEND missing steps (limited tries)
      3) If still < min_steps: REGENERATE full roadmap (limited tries)
      4) If still < min_steps: FILL missing steps (limited tries)
      5) Ensure resources exist if missing (single curator call)
    """

    nonce = _nonce()  # for variation

    # parse raw string if needed
    if isinstance(roadmap, str) and roadmap.strip():
        roadmap = _parse_roadmap_text_to_steps(roadmap)

    topic0, steps0, total0 = _normalize_roadmap_steps(roadmap)
    topic = (topic0 or context.get("topic", "") or "").strip() or "Topic"

    weak = []
    missing = []
    if isinstance(analysis, dict):
        weak = analysis.get("weak_concepts_overall", []) or []
        missing = analysis.get("missing_prerequisites", []) or []

    def _clean_and_number(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = [s for s in (steps or []) if isinstance(s, dict)]
        for idx, s in enumerate(cleaned, start=1):
            s["step"] = idx
        return cleaned

    steps = _clean_and_number(steps0)

    def _resources_missing(steps_: List[Dict[str, Any]]) -> bool:
        for s in steps_:
            r = s.get("resources")
            if not isinstance(r, dict):
                return True
            a = r.get("free_text") or []
            b = r.get("free_video") or []
            c = r.get("free_interactive") or []
            if len(a) + len(b) + len(c) == 0:
                return True
        return False

    def _curate_resources_once() -> Dict[str, Any]:
        """
        Ask model for resources once (no hardcode links).
        """
        resources_prompt = f"""
You are a resource curator.
Nonce: {nonce}

Topic: {topic}
Weak concepts: {json.dumps(weak, ensure_ascii=False)}
Missing prerequisites: {json.dumps(missing, ensure_ascii=False)}

Return STRICT JSON ONLY:
{{
  "free_text": ["https://..."],
  "free_video": ["https://..."],
  "free_interactive": ["https://..."]
}}

Rules:
- Use ONLY reputable domains.
- Do NOT invent broken URLs.
- Prefer: w3schools.com, sqlzoo.net, mode.com, freecodecamp.org, learn.microsoft.com, developer.mozilla.org, docs.oracle.com
""".strip()

        try:
            res_obj, _ = hf.generate_json(
                resources_prompt,
                max_new_tokens=350,
                temperature=0.6,
                top_p=0.9,
                retries=1,
            )
            if isinstance(res_obj, dict):
                return {
                    "free_text": res_obj.get("free_text", []) or [],
                    "free_video": res_obj.get("free_video", []) or [],
                    "free_interactive": res_obj.get("free_interactive", []) or [],
                }
        except Exception:
            pass

        # if model fails: return empty buckets (NOT hardcoded)
        return {"free_text": [], "free_video": [], "free_interactive": []}

    # ?------------------------- 0) If already enough steps: just ensure resources exist ------------------------------------------------------------------------------
    
    if len(steps) >= min_steps:
        if _resources_missing(steps):
            pack = _curate_resources_once()
            for s in steps:
                if not isinstance(s.get("resources"), dict):
                    s["resources"] = pack
                else:
                    s["resources"]["free_text"] = s["resources"].get("free_text") or pack["free_text"]
                    s["resources"]["free_video"] = s["resources"].get("free_video") or pack["free_video"]
                    s["resources"]["free_interactive"] = s["resources"].get("free_interactive") or pack["free_interactive"]

        return {
            "topic": topic,
            "roadmap": steps,
            "total_estimated_time": total0 or "N/A",
            "next_action": f"Start with Course 1: {steps[0].get('title','course 1')}",
        }

    # ?------------------------- 1) EXTEND (missing steps only) ----------------------------------------------------------------------

    for _ in range(max_extend_tries):
        if len(steps) >= min_steps:
            break

        need = min_steps - len(steps)
        existing_titles = [str(s.get("title", "") or "") for s in steps]

        extend_prompt = f"""
You are a learning roadmap generator.
Nonce: {nonce}

Topic: {topic}
Context (RAG): {json.dumps(context, ensure_ascii=False)}
Analysis (gaps): {json.dumps(analysis, ensure_ascii=False)}

Existing step titles (DO NOT repeat): {json.dumps(existing_titles, ensure_ascii=False)}

Task:
- Generate EXACTLY {need} NEW steps (only the NEW steps).
- Each step MUST include: title, why, estimated_time, what_to_learn (list), practice (list), checkpoint (string), resources (object)
- resources MUST include 3 buckets: free_text/free_video/free_interactive

Return strict JSON ONLY as an array:
[
  {{
    "step": 999,
    "title": "...",
    "why": "...",
    "estimated_time": "...",
    "what_to_learn": ["..."],
    "practice": ["..."],
    "checkpoint": "...",
    "resources": {{
      "free_text": ["https://..."],
      "free_video": ["https://..."],
      "free_interactive": ["https://..."]
    }}
  }}
]
""".strip()

        try:
            extra_obj, _ = hf.generate_json(
                extend_prompt,
                max_new_tokens=800,
                temperature=0.9,
                top_p=0.9,
                retries=1,
            )
        except Exception:
            extra_obj = []

        extra_steps = []
        if isinstance(extra_obj, list):
            extra_steps = [x for x in extra_obj if isinstance(x, dict)]
        elif isinstance(extra_obj, dict):
            rm = extra_obj.get("roadmap")
            if isinstance(rm, list):
                extra_steps = [x for x in rm if isinstance(x, dict)]

        seen = set((s.get("title", "") or "").strip().lower() for s in steps)
        for s in extra_steps:
            t = (s.get("title", "") or "").strip().lower()
            if t and t not in seen:
                steps.append(s)
                seen.add(t)

        steps = _clean_and_number(steps)

    # ?------------------------- 2) REGENERATE full roadmap ---------------------------------------------------------

    for _ in range(max_regen_tries):
        if len(steps) >= min_steps:
            break

        regen_prompt = f"""
        You are a learning roadmap generator.
        Nonce: {nonce}

        Topic: {topic}
        Context (RAG): {json.dumps(context, ensure_ascii=False)}
        Analysis (gaps): {json.dumps(analysis, ensure_ascii=False)}

        STRICT REQUIREMENTS:
        - roadmap length MUST be >= {min_steps}
        - Each step MUST include resources with 3 buckets (free_text/free_video/free_interactive)

        Return strict JSON ONLY:
        {{
        "topic": "{topic}",
        "roadmap": [{{...}}, {{...}}, {{...}}],
        "total_estimated_time": "...",
        "next_action": "1 sentence"
        }}
        """.strip()

        try:
            obj, _ = hf.generate_json(
                regen_prompt,
                max_new_tokens=1100,
                temperature=0.9,
                top_p=0.9,
                retries=1,
            )
        except Exception:
            obj = {}

        _, steps1, total1 = _normalize_roadmap_steps(obj)
        if len(steps1) >= min_steps:
            steps = _clean_and_number(steps1)
            total0 = total1 or total0

    # ?-------------------------  3) FILL missing steps (last attempt) ------------------------------------------------------------------

    for _ in range(max_fill_tries):
        if len(steps) >= min_steps:
            break

        need = min_steps - len(steps)
        existing_titles = [str(s.get("title", "") or "") for s in steps]

        fill_prompt = f"""
You are a learning roadmap generator.
Nonce: {nonce}

Topic: {topic}
Context (RAG): {json.dumps(context, ensure_ascii=False)}
Analysis (gaps): {json.dumps(analysis, ensure_ascii=False)}

Existing step titles (DO NOT repeat): {json.dumps(existing_titles, ensure_ascii=False)}

Generate EXACTLY {need} NEW steps so total becomes {min_steps}.
Return strict JSON ONLY as an array of steps with resources:
[
  {{
    "step": 999,
    "title": "...",
    "why": "...",
    "estimated_time": "...",
    "what_to_learn": ["..."],
    "practice": ["..."],
    "checkpoint": "...",
    "resources": {{
      "free_text": ["https://..."],
      "free_video": ["https://..."],
      "free_interactive": ["https://..."]
    }}
  }}
]
""".strip()

        try:
            fill_obj, _ = hf.generate_json(
                fill_prompt,
                max_new_tokens=650,
                temperature=0.95,
                top_p=0.9,
                retries=1,
            )
        except Exception:
            fill_obj = []

        fill_steps = []
        if isinstance(fill_obj, list):
            fill_steps = [x for x in fill_obj if isinstance(x, dict)]
        elif isinstance(fill_obj, dict):
            rm = fill_obj.get("roadmap")
            if isinstance(rm, list):
                fill_steps = [x for x in rm if isinstance(x, dict)]

        seen = set((s.get("title", "") or "").strip().lower() for s in steps)
        for s in fill_steps:
            t = (s.get("title", "") or "").strip().lower()
            if t and t not in seen:
                steps.append(s)
                seen.add(t)

        steps = _clean_and_number(steps)

    # ?------------------------- 4) Ensure resources exist ( if the model return without even resources) ----------------------------------------

    if steps and _resources_missing(steps):
        pack = _curate_resources_once()
        for s in steps:
            if not isinstance(s.get("resources"), dict):
                s["resources"] = pack
            else:
                s["resources"]["free_text"] = s["resources"].get("free_text") or pack["free_text"]
                s["resources"]["free_video"] = s["resources"].get("free_video") or pack["free_video"]
                s["resources"]["free_interactive"] = s["resources"].get("free_interactive") or pack["free_interactive"]

    return {
        "topic": topic,
        "roadmap": steps,
        "total_estimated_time": total0 or "N/A",
        "next_action": f"Start with Course 1: {steps[0].get('title','course 1')}" if steps else "Try Submit again for a fuller roadmap.",
    }

def _bulletify_any(x: Any, indent: int = 0) -> List[str]:
    pad = "  " * indent
    lines: List[str] = []
    if x is None:
        return [f"{pad}- (None)"]
    if isinstance(x, (str, int, float, bool)):
        return [f"{pad}- {x}"]
    if isinstance(x, list):
        for item in x:
            if isinstance(item, (dict, list)):
                lines.extend(_bulletify_any(item, indent=indent))
            else:
                lines.append(f"{pad}- {item}")
        return lines
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{k}:**")
                lines.extend(_bulletify_any(v, indent=indent + 1))
            else:
                lines.append(f"{pad}- **{k}:** {v}")
        return lines
    return [f"{pad}- {str(x)}"]



def _render_resources_md(resources) -> List[str]:
    # try parse string dict/list
    if isinstance(resources, str) and resources.strip().startswith(("{", "[")):
        parsed = _safe_eval_literal(resources)
        if parsed is None:
            try:
                parsed = json.loads(resources)
            except Exception:
                parsed = None
        if parsed is not None:
            resources = parsed

    # Always show something
    if resources is None or resources == "" or resources == {} or resources == []:
        return ["- (No resources generated)"]

    if isinstance(resources, dict):
        lines = []
        any_added = False
        for key in ["free_text", "free_video", "free_interactive", "free"]:
            vals = resources.get(key)
            if isinstance(vals, list) and vals:
                any_added = True
                title = key.replace("_", " ").title()
                lines.append(f"- **{title}:**")
                for v in vals:
                    lines.append(f"  - {v}")

        leftovers = {k: v for k, v in resources.items() if k not in ["free_text", "free_video", "free_interactive", "free"]}
        if leftovers:
            any_added = True
            lines.extend(_bulletify_any(leftovers, indent=0))

        if not any_added:
            return ["- (No resources generated)"]
        return lines

    return _bulletify_any(resources, indent=0) or ["- (No resources generated)"]


def _render_checkpoint_md(checkpoint) -> List[str]:
    # parse string dict/list if needed
    if isinstance(checkpoint, str) and checkpoint.strip().startswith(("{", "[")):
        parsed = _safe_eval_literal(checkpoint)
        if parsed is None:
            try:
                parsed = json.loads(checkpoint)
            except Exception:
                parsed = None
        if parsed is not None:
            checkpoint = parsed

    if isinstance(checkpoint, dict) and checkpoint:
        lines = ["**Checkpoint:**"]
        if checkpoint.get("question"):
            lines.append(f"- **Q:** {checkpoint.get('question')}")
        if checkpoint.get("answer_format"):
            lines.append(f"- **Answer format:** {checkpoint.get('answer_format')}")
        leftovers = {k: v for k, v in checkpoint.items() if k not in ["question", "answer_format"]}
        if leftovers:
            lines.extend(_bulletify_any(leftovers, indent=0))
        return lines

    if isinstance(checkpoint, str) and checkpoint.strip():
        return ["**Checkpoint:**", f"- {checkpoint.strip()}"]

    return ["**Checkpoint:**", "- (No checkpoint)"]


def _mermaid_label(text: Any, max_len: int = 80) -> str:
    """
    Make a safe Mermaid label.
    Mermaid is sensitive when using [Label] with special chars.
    We will use ["Label"] and escape quotes/newlines.
    """
    s = str(text or "").strip()
    s = s.replace("\\", "\\\\")      # escape backslashes
    s = s.replace('"', '\\"')        # escape double quotes
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)

    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s



def _render_roadmap_markdown(roadmap: Any, topic: str) -> str:
    # MULTI-TOPIC: ONE GRAPH
    if isinstance(roadmap, dict) and isinstance(roadmap.get("roadmaps"), list):
        rms = [rm for rm in roadmap["roadmaps"] if isinstance(rm, dict)]
        if not rms:
            return "## 🗺️ Learning Roadmap\n\n⚠️ No roadmaps found."

        # Build ONE Mermaid graph for all topics
        mer = ["```mermaid", "graph TD", 'Start(["Start"])']

        md = []
        md.append("## 🗺️ Learning Roadmap (All Topics)")
        md.append("### 🔁 Graph for all Topics")

        for ti, rm in enumerate(rms, start=1):
            tname, steps, total = _normalize_roadmap_steps(rm)
            tname = (tname or f"Topic {ti}").strip()

            topic_node = f"T{ti}"
            mer.append(f'Start --> {topic_node}["{_mermaid_label(f"Topic {ti}: {tname}")}"]')

            prev = topic_node
            for ci, s in enumerate(steps, start=1):
                title = (s.get("title") or f"Course {ci}").strip()
                course_node = f"T{ti}C{ci}"
                mer.append(f'{prev} --> {course_node}["{_mermaid_label(f"Course {ci}: {title}")}"]')
                prev = course_node

            if total:
                md.append(f"- **{tname}** — Total time: {total}")

        mer.append("```")
        md.append("\n".join(mer))
        md.append("")
        md.append("### Details (Courses order per Topic)")
        md.append("")

        # Details section
        for ti, rm in enumerate(rms, start=1):
            tname, steps, total = _normalize_roadmap_steps(rm)
            tname = (tname or f"Topic {ti}").strip()

            md.append(f"## Topic {ti}: {tname}")
            if total:
                md.append(f"**Total estimated time:** {total}")
            md.append("")

            for ci, s in enumerate(steps, start=1):
                md.append(f"### Course {ci}: {s.get('title','')}")
                if s.get("why"):
                    md.append(f"**Why:** {s.get('why')}")
                if s.get("estimated_time") != "":
                    md.append(f"**Time:** {s.get('estimated_time')}")
                md.append("")

                md.append("**What to learn:**")
                for item in (s.get("what_to_learn") or []):
                    md.append(f"- {item}")
                md.append("")

                md.append("**Practice:**")
                for item in (s.get("practice") or []):
                    md.append(f"- {item}")
                md.append("")

                md.extend(_render_checkpoint_md(s.get("checkpoint")))
                md.append("")

                md.append("**Resources:**")
                md.extend(_render_resources_md(s.get("resources")))
                md.append("---")

        return "\n".join(md)
        # raw text -> parse first
        if isinstance(roadmap, str) and roadmap.strip():
            roadmap = _parse_roadmap_text_to_steps(roadmap)

        t1, steps, total = _normalize_roadmap_steps(roadmap)
        final_topic = (t1 or (topic or "")).strip() or "Roadmap"

        if not steps:
            return f"## 🗺️ Learning Roadmap: **{final_topic}**\n\n⚠️ No courses found."

    # Mermaid: Start -> Course 1 -> Course 2 -> ...
    mer = ["```mermaid", "graph TD", 'Start(["Start"])']
    for i, s in enumerate(steps, start=1):
        title = s.get("title", f"Course {i}")
        node_label = _mermaid_label(f"Course {i}: {title}")
        if i == 1:
            mer.append(f'Start --> C{i}["{node_label}"]')
        else:
            mer.append(f'C{i-1} --> C{i}["{node_label}"]')
    mer.append("```")

    md = []
    md.append(f"## 🗺️ Learning Roadmap (Courses Order): **{final_topic}**")
    if total:
        md.append(f"**Total estimated time:** {total}")
    md.append("")
    md.append("### 🔁 Course Order (Graph)")
    md.append("\n".join(mer))
    md.append("")
    md.append("### Courses (Start → Next → Next)")
    md.append("")

    # show all courses
    for i, s in enumerate(steps, start=1):
        md.append(f"### Course {i}: {s.get('title','')}")
        if s.get("why"):
            md.append(f"**Why:** {s.get('why')}")
        if s.get("estimated_time") != "":
            md.append(f"**Time:** {s.get('estimated_time')}")
        md.append("")

        md.append("**What to learn:**")
        for item in (s.get("what_to_learn") or []):
            md.append(f"- {item}")
        md.append("")

        md.append("**Practice:**")
        for item in (s.get("practice") or []):
            md.append(f"- {item}")
        md.append("")

        md.extend(_render_checkpoint_md(s.get("checkpoint")))
        md.append("")

        md.append("**Resources:**")
        md.extend(_render_resources_md(s.get("resources")))
        md.append("---")

    # Next Action
    if isinstance(roadmap, dict) and roadmap.get("next_action"):
        md.append(f"### Next Action\n{roadmap.get('next_action')}")

    return "\n".join(md)

# TODO----------------------------- Core: Generate Questions (MCQ only) ----------------------------------------------------------------------------

def generate_questions_mcq(topic: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not topic or not topic.strip():
        raise gr.Error("Write your Topic First")

    topic = topic.strip()
    context = retrieve_context(INDEX, topic, top_k=CFG.top_k)

    hf = HF

    base = questions_prompt(context, nonce=_nonce())
    enforced = (
        base
        + "\n\n"
        + "STRICT EXTRA RULES:\n"
        + "- Generate ONLY MCQ questions.\n"
        + "- Each question MUST have exactly 4 options.\n"
        + "- Total questions MUST be between 5 and 7.\n"
        + "- Output JSON list only.\n"
    )

    questions_obj, _raw = hf.generate_json(
        enforced,
        max_new_tokens=900,
        temperature=0.9,
        top_p=0.9,
        retries=1,
    )

    questions = _ensure_mcq_schema(_normalize_questions(questions_obj))

    attempts = 0
    while len(questions) < 5 and attempts < 4:
        attempts += 1
        missing = 5 - len(questions)

        extra_prompt = f"""
Generate EXACTLY {missing} NEW MCQ questions.
Rules:
- ONLY MCQ
- Exactly 4 options each
- JSON list only
- No duplicates
Nonce: {_nonce()}
Topic: {topic}
Context: {context}
Existing questions: {[q.get("question") for q in questions]}
""".strip()

        extra_obj, _ = hf.generate_json(
            extra_prompt,
            max_new_tokens=650,
            temperature=0.95,
            top_p=0.9,
            retries=1,
        )
        extra = _ensure_mcq_schema(_normalize_questions(extra_obj))

        existing = {q["question"].strip().lower() for q in questions}
        for q in extra:
            qt = q["question"].strip().lower()
            if qt and qt not in existing:
                questions.append(q)
                existing.add(qt)

        if len(questions) > 7:
            questions = questions[:7]

    if len(questions) > 7:
        questions = questions[:7]

    if len(questions) < 5:
        raise gr.Error("model return less than 5 questions . press generate again")

    for i, q in enumerate(questions, start=1):
        q["id"] = i

    return questions, context


# TODO----------------------------- Core: Grade + Roadmap -------------------------------------------------------------------------


def _auto_topics_from_analysis(analysis: Dict[str, Any], questions: List[Dict[str, Any]], max_topics: int = 4) -> List[str]:
    topics = []
    if isinstance(analysis, dict):
        topics += (analysis.get("missing_prerequisites", []) or [])
        topics += (analysis.get("weak_concepts_overall", []) or [])

    for q in (questions or []):
        c = (q.get("concept") or "").strip()
        if c and c.lower() != "general":
            topics.append(c)

    clean, seen = [], set()
    for t in topics:
        if isinstance(t, str):
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                clean.append(t)

    return clean[:max_topics]

def analyze_and_roadmap(
    questions: List[Dict[str, Any]],
    context: Dict[str, Any],
    user_answers: List[str],
) -> Tuple[str, str, str]:
    hf = HF

    ap = analysis_prompt(context, questions, user_answers, nonce=_nonce())
    analysis, _ = hf.generate_json(ap, max_new_tokens=900, temperature=0.4, top_p=0.9, retries=1)

    # multi Topics 
    rp_topics = _auto_topics_from_analysis(
        analysis if isinstance(analysis, dict) else {},
        questions,
        max_topics=4
    )
    if not rp_topics:
        rp_topics = [context.get("topic", "Topic")]

    # Roadmap for Topic
    all_roadmaps = []
    for t in rp_topics:
        ctx_t = retrieve_context(INDEX, t, top_k=CFG.top_k)

        rp = roadmap_prompt(ctx_t, analysis, nonce=_nonce())
        rm, _ = hf.generate_json(rp, max_new_tokens=900, temperature=0.9, top_p=0.9, retries=1)

        rm = _ensure_min_steps_roadmap(
            hf=hf,
            context=ctx_t,
            analysis=(analysis if isinstance(analysis, dict) else {}),
            roadmap=rm,
            min_steps=3,
            max_extend_tries=2,
            max_regen_tries=2,
            max_fill_tries=2,
        )

        all_roadmaps.append(rm)

    # one dict roadmaps
    roadmap = {"topic": context.get("topic", ""), "roadmaps": all_roadmaps}

    session = {
        "timestamp": int(time.time()),
        "context": context,
        "questions": questions,
        "user_answers": user_answers,
        "analysis": analysis,
        "roadmap": roadmap,
    }

    out_path = f"outputs/session_{session['timestamp']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

    grading = _compute_mcq_grading(questions, user_answers)
    score_line = f"**Overall Score:** **{grading['correct']} / {grading['total']}** (**{grading['percent']}%**)"

    analysis_md = [
        "## 📊 Results",
        score_line,
        "",
        "## Questions (Correct + Your answer+ Explanation + Feedback)",
        _render_questions_feedback_md(grading),
        "",
        "**Weak Concepts (overall):**",
        "- " + "\n- ".join((analysis.get("weak_concepts_overall", []) if isinstance(analysis, dict) else []) or ["None"]),
        "",
        "**Missing Prerequisites:**",
        "- " + "\n- ".join((analysis.get("missing_prerequisites", []) if isinstance(analysis, dict) else []) or ["None"]),
        "",
        f"**Summary:** {(analysis.get('summary','') if isinstance(analysis, dict) else '')}",
        "",
        f"✅ Saved: `{out_path}`"
    ]

    roadmap_md = _render_roadmap_markdown(roadmap, topic=context.get("topic", ""))

    return "\n".join(analysis_md), roadmap_md, out_path

# TODO----------------------------- UI Styling ---------------------------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,400;0,700;1,400&display=swap');

* { box-sizing: border-box; }

:root {
  --cream:    #FAF8F5;
  --white:    #FFFFFF;
  --ink:      #1A1A2E;
  --ink2:     #3D3D5C;
  --muted:    #7A7A9A;
  --border:   #E4E1DC;
  --accent:   #4F46E5;
  --tag-bg:   #EEF2FF;
  --shadow-s: 0 1px 4px rgba(26,26,46,0.06), 0 2px 12px rgba(26,26,46,0.04);
  --radius:   14px;
}

.gradio-container {
  background: var(--cream) !important;
  font-family: 'DM Sans', sans-serif !important;
  min-height: 100vh;
}

.gradio-container > .wrap,
.gradio-container > .contain,
.gradio-container .container {
  max-width: 860px !important;
  margin: 0 auto !important;
  padding: 28px 20px 60px 20px !important;
}

.block, .gr-box, .panel, .form, .wrap {
  border-radius: var(--radius) !important;
}
.gr-box, .block, .wrap, .panel {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow-s) !important;
  padding: 14px 14px !important;
}

.gr-markdown, .prose { padding: 10px 12px !important; }
.gr-accordion .prose {
  background: var(--white) !important;
  border-radius: 12px !important;
  padding: 18px 18px !important;
  margin: 14px !important;
  overflow: visible !important;
  max-height: none !important;
}

#status_box {
  min-height: 72px !important;
  height: auto !important;
  overflow: visible !important;
  padding: 16px 20px !important;
  border-radius: var(--radius) !important;
  background: linear-gradient(135deg, #EEF2FF 0%, #F0FDF4 100%) !important;
  border: 1.5px solid #C7D2FE !important;
  box-shadow: none !important;
}

.loading_line {
  padding: 6px 10px !important;
  margin-top: 6px !important;
  color: var(--muted) !important;
  font-weight: 600 !important;
}
"""


# TODO----------------------------- Build UI -------------------------------------------------------------------------------------

with gr.Blocks(title="Learning Gap Detection") as demo:
    gr.Markdown(
        """
# Learning Gap Detection (MCQ + Roadmap)
**ChromaDB + LlamaIndex RAG + Mistral (HF API)**

- Write Your Topic → Generate MCQ (5–7)
- Choose an answer for each question.
- Submit → Analysis + Roadmap
"""
    )

    questions_state = gr.State([])
    context_state = gr.State({})
    qcount_state = gr.State(0)

    topic = gr.Textbox(
        label="Topic",
        placeholder="Example: SQL / AI / Docker ",
    )

    gen_btn = gr.Button("Generate MCQ", variant="primary")
    gen_loading = gr.Markdown("", elem_classes=["loading_line"])

    status = gr.Markdown("✅ Ready.", elem_id="status_box")

    with gr.Accordion("Questions:", open=True):
        q_md = []
        q_radio = []
        for i in range(7):
            q_md.append(gr.Markdown(value=""))  # always rendered
            q_radio.append(
                gr.Dropdown(
                    choices=["A", "B", "C", "D"],
                    label=f"Your answer for Q{i+1} (A/B/C/D)",
                    value=None,
                    interactive=False,  # disabled until questions are generated
                )
            )

    submit_btn = gr.Button("Submit", variant="primary")
    submit_loading = gr.Markdown("", elem_classes=["loading_line"])

    with gr.Row():
        with gr.Accordion("📊 Analysis", open=True):
            analysis_out = gr.Markdown()
        with gr.Accordion("🗺️ Roadmap", open=True):
            roadmap_out = gr.Markdown()

    gr.Markdown("---")
    reset_btn = gr.Button("🔄 New Topic / Reset", variant="primary")

    def _reset_all():
        q_updates = []
        r_updates = []
        for _ in range(7):
            q_updates.append(gr.update(value=""))
            r_updates.append(gr.update(value=None, interactive=False))

        return (
            "", "", "", "✅ Ready.",
            [], {}, 0,
            *q_updates, *r_updates,
            "", ""
        )



    # def _gen(topic_text: str, progress=gr.Progress(track_tqdm=True)):
    #     start = time.time()
    #     questions, ctx = generate_questions_mcq(topic_text)
    #     qcount = len(questions)

    #     q_updates = []
    #     r_updates = []

    #     letters = ["A", "B", "C", "D"]

    #     for i in range(7):
    #         if i < qcount:
    #             q = questions[i]

    #             # Question markdown
    #             # show options in markdown
    #             opts_lines = "\n".join([f"- **{L})** {opt}" for L, opt in zip(letters, q["options"])])
    #             md = f"### Q{i+1}. {q['question']}\n\n{opts_lines}"
    #             q_updates.append(gr.update(visible=True, value=md))

    #             # dropdown: choose only A/B/C/D
    #             r_updates.append(gr.update(visible=True, choices=["A","B","C","D"], value=None))
    #             # md = f"### Q{i+1}. {q['question']}"
    #             # q_updates.append(gr.update(visible=True, value=md))

    #             # # choices strings
    #             # radio_choices = [f"{L}) {opt}" for L, opt in zip(letters, q["options"])]
    #             # r_updates.append(gr.update(visible=True, choices=radio_choices, value=None))

    #         else:
    #             q_updates.append(gr.update(visible=False, value=""))
    #             r_updates.append(gr.update(visible=False, choices=[], value=None))

    #     elapsed = round(time.time() - start, 2)
    #     done_status = f"Generated {qcount} MCQ questions in **{elapsed}s**.Choose an answer for each question & Submit."

    #     return (
    #         done_status,
    #         questions,
    #         ctx,
    #         qcount,
    #         *q_updates,
    #         *r_updates,
    #         "",
    #         "",
    # )

    
    def _gen(topic_text: str, progress=gr.Progress(track_tqdm=True)):
        start = time.time()
        questions, ctx = generate_questions_mcq(topic_text)
        qcount = len(questions)

        q_updates = []
        r_updates = []

        letters = ["A", "B", "C", "D"]

        for i in range(7):
            if i < qcount:
                q = questions[i]
                opts_lines = "\n".join([f"- **{L})** {opt}" for L, opt in zip(letters, q["options"])])
                md = f"### Q{i+1}. {q['question']}\n\n{opts_lines}"

                q_updates.append(gr.update(value=md))
                r_updates.append(gr.update(value=None, interactive=True))
            else:
                q_updates.append(gr.update(value=""))
                r_updates.append(gr.update(value=None, interactive=False))

        elapsed = round(time.time() - start, 2)
        done_status = f"Generated {qcount} MCQ questions in **{elapsed}s**. Choose answers & Submit."

        return (
            done_status,
            questions,
            ctx,
            qcount,
            *q_updates,
            *r_updates,
            "",
            "",
        )


    def _submit(qs, ctx, qcount, *radios):
        if not qs:
            raise gr.Error("click Generate First")

        start = time.time()

        user_answers = []
        for i in range(qcount):
            v = (radios[i] if i < len(radios) else "") or ""
            v = v.strip()

            
            if v and v[0].upper() in ["A", "B", "C", "D"]:
                user_answers.append(v[0].upper())
            else:
                user_answers.append("")
        analysis_md, roadmap_md, _saved = analyze_and_roadmap(qs, ctx, user_answers)

        elapsed = round(time.time() - start, 2)
        status_msg = f"Done in **{elapsed}s**.The results are below."
        return status_msg, analysis_md, roadmap_md

    def _show_gen_loading():
        return "⏳ Generating questions..."

    def _hide_gen_loading():
        return ""

    def _show_submit_loading():
        return "⏳ Analyzing answers + building roadmap..."

    def _hide_submit_loading():
        return ""

    gen_btn.click(_show_gen_loading, inputs=[], outputs=[gen_loading], show_progress=False).then(
        _gen,
        inputs=[topic],
        outputs=[status, questions_state, context_state, qcount_state, *q_md, *q_radio, analysis_out, roadmap_out],
        show_progress=True,
    ).then(_hide_gen_loading, inputs=[], outputs=[gen_loading], show_progress=False)

    submit_btn.click(_show_submit_loading, inputs=[], outputs=[submit_loading], show_progress=False).then(
        _submit,
        inputs=[questions_state, context_state, qcount_state, *q_radio],
        outputs=[status, analysis_out, roadmap_out],
        show_progress=True,
    ).then(_hide_submit_loading, inputs=[], outputs=[submit_loading], show_progress=False)

    reset_btn.click(
        _reset_all,
        inputs=[],
        outputs=[
            topic, gen_loading, submit_loading, status,
            questions_state, context_state, qcount_state,
            *q_md, *q_radio,
            analysis_out, roadmap_out
        ],
        show_progress=False,
    )

demo.launch(css=CSS)
