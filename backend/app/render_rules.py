from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime

def md_escape(s: str) -> str:
    return s.replace("|", "\\|")

def render_markdown_policy(*, rules: List[Dict[str, Any]], exclusions: List[str], boilerplate_examples: List[str]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = []
    out.append(f"# DLP Keyword/Regex Policy Proposal\n")
    out.append(f"**Generated:** {ts}\n")
    out.append("This output is **human-readable** and designed to be pasted into a policy document.\n")
    out.append("---\n")

    if boilerplate_examples:
        out.append("## Detected Boilerplate (auto-removed)\n")
        out.append("Examples of repeated header/footer/template lines found across documents (used to auto-derive exclusions):\n")
        for ln in boilerplate_examples[:12]:
            out.append(f"- {md_escape(ln)}")
        out.append("\n---\n")

    if exclusions:
        out.append("## Auto Exclusions (Generic / Stop-like terms)\n")
        out.append("These are high-frequency template terms that should **not** be relied on as indicators by themselves:\n")
        out.append(", ".join(sorted(set(exclusions))[:80]))
        out.append("\n---\n")

    out.append("## Rules (Topic-based)\n")
    for i, r in enumerate(rules, 1):
        out.append(f"### Rule {i}: {r['title']}\n")
        out.append(f"**Intent:** {r['intent']}\n")

        out.append("\n**Match logic (readable):**\n")
        out.append(f"- {r['logic']}\n")

        if r.get("phrases"):
            out.append("\n**High-signal phrases (use as Keyword/Phrase list):**\n")
            for p in r["phrases"][:20]:
                out.append(f"- “{md_escape(p)}”")

        if r.get("keywords"):
            out.append("\n**Keywords (use as Keyword list):**\n")
            out.append(", ".join(r["keywords"][:30]))

        if r.get("regexes"):
            out.append("\n**Regex patterns (use as Regex rules):**\n")
            for rx in r["regexes"]:
                out.append(f"- **{md_escape(rx['name'])}:** `{rx['regex']}`")

        if r.get("avoid"):
            out.append("\n**Avoid / Exclude:**\n")
            out.append(", ".join(r["avoid"][:40]))

        if r.get("notes"):
            out.append("\n**Notes:**\n")
            for n in r["notes"]:
                out.append(f"- {md_escape(n)}")

        out.append("\n---\n")

    out.append("## Implementation Notes\n")
    out.append("- Keep each rule **tight**: phrases + IDs/number-regex beats generic words.\n")
    out.append("- Because future documents may be stylistically similar but not identical, prefer **structural identifiers** (INV/PO/MSA/WP/CC/PRJ patterns) + a small set of topic phrases.\n")
    out.append("- If false positives appear, the first fix is usually: add exclusions + reduce generic keywords.\n")
    return "\n".join(out)
