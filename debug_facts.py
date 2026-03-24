import sqlite3, os, json, requests

MEM_DB = "C:/Users/uscha/AppData/Roaming/gateclaw/memory.db"
conn = sqlite3.connect(MEM_DB)

row = conn.execute(
    "SELECT episode_id, summary_json FROM compressed_episodes ORDER BY created_at DESC LIMIT 1"
).fetchone()
if row:
    ep_id, summary_json = row
    summary = json.loads(summary_json)
    print(f"Episode: {ep_id[:30]}")
    print(f"Summary: {summary.get('summary', '')[:300]}")
    print(f"Topics: {summary.get('topics', [])}")
conn.close()


def try_parse_json(content):
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        for part in parts[1:]:
            if part.startswith("json"):
                content = part[4:]
                break
            elif part.strip():
                content = part
                break
    for start_char in ["{", "["]:
        json_start = content.find(start_char)
        if json_start < 0:
            continue
        end_char = "}" if start_char == "{" else "]"
        json_end = content.rfind(end_char)
        if json_end <= json_start:
            continue
        for end_pos in range(json_end, json_start - 1, -1):
            candidate = content[json_start : end_pos + 1]
            try:
                return json.loads(candidate), None
            except:
                continue
    return None, "No valid JSON"


prompt = f"""Extract persistent facts about the user from this conversation summary.
Return a JSON array only. No explanation. No markdown. No code blocks. No intro text.
If no clear facts, return empty array: []

Facts should be specific and useful for future context:
- Technical preferences (languages, tools, frameworks)
- Project names and what they do
- Problems encountered and whether resolved
- Decisions made
- Personal context (location, hardware, workflow)

Output format: [{{"key": "fact_name", "value": "fact_value", "topic": "topic"}}]

Summary: {summary.get("summary", "")}
Topics: {", ".join(summary.get("topics", []))}
Outcome: {summary.get("outcome", "unknown")}"""

resp = requests.post(
    "http://localhost:8888/v1/chat/completions",
    json={
        "model": "Claude-4.6-Opus-4B",
        "messages": [
            {
                "role": "system",
                "content": "You extract facts from conversations. Return ONLY a JSON array. No explanation. No markdown. No text outside the array.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 600,
    },
    timeout=60,
)
content = resp.json()["choices"][0]["message"]["content"]
print(f"Raw content:\n{repr(content[:500])}")
parsed, err = try_parse_json(content)
print(f"\nParsed: {parsed}\nError: {err}")
