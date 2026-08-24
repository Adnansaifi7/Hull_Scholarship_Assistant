with open("generate.py") as f:
    content = f.read()

old = '''        parsed = json.loads(match.group(0))
        parsed.setdefault("citations", [])
        parsed.setdefault("confidence", "low")'''

new = '''        parsed = json.loads(match.group(0))
        parsed.setdefault("citations", [])
        parsed["citations"] = [
            c.split("chunk_id:", 1)[-1].strip() if "chunk_id:" in c else c.strip()
            for c in parsed["citations"]
        ]
        parsed.setdefault("confidence", "low")'''

if old in content:
    content = content.replace(old, new)
    with open("generate.py", "w") as f:
        f.write(content)
    print("Patched successfully.")
elif 'c.split("chunk_id:"' in content:
    print("Already patched — no change needed.")
else:
    print("Pattern not found — file may differ from expected. No changes made.")
