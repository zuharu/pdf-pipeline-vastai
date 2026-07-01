#!/usr/bin/env python3
"""
vlm_prompt_gen.py — Phase B: Generate VLM prompts + classify figures via DeepSeek.
"""
import os, re, sys, json, argparse
from pathlib import Path
DEEPSEEK_MODEL = "deepseek-v4-flash"; MAX_CHARS = 50000
def load_api_key(env_var="DEEPSEEK_API_KEY"):
    k = os.environ.get(env_var, "")
    if k: return k
    for env_path in [Path("/workspace/.env"), Path(".env")]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith(f"{env_var}="):
                        return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return ""
def sample_markdown(md_path, max_chars=MAX_CHARS):
    with open(md_path, encoding="utf-8") as f: lines = f.readlines()
    total = len(lines); parts = []
    early = lines[:min(800, total)]; parts.append("".join(early))
    mid = [];
    for line in lines[800:]:
        s = line.lstrip();
        if s.startswith("#") and not s.startswith("####"): mid.append(s.rstrip())
    if mid: parts.append("\n## Section Index (remainder):\n" + "\n".join(mid[:200]))
    caps = []
    for i, line in enumerate(lines):
        if line.startswith("![") and "(_page_" in line:
            for j in range(i+1, min(i+4, len(lines))):
                ll = lines[j].strip();
                if ll and not ll.startswith("![]") and not ll.startswith("|"): caps.append(ll); break
    if len(caps) > 100:
        step = max(1, len(caps)//50); caps = caps[::step]
    if caps: parts.append(f"\n## Figure Captions (sample of {len(caps)}):\n" + "\n".join(caps[:100]))
    return "\n".join(parts)[:max_chars]
def extract_figures_with_context(md_path, lines_above=5, lines_below=10, max_context_chars=300):
    with open(md_path, encoding="utf-8") as f: lines = f.readlines()
    figures = []; total = len(lines)
    for i, line in enumerate(lines):
        m = re.match(r'!\[(.*?)\]\((_page_\d+_[^.]+\.[jJ][pP][eE]?[gG])\)', line.strip())
        if not m: continue
        alt = m.group(1).strip(); fn = m.group(2); name = re.sub(r'\.[jJ][pP][eE]?[gG]$', '', fn)
        cap = ""
        for j in range(i+1, min(i+4, total)):
            ll = lines[j].strip();
            if ll and not ll.startswith("![]") and not ll.startswith("|"): cap = ll; break
        ax = max(0, i-lines_above); ca = "".join(lines[ax:i]).strip()
        bx = min(total, i+1+lines_below); cb = "".join(lines[i+1:bx]).strip()
        ctx = (ca + "\n" + cb).strip()
        if len(ctx) > max_context_chars: ctx = ctx[:150] + "\n...\n" + ctx[-150:]
        figures.append({"filename":fn, "name":name, "alt_text":alt, "caption":cap, "context":ctx})
    return figures
SYSTEM_PROMPT = """You are an expert in technical document analysis. Output ONLY valid JSON with this exact structure: {"circuit":{"description":"...","prompt":"..."},"graph":{...},"block_diagram":{...},"photo":{...}}"""
CLASSIFY_FIGURES_SYSTEM_PROMPT = """You are an expert in classifying figures. Categories: circuit, graph, block_diagram, photo, other_subtype. Output ONLY valid JSON: {"figures":[{"filename":"string","category":"string"}]}"""
def generate_prompts(markdown_sample, model, api_key):
    from openai import OpenAI
    c = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    r = c.chat.completions.create(model=model, messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":f"Analyze this book content and generate figure description prompts:\n\n{markdown_sample}"}], reasoning_effort="high", extra_body={"thinking":{"type":"enabled"}}, temperature=0.3, max_tokens=4096, timeout=120)
    content = r.choices[0].message.content.strip()
    if "```" in content: content = re.sub(r"```json\s*", "", content); content = re.sub(r"```\s*", "", content)
    return json.loads(content)
def classify_figures_batch(figures, book_context, model, api_key, max_figures_per_call=500):
    from openai import OpenAI
    c = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    all_cls = []
    for bs in range(0, len(figures), max_figures_per_call):
        batch = figures[bs:bs+max_figures_per_call]
        flt = [];
        for fig in batch:
            e = f"File: {fig['filename']}\n"
            if fig['caption']: e += f"Caption: {fig['caption']}\n"
            if fig['context']: e += f"Context: {fig['context']}\n"
            flt.append(e)
        up = f"Book: {book_context}\n\nClassify each of the following {len(batch)} figures:\n\n" + "\n---\n".join(flt)
        try:
            r = c.chat.completions.create(model=model, messages=[{"role":"system","content":CLASSIFY_FIGURES_SYSTEM_PROMPT},{"role":"user","content":up}], temperature=0.1, max_tokens=8192, timeout=120)
        except Exception as e: raise RuntimeError(f"DeepSeek classification API call failed: {e}") from e
        content = r.choices[0].message.content.strip()
        if "```" in content: content = re.sub(r"```json\s*", "", content); content = re.sub(r"```\s*", "", content)
        try: result = json.loads(content)
        except json.JSONDecodeError as e: raise RuntimeError(f"DeepSeek returned invalid JSON: {e}\nFirst 500 chars: {content[:500]}") from e
        if "figures" not in result: raise RuntimeError(f"Response missing 'figures' key. Keys: {list(result.keys())}")
        cm = {item["filename"]: item["category"] for item in result["figures"]}
        for fig in batch: fig["category"] = cm.get(fig["filename"], "other_unknown"); all_cls.append(fig)
    return all_cls
OTHER_PROMPTS_SYSTEM_PROMPT = """You are an expert at generating VLM prompts for scientific figures. Output ONLY valid JSON: {"prompts":{"other_subtype":{"description":"...","prompt":"..."}}}"""
def generate_other_prompts(figures, book_context, model, api_key):
    subtypes = set();
    for fig in figures:
        cat = fig.get("category", "");
        if cat.startswith("other_"): subtypes.add(cat)
    subtypes.add("other_unknown")
    sex = {}
    for st in subtypes:
        ex = [];
        for fig in figures:
            if fig.get("category") == st:
                e = f"File: {fig['filename']}\n"
                if fig.get("caption"): e += f"Caption: {fig['caption']}\n"
                if fig.get("context"): e += f"Context: {fig['context'][:200]}\n"
                ex.append(e)
                if len(ex) >= 5: break
        sex[st] = ex
    sl = "\n".join(f"- {s}: {len(sex.get(s,[]))} examples" for s in sorted(subtypes))
    et = ""
    for st, exl in sorted(sex.items()):
        if exl: et += f"\n### {st} examples:\n" + "\n---\n".join(exl[:3])
    up = f"Book: {book_context}\n\nGenerate VLM description prompts for:\n{sl}\n\nExamples:{et if et else '(none)'}\n\nGenerate a detailed VLM prompt for EACH subtype."
    from openai import OpenAI
    c = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        r = c.chat.completions.create(model=model, messages=[{"role":"system","content":OTHER_PROMPTS_SYSTEM_PROMPT},{"role":"user","content":up}], reasoning_effort="medium", extra_body={"thinking":{"type":"enabled"}}, temperature=0.3, max_tokens=4096)
    except Exception as e: print(f"WARN: Other prompts failed: {e}"); return _hardcoded_other_prompts(subtypes)
    content = r.choices[0].message.content.strip()
    if "```" in content: content = re.sub(r"```json\s*", "", content); content = re.sub(r"```\s*", "", content)
    try: return json.loads(content).get("prompts", _hardcoded_other_prompts(subtypes))
    except json.JSONDecodeError: return _hardcoded_other_prompts(subtypes)
def _hardcoded_other_prompts(subtypes):
    fb = {"other_unknown":{"description":"Generic fallback","prompt":"Describe this figure in detail."},"other_micrograph":{"description":"Microscope images","prompt":"Describe cellular structures visible."},"other_table":{"description":"Data tables","prompt":"Extract and describe data in this table."},"other_screenshot":{"description":"Software screenshots","prompt":"Describe the software interface shown."}}
    result = {};
    for s in subtypes: result[s] = fb.get(s, fb["other_unknown"])
    return result
def save_figure_metadata(figures, category_prompts, book_context, output_path):
    entries = []
    for fig in figures:
        cat = fig.get("category", "other_unknown"); ci = category_prompts.get(cat, category_prompts.get("other_unknown", {}))
        entries.append({"filename":fig["filename"],"name":fig["name"],"caption":fig.get("caption",""),"category":cat,"prompt":ci.get("prompt","Describe this figure.")})
    meta = {"book_context":book_context,"category_prompts":category_prompts,"figures":entries}
    with open(output_path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2, ensure_ascii=False)
    cc = {};
    for fig in entries: cat = fig["category"]; cc[cat] = cc.get(cat, 0) + 1
    print(f"[METADATA] Saved {output_path}")
    print(f"[METADATA] {len(entries)} figures:")
    for cat in sorted(cc.keys()): print(f"  {cat}: {cc[cat]}")
def validate_config(config):
    warnings = []
    if not isinstance(config, dict): raise ValueError(f"Config is not a dict: {type(config)}")
    for cat, cd in config.items():
        if not isinstance(cd, dict): warnings.append(f"{cat}: not a dict"); continue
        if "prompt" not in cd: warnings.append(f"{cat}: missing prompt field")
    return warnings
def main():
    p = argparse.ArgumentParser(description="Generate VLM prompts + classify figures via DeepSeek API")
    p.add_argument("book_dir", help="Path to book staging directory")
    p.add_argument("--model", default=DEEPSEEK_MODEL, help=f"DeepSeek model (default: {DEEPSEEK_MODEL})")
    p.add_argument("--dry-run", action="store_true", help="Validate without API call")
    p.add_argument("--skip-classify", action="store_true", help="Skip figure classification")
    args = p.parse_args()
    bd = Path(args.book_dir)
    if not bd.is_dir(): print(f"ERROR: Directory not found: {bd}"); sys.exit(1)
    mf = list(bd.glob("*.md"));
    if not mf: print(f"ERROR: No .md file in {bd}"); sys.exit(1)
    mp = str(mf[0]); bn = Path(mp).stem
    print(f"[PROMPT GEN] Book: {bn} ({mp})")
    sample = sample_markdown(mp)
    print(f"[PROMPT GEN] Sampled {len(sample):,} chars from markdown")
    if args.dry_run: print("[PROMPT GEN] Dry run — skipping"); sys.exit(0)
    ak = load_api_key()
    if not ak: print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)
    print(f"[PROMPT GEN] Calling DeepSeek API ({args.model})...")
    try: config = generate_prompts(sample, args.model, ak)
    except json.JSONDecodeError as e: print(f"ERROR: DeepSeek returned invalid JSON for prompts: {e}"); sys.exit(1)
    except Exception as e: print(f"ERROR: DeepSeek API call failed: {e}"); sys.exit(1)
    warnings = validate_config(config)
    for w in warnings: print(f"[PROMPT GEN] ⚠️  {w}")
    print("[PROMPT GEN] Extracting figures from markdown...")
    try: figures = extract_figures_with_context(mp)
    except Exception as e: print(f"ERROR: Figure extraction failed: {e}"); sys.exit(1)
    print(f"[PROMPT GEN] Found {len(figures)} figure references")
    if len(figures) == 0: print("[PROMPT GEN] ⚠️  No figures — skipping classification"); args.skip_classify = True
    if not args.skip_classify and len(figures) > 0:
        bctx = bn.replace("_", " ")
        print(f"[PROMPT GEN] Classifying {len(figures)} figures via DeepSeek {args.model}...")
        try: figures = classify_figures_batch(figures, bctx, args.model, ak)
        except Exception as e: print(f"ERROR: Figure classification failed: {e}"); sys.exit(1)
        cc = {};
        for fig in figures: cat = fig.get("category", "other_unknown"); cc[cat] = cc.get(cat, 0) + 1
        print("[PROMPT GEN] Classification results:")
        for cat in sorted(cc.keys()): print(f"  {cat}: {cc[cat]}")
        ots = {f.get("category") for f in figures if f.get("category","").startswith("other_")}
        co = ots - {"other_unknown"}
        if co:
            print(f"[PROMPT GEN] Generating prompts for {len(co)} 'other' subtypes...")
            try:
                op = generate_other_prompts(figures, bctx, args.model, ak)
                for st, pd in op.items():
                    if st not in config: config[st] = pd
            except Exception as e: print(f"WARN: Other prompt generation failed: {e}")
    if "other_unknown" not in config:
        config["other_unknown"] = {"description":"Generic fallback","prompt":"Describe this figure in detail."}
    pcp = bd / "prompt_config.json"
    with open(pcp, "w", encoding="utf-8") as f: json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[PROMPT GEN] ✅ Saved {pcp} ({len(json.dumps(config))} bytes)")
    for cat in sorted(config.keys()):
        desc = config[cat].get("description", "no description")
        plen = len(config[cat].get("prompt", ""))
        print(f"  {cat}: {desc} (prompt: {plen} chars)")
    mdp = bd / "figure_metadata.json"
    if args.skip_classify or len(figures) == 0:
        save_figure_metadata([], config, bctx, mdp)
    else:
        save_figure_metadata(figures, config, bctx, mdp)
    print("[PROMPT GEN] ✅ Phase B complete")
if __name__ == "__main__": main()
