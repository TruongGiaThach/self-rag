#!/usr/bin/env python3
"""
SELF-RAG Experiment Runner (Forked Repo) — vLLM + INT8
=======================================================
Pipeline tự động cho workspace đã fork:
  1. Setup venv + cài packages (torch, vllm>=0.3.0, transformers, bitsandbytes)
  2. Download PopQA dataset + enrich với Wikipedia passages
  3. Check server compatibility (CUDA, GPU VRAM)
  4. Run 3 scenarios qua script gốc trong repo:
       A. Llama-2-7B — No Retrieval
       B. Llama-2-7B — Standard RAG (K=5 docs)
       C. SELF-RAG 7B — Adaptive Retrieval (δ=0.2, Beam=2)
  5. Tổng hợp kết quả bảng so sánh

Yêu cầu:
  - GPU VRAM >= 8GB (khuyến nghị 12GB)
  - Python 3.9+, CUDA 11.8 hoặc 12.x
  - HuggingFace token (cho Llama-2 gated model)

Usage:
  python run_experiment.py --hf-token hf_xxxx
  python run_experiment.py --hf-token hf_xxxx --skip-setup
  python run_experiment.py --hf-token hf_xxxx --skip-scenarios A,B
  python run_experiment.py --hf-token hf_xxxx --check-only
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd, cwd=None, check=True, env=None):
    """Chạy shell command với realtime output."""
    print(f"\n💻 [CMD] {cmd}")
    import os
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(cmd, shell=True, cwd=cwd, stdout=None, stderr=None, env=run_env)
    if check and result.returncode != 0:
        print(f"❌ [ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def run_capture(cmd, cwd=None, env=None):
    """Chạy command và capture stdout."""
    import os
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=run_env
    )
    return result.stdout.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PopQA Download Script
# ══════════════════════════════════════════════════════════════════════════════

POPQA_DOWNLOAD_SCRIPT = '''
import json, ast, sys, time, urllib.request, urllib.parse
from pathlib import Path

output_path = Path(sys.argv[1])
ndocs = int(sys.argv[2])

print("[DATA] Downloading PopQA from HuggingFace...")
try:
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    print(f"[DATA] Loaded {len(ds)} samples")
except Exception as e:
    print(f"[ERROR] Cannot load dataset: {e}")
    sys.exit(1)

def parse_answers(item):
    """Parse answers từ nhiều format khác nhau."""
    for field in ("possible_answers", "answers", "answer"):
        val = item.get(field)
        if val is None:
            continue
        if isinstance(val, list):
            return [str(v) for v in val]
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except:
            pass
        return [str(val)]
    return []

# Fetch Wikipedia passages
WP_SEARCH = "https://en.wikipedia.org/w/api.php"
WP_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
HEADERS = {"User-Agent": "SelfRAG-Experiment/1.0"}

def wp_search(query, n=5):
    """Search Wikipedia, return article titles."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": n,
        "format": "json", "srprop": "",
    })
    try:
        req = urllib.request.Request(f"{WP_SEARCH}?{params}", headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        return [h["title"] for h in data["query"]["search"]]
    except:
        return []

def wp_summary(title):
    """Fetch Wikipedia article summary."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    try:
        req = urllib.request.Request(WP_SUMMARY.format(encoded), headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=10)
        d = json.loads(resp.read())
        return {"title": d.get("title", title), "text": d.get("extract", "")}
    except:
        return {"title": title, "text": ""}

def get_ctxs(item, n):
    """Build Wikipedia contexts cho một câu hỏi."""
    subj = item.get("subj", item.get("s_aliases", [None])[0] if item.get("s_aliases") else None)
    query = subj if subj else item.get("question", "")
    titles = wp_search(query, n=n)
    ctxs = []
    for title in titles[:n]:
        ctx = wp_summary(title)
        ctxs.append(ctx)
        time.sleep(0.05)  # rate limit
    while len(ctxs) < n:
        ctxs.append({"title": "", "text": ""})
    return ctxs

output_path.parent.mkdir(parents=True, exist_ok=True)
records = []
total = len(ds)
for idx, item in enumerate(ds):
    if idx % 50 == 0:
        print(f"[DATA] Processing {idx}/{total}...")
    answers = parse_answers(item)
    question = item.get("question", item.get("query", ""))
    ctxs = get_ctxs(item, ndocs)
    records.append({
        "question": question,
        "instruction": question,
        "answers": answers,
        "ctxs": ctxs,
    })

with open(output_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print(f"[DATA] ✅ Saved {len(records)} records to {output_path}")
'''


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Setup Environment
# ══════════════════════════════════════════════════════════════════════════════

def setup_environment(workspace_dir: Path, hf_token: str, skip: bool):
    """Setup venv + cài packages cần thiết."""
    venv_dir = workspace_dir / "venv_selfrag"
    venv_bin = venv_dir / "bin"

    if skip:
        print("\n⏭️  [SKIP] Bỏ qua setup, sử dụng venv có sẵn")
        if not venv_bin.exists():
            print(f"❌ [ERROR] --skip-setup nhưng không tìm thấy {venv_bin}")
            sys.exit(1)
        return venv_bin

    print("\n" + "=" * 70)
    print("📦 SETUP: Cài đặt môi trường Python")
    print("=" * 70)

    # 1. Tạo venv
    if not venv_dir.exists():
        print(f"🔨 Tạo virtual environment...")
        run(f"python3 -m venv {venv_dir}", cwd=workspace_dir)
    else:
        print(f"✓ Venv đã tồn tại")
    
    # Show venv info
    print(f"\n📍 Virtual environment:")
    print(f"  Path: {venv_dir}")
    print(f"  Python: {venv_bin}/python")
    print(f"  Pip: {venv_bin}/pip")

    # 2. Upgrade pip
    print(f"🔨 Upgrade pip, setuptools, wheel...")
    run(f"{venv_bin}/pip install --quiet --upgrade pip setuptools wheel", cwd=workspace_dir)

    # 3. Cài PyTorch TRƯỚC (vLLM cần torch để build)
    print(f"🔨 Cài PyTorch + CUDA 11.8...")
    print(f"    Note: Đây là dependency cho vLLM")
    run(
        f"{venv_bin}/pip install --quiet "
        "torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118",
        cwd=workspace_dir
    )

    # 4. Cài vLLM 0.4.2 (stable, không cần pyairports)
    print(f"🔨 Cài vLLM 0.4.2...")
    # Use vLLM 0.4.2 - stable version without guided decoding dependencies
    # vLLM 0.5+ requires outlines/pyairports which has dependency issues
    run(f"{venv_bin}/pip install --quiet 'vllm==0.4.2'", cwd=workspace_dir)

    # 4.5. Fix missing dependencies (only needed for vLLM <= 0.4.x)
    print(f"🔨 Cài dependencies cho vLLM...")
    run(f"{venv_bin}/pip install 'ray>=2.5.1' 'xformers>=0.0.23'", cwd=workspace_dir, check=False)

    # 5. Transformers, bitsandbytes, accelerate
    print(f"🔨 Cài transformers, bitsandbytes, accelerate...")
    run(
        f"{venv_bin}/pip install --quiet "
        "'transformers>=4.36.0' "
        "'bitsandbytes>=0.41.0' "
        "'accelerate>=0.25.0' "
        "sentencepiece "
        "'datasets>=2.15.0' "
        "jsonlines tqdm numpy scipy spacy backoff 'openai>=1.0'",
        cwd=workspace_dir
    )

    # 6. Spacy model
    print(f"🔨 Tải spacy model...")
    run(f"{venv_bin}/python -m spacy download en_core_web_sm", cwd=workspace_dir, check=False)

    # 7. HuggingFace token (set environment variable instead of CLI login)
    if hf_token:
        print(f"🔨 Setup HuggingFace token...")
        token_file = workspace_dir / ".hf_token"
        token_file.write_text(hf_token)
        print(f"  ✓ Token saved to {token_file}")
        print(f"  ✓ Export: HF_TOKEN will be set when running experiments")

    # 8. Final verification
    print(f"\n🔍 Verify venv setup...")
    cuda_env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    vllm_check = run_capture(
        f"{venv_bin}/python -c \"import vllm; print(vllm.__version__)\"",
        env=cuda_env
    )
    if vllm_check:
        print(f"  ✓ vLLM version: {vllm_check}")
        if not vllm_check.startswith("0.4"):
            print(f"  ⚠️  Warning: Expected vLLM 0.4.x for stability")
    else:
        print(f"  ⚠️ Cannot import vllm!")

    print("\n✅ Setup hoàn tất!")
    return venv_bin


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Download PopQA Dataset
# ══════════════════════════════════════════════════════════════════════════════

def prepare_popqa(workspace_dir: Path, venv_bin: Path, ndocs: int) -> Path:
    """Download PopQA dataset nếu chưa có."""
    data_dir = workspace_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "popqa_with_ctxs.jsonl"

    if output_path.exists():
        count_str = run_capture(f"wc -l < {output_path}")
        count = int(count_str) if count_str.isdigit() else 0
        if count > 100:
            print(f"\n✓ PopQA dataset đã có ({count} samples)")
            return output_path

    print("\n" + "=" * 70)
    print("📥 DOWNLOAD: PopQA + Wikipedia passages")
    print("=" * 70)

    script_path = workspace_dir / "_download_popqa.py"
    script_path.write_text(POPQA_DOWNLOAD_SCRIPT)
    run(f"{venv_bin}/python {script_path} {output_path} {ndocs}", cwd=workspace_dir)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Check Server
# ══════════════════════════════════════════════════════════════════════════════

def check_server(workspace_dir: Path):
    """Kiểm tra CUDA, GPU VRAM."""
    print("\n" + "=" * 70)
    print("🔍 CHECK: Server Compatibility")
    print("=" * 70)

    # NVIDIA Driver
    cuda_version = run_capture("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if cuda_version:
        print(f"  ✓ NVIDIA Driver: {cuda_version}")
    else:
        print(f"  ⚠ NVIDIA Driver không phát hiện")

    # GPU VRAM (check all GPUs, find max FREE memory)
    vram_info = run_capture("nvidia-smi --query-gpu=name,memory.free --format=csv,noheader")
    if vram_info:
        print(f"  ✓ GPU: {vram_info.replace('MiB', 'MiB free')}")
        # Parse all GPUs and find max FREE VRAM
        max_free_gb = 0
        for line in vram_info.strip().split('\n'):
            if "MiB" in line:
                try:
                    vram_mb = int(line.split(",")[1].strip().split()[0])
                    max_free_gb = max(max_free_gb, vram_mb / 1024)
                except:
                    pass
        if max_free_gb > 0:
            if max_free_gb < 7:
                print(f"  ⚠ Max FREE VRAM {max_free_gb:.1f}GB < 7GB — có thể không đủ cho INT8!")
            elif max_free_gb >= 10:
                print(f"  ✅ Max FREE VRAM {max_free_gb:.1f}GB >= 10GB — đủ thoải mái!")
            else:
                print(f"  ✓ Max FREE VRAM {max_free_gb:.1f}GB — vừa đủ cho INT8 7B model")
        else:
            print(f"  ⚠ Không parse được VRAM free")
    else:
        print(f"  ⚠ Không đọc được VRAM info")

    # Python
    py_ver = run_capture("python3 --version")
    print(f"  ✓ Python: {py_ver}")

    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Run Scenarios
# ══════════════════════════════════════════════════════════════════════════════

def _already_done(output_path: Path) -> bool:
    """Check xem kết quả đã có chưa."""
    if not output_path.exists():
        return False
    try:
        with open(output_path) as f:
            d = json.load(f)
        return "metric_mean" in d or "accuracy" in d
    except:
        return False


def run_scenario_a(workspace_dir: Path, venv_bin: Path, data_path: Path, args) -> Path:
    """Scenario A: Llama-2-7B No Retrieval (vLLM)."""
    out = workspace_dir / "outputs" / "result_A_no_retrieval.json"
    if _already_done(out):
        print("\n⏭️  Scenario A: Kết quả đã có, bỏ qua")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("🚀 SCENARIO A: Llama-2-7B No Retrieval")
    print("=" * 70)

    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    token_file = workspace_dir / ".hf_token"
    if token_file.exists():
        env["HF_TOKEN"] = token_file.read_text().strip()

    run(
        f"{venv_bin}/python retrieval_lm/run_baseline_lm.py "
        f"--model_name {args.llama_model} "
        f"--input_file {data_path} "
        f"--result_fp {out} "
        f"--mode vanilla "
        f"--metric match "
        f"--max_new_tokens {args.max_new_tokens} "
        f"--dtype half "
        f"--batch_size 1 "
        f"--task popqa "
        f"--download_dir {workspace_dir}/model_cache",
        cwd=workspace_dir,
        env=env,
    )
    return out


def run_scenario_b(workspace_dir: Path, venv_bin: Path, data_path: Path, args) -> Path:
    """Scenario B: Standard RAG — Always retrieve K docs (vLLM)."""
    out = workspace_dir / "outputs" / "result_B_standard_rag.json"
    if _already_done(out):
        print("\n⏭️  Scenario B: Kết quả đã có, bỏ qua")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print(f"🚀 SCENARIO B: Standard RAG (Always K={args.ndocs})")
    print("=" * 70)

    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    token_file = workspace_dir / ".hf_token"
    if token_file.exists():
        env["HF_TOKEN"] = token_file.read_text().strip()

    run(
        f"{venv_bin}/python retrieval_lm/run_baseline_lm.py "
        f"--model_name {args.llama_model} "
        f"--input_file {data_path} "
        f"--result_fp {out} "
        f"--mode retrieval "
        f"--top_n {args.ndocs} "
        f"--metric match "
        f"--max_new_tokens {args.max_new_tokens} "
        f"--dtype half "
        f"--batch_size 1 "
        f"--task popqa "
        f"--download_dir {workspace_dir}/model_cache",
        cwd=workspace_dir,
        env=env,
    )
    return out


def run_scenario_c(workspace_dir: Path, venv_bin: Path, data_path: Path, args) -> Path:
    """Scenario C: SELF-RAG — Adaptive Retrieval + Critique (vLLM)."""
    out = workspace_dir / "outputs" / "result_C_selfrag.json"
    if _already_done(out):
        print("\n⏭️  Scenario C: Kết quả đã có, bỏ qua")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print(f"🚀 SCENARIO C: SELF-RAG Adaptive (δ={args.threshold}, Beam={args.beam_width})")
    print("=" * 70)

    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    token_file = workspace_dir / ".hf_token"
    if token_file.exists():
        env["HF_TOKEN"] = token_file.read_text().strip()

    run(
        f"{venv_bin}/python retrieval_lm/run_short_form.py "
        f"--model_name {args.selfrag_model} "
        f"--input_file {data_path} "
        f"--output_file {out} "
        f"--task popqa "
        f"--ndocs {args.ndocs} "
        f"--threshold {args.threshold} "
        f"--use_groundness "
        f"--use_utility "
        f"--w_rel {args.w_rel} "
        f"--w_sup {args.w_sup} "
        f"--w_use {args.w_use} "
        f"--mode adaptive_retrieval "
        f"--metric match "
        f"--max_new_tokens {args.max_new_tokens} "
        f"--dtype half "
        f"--beam_width {args.beam_width} "
        f"--download_dir {workspace_dir}/model_cache",
        cwd=workspace_dir,
        env=env,
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Summarize Results
# ══════════════════════════════════════════════════════════════════════════════

def read_accuracy(result_file: Path) -> tuple:
    """Đọc accuracy từ result file."""
    if not result_file.exists():
        return None, 0, "file not found"
    try:
        with open(result_file) as f:
            d = json.load(f)

        # run_short_form.py format
        if "metric_mean" in d:
            acc = float(d["metric_mean"]) * 100
            n = len(d.get("preds", []))
            freq = d.get("Retrieval Frequencies", "N/A")
            return acc, n, f"Retrieval freq: {freq}"

        # run_baseline_lm.py format (list)
        if isinstance(d, list):
            match_count = sum(
                1 for item in d
                if any(a.lower() in str(item.get("output", "")).lower()
                       for a in item.get("answers", item.get("golds", [])))
            )
            acc = 100.0 * match_count / len(d) if d else 0.0
            return acc, len(d), ""

        # Custom format
        if "accuracy" in d:
            return float(d["accuracy"]), int(d.get("n_samples", 0)), ""

        return None, 0, "unknown format"
    except Exception as e:
        return None, 0, str(e)


def summarize_results(scenarios):
    """In bảng tổng hợp kết quả."""
    print("\n" + "=" * 80)
    print("  📊 KẾT QUẢ THỰC NGHIỆM SELF-RAG trên PopQA (Accuracy = match/inclusion)")
    print("=" * 80)
    print(f"  {'Kịch bản':<45} {'Accuracy':>10}   {'Ghi chú'}")
    print("-" * 80)
    for label, path in scenarios:
        acc, n, note = read_accuracy(path)
        if acc is None:
            print(f"  {label:<45} {'N/A':>10}   {note}")
        else:
            print(f"  {label:<45} {acc:>9.2f}%   n={n}  {note}")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SELF-RAG Experiment Runner — vLLM + INT8 (12GB VRAM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Ví dụ sử dụng:
          # Chạy đầy đủ 3 kịch bản
          python run_experiment.py --hf-token hf_xxxx
          
          # Chỉ chạy SELF-RAG (bỏ qua A, B)
          python run_experiment.py --hf-token hf_xxxx --skip-scenarios A,B
          
          # Đã setup venv lần trước
          python run_experiment.py --hf-token hf_xxxx --skip-setup
          
          # Chỉ kiểm tra dataset + model cache
          python run_experiment.py --check-only
        """)
    )
    
    # Required
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token cho Llama-2 gated model")
    
    # Optional paths
    parser.add_argument("--workspace", type=str, default=None,
                        help="Workspace directory (repo fork). Default: script location")
    
    # Dataset & model config
    parser.add_argument("--ndocs", type=int, default=5,
                        help="Số tài liệu truy xuất K (default: 5)")
    parser.add_argument("--max-new-tokens", type=int, default=15,
                        help="Max tokens generation (default: 15)")
    parser.add_argument("--llama-model", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Llama-2 baseline model")
    parser.add_argument("--selfrag-model", type=str,
                        default="selfrag/selfrag_llama2_7b",
                        help="SELF-RAG model")
    
    # SELF-RAG hyperparameters
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Threshold δ cho adaptive retrieval (default: 0.2)")
    parser.add_argument("--beam-width", type=int, default=2,
                        help="Beam width cho SELF-RAG (default: 2)")
    parser.add_argument("--w-rel", type=float, default=1.0,
                        help="Weight ISREL (default: 1.0)")
    parser.add_argument("--w-sup", type=float, default=1.0,
                        help="Weight ISSUP (default: 1.0)")
    parser.add_argument("--w-use", type=float, default=0.5,
                        help="Weight ISUSE (default: 0.5)")
    
    # Control flags
    parser.add_argument("--skip-setup", action="store_true",
                        help="Bỏ qua setup venv (đã có sẵn)")
    parser.add_argument("--skip-scenarios", type=str, default="",
                        help="Bỏ qua scenarios (ví dụ: A,B)")
    parser.add_argument("--check-only", action="store_true",
                        help="Chỉ check dataset + model cache, không chạy")
    
    args = parser.parse_args()

    # Xác định workspace
    if args.workspace:
        workspace_dir = Path(args.workspace).resolve()
    else:
        workspace_dir = Path(__file__).parent.resolve()

    skip_set = {s.strip().upper() for s in args.skip_scenarios.split(",") if s.strip()}

    # Banner
    print("\n" + "=" * 80)
    print("  🚀 SELF-RAG Experiment Runner (Forked Repo)")
    print("=" * 80)
    print(f"  Workspace       : {workspace_dir}")
    print(f"  Llama-2 model   : {args.llama_model}")
    print(f"  SELF-RAG model  : {args.selfrag_model}")
    print(f"  ndocs K         : {args.ndocs}")
    print(f"  threshold δ     : {args.threshold}")
    print(f"  beam width      : {args.beam_width}")
    print(f"  Weights         : w_rel={args.w_rel}, w_sup={args.w_sup}, w_use={args.w_use}")
    print(f"  max_new_tokens  : {args.max_new_tokens}")
    print(f"  vLLM version    : 0.4.2 (stable, FP16/INT8 support)")
    print(f"  Skip setup      : {args.skip_setup}")
    print(f"  Skip scenarios  : {skip_set or 'none'}")
    print(f"  Check only      : {args.check_only}")
    print("=" * 80)

    # ─────────────────────────────────────────
    # Pipeline Steps
    # ─────────────────────────────────────────

    # Step 1: Setup venv
    venv_bin = setup_environment(workspace_dir, args.hf_token, skip=args.skip_setup)

    # Step 2: Check server
    check_server(workspace_dir)

    # Step 3: Download dataset
    popqa_path = prepare_popqa(workspace_dir, venv_bin, args.ndocs)

    # Validate dataset
    print(f"\n🔍 Validate dataset: {popqa_path}")
    if popqa_path.exists():
        line_count_str = run_capture(f"wc -l < {popqa_path}")
        line_count = int(line_count_str) if line_count_str.isdigit() else 0
        print(f"  ✓ Dataset: {line_count} samples")
        
        # Show sample
        first_line = run_capture(f"head -1 {popqa_path}")
        if first_line:
            try:
                sample = json.loads(first_line)
                print(f"  ✓ Sample: '{sample.get('question', 'N/A')[:50]}...'")
                print(f"    → answers: {sample.get('answers', [])[:2]}")
                print(f"    → ctxs: {len(sample.get('ctxs', []))} docs")
            except:
                pass
    else:
        print(f"  ❌ Dataset không tồn tại!")
        sys.exit(1)

    # Check model cache
    model_cache_dir = workspace_dir / "model_cache"
    print(f"\n🔍 Model cache: {model_cache_dir}")
    if model_cache_dir.exists():
        cached = list(model_cache_dir.glob("models--*"))
        if cached:
            print(f"  ✓ {len(cached)} model(s) cached")
            for m in cached[:3]:
                print(f"    → {m.name}")
        else:
            print(f"  ⚠ Cache trống → sẽ download (~13GB/model)")
    else:
        print(f"  ⚠ Chưa có cache → sẽ download lần đầu (~13GB/model)")

    if args.check_only:
        print("\n✅ Check-only mode, thoát.")
        return

    # Step 4: Run scenarios
    output_dir = workspace_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_a = output_dir / "result_A_no_retrieval.json"
    if "A" not in skip_set:
        out_a = run_scenario_a(workspace_dir, venv_bin, popqa_path, args)
    else:
        print("\n⏭️  SKIP Scenario A")

    out_b = output_dir / "result_B_standard_rag.json"
    if "B" not in skip_set:
        out_b = run_scenario_b(workspace_dir, venv_bin, popqa_path, args)
    else:
        print("\n⏭️  SKIP Scenario B")

    out_c = output_dir / "result_C_selfrag.json"
    if "C" not in skip_set:
        out_c = run_scenario_c(workspace_dir, venv_bin, popqa_path, args)
    else:
        print("\n⏭️  SKIP Scenario C")

    # Step 5: Summarize
    summarize_results([
        ("A: Llama-2-7B (No Retrieval)", out_a),
        (f"B: Standard RAG (Always K={args.ndocs})", out_b),
        (f"C: SELF-RAG 7B (Adaptive δ={args.threshold}, Beam={args.beam_width})", out_c),
    ])

    print(f"\n✅ Hoàn tất! Kết quả: {workspace_dir / 'outputs'}\n")


if __name__ == "__main__":
    main()
