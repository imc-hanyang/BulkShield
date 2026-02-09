"""
vLLM-Based LLM Inference Script for Fraud Detection Explanations

This module reads prompts from JSONL, generates LLM outputs, and saves results.

Features:
    1. Automatic input/output JSONL path resolution
    2. GPU auto-selection based on nvidia-smi free memory
    3. vLLM-first approach with automatic transformers fallback
    4. OOM handling: auto-reduces gpu_mem_util and retries once

Supported Models:
    - meta-llama/Meta-Llama-3.1-70B-Instruct (default)
    - Any HuggingFace model compatible with vLLM

Usage:
    python run_vllm_generate_from_jsonl.py
    python run_vllm_generate_from_jsonl.py --model meta-llama/Llama-2-70B --gpu 0,1
"""


import os
import json
import glob
import argparse
import datetime
import subprocess
import re
from typing import Optional, List, Dict, Any, Tuple


# ===== (선택) HF token =====
# os.environ["HF_TOKEN"] = "hf_xxx"


# -------------------------
# 1) 입력/출력 자동 선택
# -------------------------
DEFAULT_HINT_PATH = "/home/srt/ml_results/transformer/2026-01-25_16-04/LLM_output/llm_prompts_test_attn_en_2026-01-28_15-27.jsonl"
DEFAULT_GLOB = "/home/srt/ml_results/transformer/**/LLM_output/llm_prompts_test_attn_en_*.jsonl"


def pick_latest_jsonl(glob_pattern: str) -> Optional[str]:
    cands = glob.glob(glob_pattern, recursive=True)
    cands = [p for p in cands if os.path.isfile(p)]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def resolve_paths(args) -> (str, str):
    in_jsonl = args.in_jsonl
    if not in_jsonl:
        if os.path.isfile(DEFAULT_HINT_PATH):
            in_jsonl = DEFAULT_HINT_PATH
        else:
            latest = pick_latest_jsonl(DEFAULT_GLOB)
            if latest is None:
                raise FileNotFoundError(
                    "입력 jsonl을 찾지 못했습니다.\n"
                    f"- 힌트 경로: {DEFAULT_HINT_PATH}\n"
                    f"- 검색 패턴: {DEFAULT_GLOB}\n"
                    "직접 --in_jsonl 로 지정하거나, 파일이 위 경로 중 하나에 존재하는지 확인해줘."
                )
            in_jsonl = latest

    out_jsonl = args.out_jsonl
    if not out_jsonl:
        base, ext = os.path.splitext(in_jsonl)
        out_jsonl = f"{base}_with_output{ext}"

    return in_jsonl, out_jsonl


# -------------------------
# 2) JSONL IO
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# 3) GPU 자동 선택 (nvidia-smi 기반)
# -------------------------
def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def get_gpu_free_mem_mb() -> List[Tuple[int, int]]:
    """
    return: [(gpu_index, free_mem_mb), ...]  내림차순 정렬
    실패하면 [] 반환
    """
    rc, out, err = _run_cmd([
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"
    ])
    if rc != 0 or not out.strip():
        return []

    items = []
    for line in out.strip().splitlines():
        # 예: "0, 81234"
        m = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", line)
        if not m:
            continue
        idx = int(m.group(1))
        free = int(m.group(2))
        items.append((idx, free))

    items.sort(key=lambda x: x[1], reverse=True)
    return items


def choose_cuda_visible_devices(mode: str = "auto", num_gpus: int = 0) -> Optional[str]:
    """
    mode:
      - "auto": free mem 큰 GPU부터 num_gpus개 선택 (num_gpus<=0이면 가능한 만큼)
      - "all" : 시스템 GPU 모두
      - "0,1,2": 직접 지정
    return: "0,1,2" 형태 or None (설정 안함)
    """
    mode = (mode or "auto").strip()

    if mode not in ("auto", "all") and re.fullmatch(r"\d+(,\d+)*", mode):
        return mode

    gpus = get_gpu_free_mem_mb()
    if not gpus:
        # nvidia-smi 실패 시 그냥 현재 환경 그대로 사용
        return None

    if mode == "all":
        return ",".join(str(i) for i, _ in gpus)

    # auto
    if num_gpus is None or num_gpus <= 0:
        chosen = [i for i, _ in gpus]
    else:
        chosen = [i for i, _ in gpus[:num_gpus]]
    return ",".join(str(i) for i in chosen)


# -------------------------
# 4) LLM 생성: vLLM 우선, 실패 시 transformers fallback
# -------------------------
def generate_with_llm(
    prompts: List[str],
    model_name: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    seed: int = 0,
    prefer_vllm: bool = True,
    gpu_mem_util: float = 0.90,
) -> List[str]:
    """
    - vLLM이 가능하면 vLLM 사용
    - vLLM init/런타임 에러면 transformers로 fallback
    - vLLM OOM(유사)면 gpu_mem_util 낮춰 1회 재시도
    """

    # ---- 내부 유틸: 결과에서 텍스트만 뽑기 ----
    def _safe_str(x) -> str:
        return "" if x is None else str(x)

    # ---- 4-1) vLLM 시도 ----
    if prefer_vllm:
        try:
            from vllm import LLM, SamplingParams  # CUDA_VISIBLE_DEVICES 설정 이후 import

            # tensor_parallel_size는 "현재 보이는 GPU 개수"로 자동
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if visible:
                tp = len([x for x in visible.split(",") if x.strip() != ""])
            else:
                # CUDA_VISIBLE_DEVICES 미설정이면 vLLM이 알아서 할당하게 두되, tp=1로 두는 편이 안전
                tp = 1

            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
            )

            def _run_vllm(mem_util: float) -> List[str]:
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=mem_util,
                    trust_remote_code=False,
                )
                outs = llm.generate(prompts, sampling)
                texts = []
                for o in outs:
                    # vllm output 구조: RequestOutput -> outputs[0].text
                    if hasattr(o, "outputs") and o.outputs:
                        texts.append(_safe_str(o.outputs[0].text))
                    else:
                        texts.append("")
                return texts

            try:
                return _run_vllm(gpu_mem_util)
            except Exception as e:
                msg = _safe_str(e).lower()
                is_oom_like = ("out of memory" in msg) or ("oom" in msg) or ("cuda" in msg and "memory" in msg)
                if is_oom_like:
                    # 1회 하향 재시도
                    lowered = max(0.60, gpu_mem_util - 0.15)
                    return _run_vllm(lowered)
                # OOM 아닌 에러는 fallback로 넘김
                raise

        except Exception as e:
            # vLLM이 타입에러(파이썬 3.9 union 등)나 환경 문제로 죽어도 fallback
            print(f"[WARN] vLLM init/generate failed. Will fallback to transformers.\n  vLLM error: {e}")

    # ---- 4-2) transformers fallback ----
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # pad_token이 없으면 eos_token으로 대체 (LLaMA 계열에서 흔함)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # dtype 선택: bf16 가능하면 bf16, 아니면 fp16
        # (H100이면 bf16 OK)
        dtype = torch.bfloat16
        if not torch.cuda.is_available():
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()

        outputs: List[str] = []
        batch_size = 2  # 안전하게 작게. (원하면 args로 노출 가능)

        # 간단한 채팅 템플릿 지원 (가능하면 apply_chat_template 사용)
        def _wrap_prompt(p: str) -> str:
            p = p.strip()
            # prompts가 이미 system/user 포함한 완성 프롬프트면 그대로
            return p

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = [_wrap_prompt(x) for x in prompts[i:i + batch_size]]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                if torch.cuda.is_available():
                    enc = {k: v.to(model.device) for k, v in enc.items()}

                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # prompt 부분을 제외하고 디코딩
                prompt_len = enc["input_ids"].shape[1]
                for j in range(gen.shape[0]):
                    new_tokens = gen[j, prompt_len:]
                    outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

        return outputs

    except Exception as e:
        # transformers까지 실패하면 빈 문자열로 채우되, 에러는 확실히 로그
        print(f"[ERROR] transformers fallback also failed.\n  transformers error: {e}")
        return [""] * len(prompts)


# -------------------------
# 5) main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # ✅ 인자 없어도 동작하도록 default=None + 자동 탐색
    parser.add_argument("--in_jsonl", type=str, default=None, help="입력 prompt jsonl (미지정시 자동 선택)")
    parser.add_argument("--out_jsonl", type=str, default=None, help="출력 jsonl (미지정시 자동 생성)")
    parser.add_argument("--overwrite", action="store_true")

    # ✅ 모델/생성 파라미터도 기본값으로 제공 (커맨드 없이도 돌아감)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)

    # ✅ GPU 자동 선택: 기본은 auto + 3개(H100 3대) 우선
    parser.add_argument("--gpu", type=str, default="auto", help="auto|all|0,1,2 형태")
    parser.add_argument("--num_gpus", type=int, default=3, help="gpu=auto일 때 선택할 GPU 개수 (<=0이면 가능한 만큼)")

    # ✅ vLLM 관련
    parser.add_argument("--prefer_vllm", action="store_true", help="vLLM 우선 사용 (기본 True로 처리)")
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)

    args = parser.parse_args()

    # 1) CUDA_VISIBLE_DEVICES 설정 (중요: torch/vllm/transformers import 전에!)
    cuda_vis = choose_cuda_visible_devices(args.gpu, args.num_gpus)
    if cuda_vis is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_vis
        print(f"[INFO] CUDA_VISIBLE_DEVICES set to: {cuda_vis}")
    else:
        print("[INFO] CUDA_VISIBLE_DEVICES unchanged (nvidia-smi unavailable or parsing failed).")

    in_jsonl, out_jsonl = resolve_paths(args)

    if os.path.exists(out_jsonl) and not args.overwrite:
        raise FileExistsError(
            f"출력 파일이 이미 존재합니다: {out_jsonl}\n"
            "덮어쓰려면 --overwrite 를 붙여줘."
        )

    rows = read_jsonl(in_jsonl)

    prompts = []
    idx_map = []
    for i, r in enumerate(rows):
        if "prompt" not in r:
            continue
        prompts.append(r["prompt"])
        idx_map.append(i)

    if not prompts:
        raise ValueError(f"prompt 필드를 가진 row가 없습니다: {in_jsonl}")

    # prefer_vllm 기본 True로 하고 싶으면 여기서 강제
    prefer_vllm = True

    outputs = generate_with_llm(
        prompts=prompts,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prefer_vllm=prefer_vllm,
        gpu_mem_util=args.gpu_mem_util,
    )

    for out, i in zip(outputs, idx_map):
        rows[i]["llm_output"] = out
        rows[i]["llm_output_generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    write_jsonl(out_jsonl, rows)

    print(f"[OK] in : {in_jsonl}")
    print(f"[OK] out: {out_jsonl}")
    print(f"[OK] rows: {len(rows)}  generated: {len(outputs)}")


if __name__ == "__main__":
    main()
