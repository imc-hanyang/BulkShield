"""
LLM Prompt Builder with Attention-Based Event Selection

This module generates LLM prompts for fraud detection explanation by:
1. Running trained Transformer model on test set to get risk scores
2. Computing per-user window statistics (refund rate, amounts, counts)
3. Selecting top-K important events via CLS token attention weights
4. Building structured English prompts for LLM interpretation

IMPORTANT: This script does NOT call LLM - it only builds prompts and saves
them to JSONL format. This separation prevents CUDA/vLLM memory conflicts.

Output:
    JSONL file with: test_path, y_true, risk_score, window_summary,
    key_events, prompt, attn_topk_indices_meta

Usage:
    CUDA_VISIBLE_DEVICES=0 python build_prompts_attn_jsonl.py
"""


import os
import json
import csv
import math
import datetime
from typing import Dict, Any, List, Tuple, Optional
import types
import inspect
import pdb

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===== project import =====
from models.transformer_model import SRTTransformerClassifier, CFG, CATEGORICAL_COLS, NUMERIC_COLS
from data_utils import load_train_data, load_test_data

# ===== prompt schema (optional) =====
try:
    from prompt_schema import COL_ALIAS  # noqa: F401
except Exception:
    COL_ALIAS = {}

# Token
VALID_TOKEN = "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"
os.environ["HF_TOKEN"] = VALID_TOKEN
hf_token = VALID_TOKEN

# NOTE: 이 스크립트는 Transformer만 돌릴 거라서,
# CUDA_VISIBLE_DEVICES는 여기서 원하는대로 설정해도 됨.
# 예) Transformer는 GPU 3에서만 돌리고 싶다면:
#   CUDA_VISIBLE_DEVICES=3 python build_prompts_attn_jsonl.py
#
# 코드 안에서 박아두고 싶으면 아래처럼:
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# =========================================================
# 0) Utils
# =========================================================
TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
]
DOW_EN = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

def fast_parse_ts(ts_str: str):
    if not ts_str:
        return None
    for fmt in TS_FORMATS:
        try:
            return datetime.datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None

def safe_int(x, default=-1):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except:
        return default

def safe_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except:
        return default

def fmt_money_en(v: float) -> str:
    try:
        return f"₩{float(v):,.0f}"
    except:
        return "₩0"

def fmt_min_en(v: float) -> str:
    try:
        return f"{float(v):.0f} min"
    except:
        return "0 min"

def fmt_pct(v: float) -> str:
    try:
        return f"{float(v):.1f}%"
    except:
        return "0.0%"

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_date_str(dt_obj: Optional[datetime.datetime]) -> str:
    if dt_obj is None:
        return ""
    try:
        return dt_obj.strftime("%Y-%m-%d")
    except:
        return ""

# =========================================================
# 0.6) Station code map
# =========================================================
def load_station_map(csv_path="/home/srt/Dataset/feature/station_code_map.csv"):
    m = {}
    if not os.path.exists(csv_path):
        return m
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = (r.get("예발역코드") or "").strip()
            name = (r.get("예발역") or "").strip()
            if code:
                m[code] = name
                m[code.lstrip("0")] = name
    return m

def fmt_station_name(code, station_map):
    if code is None:
        return ""
    s = str(code).strip()
    name = station_map.get(s) or station_map.get(s.lstrip("0"))
    return f"{name}" if name else s

# =========================================================
# 1) Vocabs load / fallback build
# =========================================================
def load_vocabs_if_exists(save_dir):
    vocab_json = os.path.join(save_dir, "vocabs.json")
    if os.path.exists(vocab_json):
        with open(vocab_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def build_vocabs_on_the_fly(paths, limit=20000):
    vocabs = {c: {} for c in CATEGORICAL_COLS}
    sample_paths = paths[:limit] if len(paths) > limit else paths

    for p in tqdm(sample_paths, desc="Vocab Building (fallback)"):
        try:
            with open(p, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    for c in CATEGORICAL_COLS:
                        v = safe_int(r.get(c, ""), default=-1)
                        if v not in vocabs[c]:
                            vocabs[c][v] = len(vocabs[c]) + 1  # 1-based
        except:
            pass
    return vocabs

# =========================================================
# 2) Dataset: model input + meta 반환
# =========================================================
class TransformerDatasetWithMeta(Dataset):
    def __init__(self, input_paths, labels, vocabs, cfg: CFG):
        self.input_paths = input_paths
        self.labels = labels
        self.vocabs = vocabs
        self.cfg = cfg
        self.max_len = cfg.max_len

        self.meta_cols = [
            "timestamp",
            "dep_station_id", "arr_station_id", "route_id", "train_id",
            "action_type", "seat_cnt",
            "buy_amt", "refund_amt", "cancel_fee",
            "route_dist_km", "travel_time",
            "lead_time_buy", "lead_time_ref", "hold_time",
            "dep_dow", "dep_hour",
            "route_buy_cnt",
            "fwd_dep_hour_median", "fwd_dep_dow_median",
            "rev_buy_cnt", "rev_ratio",
            "completed_fwd_cnt",
            "completed_fwd_dep_interval_median",
            "completed_fwd_dep_hour_median",
            "completed_fwd_dep_dow_median",
            "completed_rev_cnt",
            "completed_rev_dep_interval_median",
            "completed_rev_dep_hour_median",
            "completed_rev_dep_dow_median",
            "unique_route_cnt",
            "rev_dep_hour_median", "rev_dep_dow_median",
            "rev_return_gap",
            "overlap_cnt", "same_route_cnt", "rev_route_cnt",
            "repeat_interval",
            "adj_seat_refund_flag",
            "recent_ref_cnt", "recent_ref_amt", "recent_ref_rate",
        ]

    def __len__(self):
        return len(self.input_paths)

    def _enc_cat(self, col: str, v: int) -> int:
        return self.vocabs.get(col, {}).get(v, 0)

    def __getitem__(self, idx):
        seq_path = self.input_paths[idx]
        label = self.labels[idx]

        rows = []
        try:
            with open(seq_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ts_val = fast_parse_ts(r.get("timestamp", "")) if "timestamp" in r else None
                    r["_ts"] = ts_val
                    rows.append(r)
        except:
            rows = []

        if not rows:
            T = 1
            cat = torch.zeros(T, len(CATEGORICAL_COLS), dtype=torch.long)
            num = torch.zeros(T, len(NUMERIC_COLS), dtype=torch.float32)
            delta = torch.zeros(T, dtype=torch.long)
            meta = [None] * T
            # return {"cat": cat, "num": num, "delta": delta, "meta": meta, "y": torch.tensor(label, dtype=torch.long)}
            return {
                "cat": cat,
                "num": num,
                "delta": delta,
                "meta": meta,
                "y": torch.tensor(label, dtype=torch.long),
                "seq_path": seq_path,  # ✅ 추가
            }

        max_ts = datetime.datetime.max
        rows.sort(key=lambda x: x["_ts"] if x.get("_ts") else max_ts)

        if len(rows) > self.max_len:
            rows = rows[-self.max_len:]

        T = len(rows)
        cat_feats, num_feats, timestamps = [], [], []
        meta = []

        for r in rows:
            meta.append({c: r.get(c, "") for c in self.meta_cols})

            c_vec = []
            for c in CATEGORICAL_COLS:
                v = safe_int(r.get(c, ""), default=-1)
                c_vec.append(self._enc_cat(c, v))
            cat_feats.append(c_vec)

            n_vec = []
            for c in NUMERIC_COLS:
                raw = safe_float(r.get(c, ""), default=0.0)
                val = math.log1p(abs(raw))
                if raw < 0:
                    val = -val
                n_vec.append(val)
            num_feats.append(n_vec)

            timestamps.append(r.get("_ts", None))

        cat = torch.tensor(cat_feats, dtype=torch.long)
        num = torch.tensor(num_feats, dtype=torch.float32)

        delta_vals = [0] * T
        if T > 1 and any(timestamps):
            delta_vals[0] = 0
            for i in range(1, T):
                t_curr = timestamps[i]
                t_prev = timestamps[i - 1]
                if t_curr and t_prev:
                    diff_min = (t_curr - t_prev).total_seconds() / 60.0
                else:
                    diff_min = 0.0
                diff_min = clamp(diff_min, 0.0, 60.0 * 48)
                b = int(diff_min // self.cfg.delta_bucket_size_min)
                b = clamp(b, 0, self.cfg.delta_max_bucket)
                delta_vals[i] = int(b)

        delta = torch.tensor(delta_vals, dtype=torch.long)
        # return {"cat": cat, "num": num, "delta": delta, "meta": meta, "y": torch.tensor(label, dtype=torch.long)}
        return {
            "cat": cat,
            "num": num,
            "delta": delta,
            "meta": meta,
            "y": torch.tensor(label, dtype=torch.long),
            "seq_path": seq_path,  # ✅ 추가
        }
def collate_fn_with_meta(batch):
    maxT = max(x["cat"].shape[0] for x in batch)
    n_cat = batch[0]["cat"].shape[1]
    n_num = batch[0]["num"].shape[1]
    B = len(batch)

    cat_pad = torch.zeros(B, maxT, n_cat, dtype=torch.long)
    num_pad = torch.zeros(B, maxT, n_num, dtype=torch.float32)
    delta_pad = torch.zeros(B, maxT, dtype=torch.long)
    pad_mask = torch.ones(B, maxT, dtype=torch.bool)  # True=PAD
    ys = torch.zeros(B, dtype=torch.long)
    meta_pad = [[None for _ in range(maxT)] for _ in range(B)]

    seq_paths = [None] * B  # ✅ 추가

    for i, x in enumerate(batch):
        T = x["cat"].shape[0]
        cat_pad[i, :T] = x["cat"]
        num_pad[i, :T] = x["num"]
        delta_pad[i, :T] = x["delta"]
        pad_mask[i, :T] = False
        ys[i] = x["y"]
        for t in range(T):
            meta_pad[i][t] = x["meta"][t]

        seq_paths[i] = x.get("seq_path")  # ✅ 추가

    return {
        "cat": cat_pad,
        "num": num_pad,
        "delta": delta_pad,
        "pad_mask": pad_mask,
        "y": ys,
        "meta": meta_pad,
        "seq_path": seq_paths,  # ✅ 추가
    }

# =========================================================
# 3) Attention extraction: runtime patch
# =========================================================
def patch_encoder_layers_to_save_attn(model) -> bool:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        print("[WARN] model.encoder.layers not found. Skip patch.")
        return False

    patched_any = False
    for li, layer in enumerate(model.encoder.layers):
        if not hasattr(layer, "self_attn") or not hasattr(layer, "dropout1"):
            continue

        if hasattr(layer, "last_attn_weights"):
            patched_any = True
            continue

        layer.last_attn_weights = None  # (B,H,S,S)

        def _sa_block_patched(self, x, attn_mask, key_padding_mask, is_causal=False):
            attn_output, attn_weights = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
            self.last_attn_weights = attn_weights
            return self.dropout1(attn_output)

        layer._sa_block = types.MethodType(_sa_block_patched, layer)
        patched_any = True

    if patched_any:
        print("[INFO] Encoder layers patched to save attention weights.")
    else:
        print("[WARN] No encoder layers patched.")
    return patched_any

def forward_with_optional_attn(model, cat, num, delta, pad_mask, return_attn=False):
    sig = inspect.signature(model.forward)
    supports_return_attn = ("return_attn" in sig.parameters)

    if supports_return_attn:
        out = model(cat, num, delta, pad_mask, return_attn=return_attn)
        return out

    logits = model(cat, num, delta, pad_mask)

    if not return_attn:
        return logits, None

    attn_last = None
    try:
        attn_last = model.encoder.layers[-1].last_attn_weights
    except Exception:
        attn_last = None

    return logits, attn_last

def select_topk_events_by_cls_attention(
    attn_last: Optional[torch.Tensor],
    pad_mask: torch.Tensor,
    k: int = 3
) -> List[List[int]]:
    B, T = pad_mask.shape
    topk_idxs: List[List[int]] = []

    if attn_last is None or not isinstance(attn_last, torch.Tensor):
        for b in range(B):
            valid_evt = torch.where(~pad_mask[b])[0].tolist()
            topk_idxs.append(valid_evt[-k:] if len(valid_evt) >= k else valid_evt)
        return topk_idxs

    S = attn_last.size(-1)
    if S == T + 1:
        cls_col = torch.zeros((B, 1), dtype=torch.bool, device=pad_mask.device)
        pad_mask_aligned = torch.cat([cls_col, pad_mask], dim=1)
        has_cls = True
    elif S == T:
        pad_mask_aligned = pad_mask
        has_cls = False
    else:
        minL = min(S, T)
        pad_mask_aligned = pad_mask[:, :minL]
        attn_last = attn_last[:, :, :minL, :minL]
        S = minL
        has_cls = False

    pdb.set_trace()
    cls_attn = attn_last[:, :, 0, :].mean(dim=1)  # (B,S)

    scores = cls_attn.clone()
    scores[pad_mask_aligned] = -1e9
    if has_cls and S > 0:
        scores[:, 0] = -1e9

    for b in range(B):
        valid_count = int((~pad_mask_aligned[b]).sum().item())
        effective_valid = valid_count - (1 if has_cls else 0)
        kk = min(k, max(0, effective_valid))

        if kk <= 0:
            valid_evt = torch.where(~pad_mask[b])[0].tolist()
            topk_idxs.append(valid_evt[-k:] if len(valid_evt) >= k else valid_evt)
            continue

        idx_token = torch.topk(scores[b], k=kk, largest=True).indices.tolist()
        if has_cls:
            mapped = [t - 1 for t in idx_token if t > 0]
        else:
            mapped = idx_token
        mapped = [t for t in mapped if 0 <= t < T]
        topk_idxs.append(mapped)

    return topk_idxs

# =========================================================
# 4) Summarize a single event (English)
# =========================================================
def summarize_event_row_en(m: Dict[str, Any], station_map: dict) -> str:
    if m is None:
        return "No event data."

    ts = (m.get("timestamp") or "").strip()
    event_dt = fast_parse_ts(ts)

    action = safe_int(m.get("action_type", ""), default=-1)
    action_str = "PURCHASE" if action == 0 else ("REFUND" if action == 1 else "OTHER")

    ttb = safe_float(m.get("lead_time_buy", ""), 0.0)
    ttr = safe_float(m.get("lead_time_ref", ""), 0.0)

    dep_dt = None
    if event_dt is not None:
        if action == 0 and ttb > 0:
            dep_dt = event_dt + datetime.timedelta(minutes=ttb)
        elif action == 1 and ttr > 0:
            dep_dt = event_dt + datetime.timedelta(minutes=ttr)
        else:
            dep_dt = None

    dep_dow = safe_int(m.get("dep_dow", ""), default=-1)
    dep_hour = safe_int(m.get("dep_hour", ""), default=-1)
    if dep_dt is not None and dep_hour >= 0:
        dep_dt = dep_dt.replace(hour=int(dep_hour), minute=0, second=0, microsecond=0)

    dep_date = safe_date_str(dep_dt)
    dow_str = DOW_EN.get(dep_dow, str(dep_dow)) if dep_dow >= 0 else ""
    dep_time_str = f"{dow_str} {dep_hour}:00 departure" if (dow_str and dep_hour >= 0) else (
        f"{dep_hour}:00 departure" if dep_hour >= 0 else "departure time unknown"
    )

    dep = fmt_station_name(m.get("dep_station_id", ""), station_map)
    arr = fmt_station_name(m.get("arr_station_id", ""), station_map)

    seat = safe_int(m.get("seat_cnt", ""), default=0)
    buy_amt = safe_float(m.get("buy_amt", ""), 0.0)
    refund_amt = safe_float(m.get("refund_amt", ""), 0.0)
    fee = safe_float(m.get("cancel_fee", ""), 0.0)
    hold = safe_float(m.get("hold_time", ""), 0.0)

    route_buy_cnt = safe_int(m.get("route_buy_cnt", ""), default=0)
    rev_buy_cnt = safe_int(m.get("rev_buy_cnt", ""), default=0)
    rev_ratio = safe_float(m.get("rev_ratio", ""), default=0.0)
    unique_route_cnt = safe_int(m.get("unique_route_cnt", ""), default=0)
    rev_return_gap = safe_float(m.get("rev_return_gap", ""), default=0.0)

    overlap = safe_int(m.get("overlap_cnt", ""), 0)
    same_route = safe_int(m.get("same_route_cnt", ""), 0)
    repeat_gap = safe_float(m.get("repeat_interval", ""), 0.0)
    recent_ref_cnt = safe_int(m.get("recent_ref_cnt", ""), 0)
    recent_ref_amt = safe_float(m.get("recent_ref_amt", ""), 0.0)
    recent_ref_rate = safe_float(m.get("recent_ref_rate", ""), 0.0)
    adj_flag = safe_int(m.get("adj_seat_refund_flag", ""), 0)

    if action == 0:
        money_part = f"paid {fmt_money_en(buy_amt)}"
        time_part = f"purchased {fmt_min_en(ttb)} before departure"
    elif action == 1:
        money_part = f"refunded {fmt_money_en(refund_amt)} (fee {fmt_money_en(fee)})"
        time_part = f"refunded {fmt_min_en(ttr)} before departure; held {fmt_min_en(hold)}"
    else:
        money_part = f"amounts: buy {fmt_money_en(buy_amt)}, refund {fmt_money_en(refund_amt)}"
        time_part = f"timing: buy {fmt_min_en(ttb)} before, refund {fmt_min_en(ttr)} before"

    intent_bits = []
    if route_buy_cnt > 0:
        intent_bits.append(f"same-route purchases in window: {route_buy_cnt}")
    if rev_buy_cnt > 0:
        intent_bits.append(f"reverse-direction purchases: {rev_buy_cnt} (ratio {rev_ratio:.2f})")
    if unique_route_cnt > 0:
        intent_bits.append(f"unique routes in window: {unique_route_cnt}")
    if rev_return_gap > 0:
        intent_bits.append(f"min gap arrival→return trip: {fmt_min_en(rev_return_gap)}")
    intent_part = "; ".join(intent_bits) if intent_bits else "limited usage-pattern context"

    fraud_bits = []
    if overlap > 0:
        fraud_bits.append(f"overlapping tickets: {overlap}")
    if same_route > 0:
        fraud_bits.append(f"same-route tickets held: {same_route}")
    if repeat_gap > 0:
        fraud_bits.append(f"median same-route re-purchase gap: {fmt_min_en(repeat_gap)}")
    if adj_flag == 1:
        fraud_bits.append("adjacent-seat refund observed")
    if recent_ref_cnt > 0:
        fraud_bits.append(f"recent same-route refunds: {recent_ref_cnt}")
    if recent_ref_rate > 0:
        fraud_bits.append(f"recent same-route refund rate: {recent_ref_rate:.2f}")
    if recent_ref_amt > 0:
        fraud_bits.append(f"recent same-route refund amount: {fmt_money_en(recent_ref_amt)}")
    fraud_part = "; ".join(fraud_bits) if fraud_bits else "no notable anomaly signal"

    return (
        f"[{ts}] (dep_date={dep_date}) {dep}→{arr} ({dep_time_str}) | {action_str} | seats={seat} | "
        f"{money_part} | {time_part} | "
        f"usage-context: {intent_part} | anomaly-signals: {fraud_part}"
    )

# =========================================================
# 5) Window-level summary
# =========================================================
def compute_window_summary(meta_seq: List[Optional[Dict[str, Any]]]) -> Dict[str, float]:
    ms = [m for m in meta_seq if m is not None]
    total_purchase_amount = sum(safe_float(m.get("buy_amt", ""), 0.0) for m in ms)
    total_refund_amount = sum(safe_float(m.get("refund_amt", ""), 0.0) for m in ms)

    purchase_count = sum(1 for m in ms if safe_int(m.get("action_type", ""), -1) == 0)
    refund_count = sum(1 for m in ms if safe_int(m.get("action_type", ""), -1) == 1)

    refund_rate_cnt = 0.0
    if purchase_count > 0:
        refund_rate_cnt = 100.0 * (refund_count / purchase_count)

    return {
        "total_purchase_amount": float(total_purchase_amount),
        "total_refund_amount": float(total_refund_amount),
        "purchase_count": int(purchase_count),
        "refund_count": int(refund_count),
        "refund_rate_count_pct": float(refund_rate_cnt),
        "n_events": int(len(ms)),
    }

# =========================================================
# 6) English prompt (원문 그대로 유지)
# =========================================================
USER_TYPE_LIST_EN = [
    "Commuter (weekday-centered, repeated specific route during rush hours)",
    "Weekend home-and-return (Fri outbound, Sun/Mon inbound reverse direction)",
    "Weekend/leisure traveler (Fri–Sun, diverse routes/times, irregular)",
    "Weekend hospital visitor (Sat/Sun or specific day, early departures, same-day round trip/short stay)",
    "One-time user (short period, single route, little repetition)",
    "Corporate booking agent (routes centered around company location; administrative purchasing)",
]

def build_llm_prompt_en(summary: Dict[str, float], key_event_lines: List[str]) -> str:
    total_purchase_amt = summary["total_purchase_amount"]
    total_refund_amt = summary["total_refund_amount"]
    purchase_cnt = summary["purchase_count"]
    refund_cnt = summary["refund_count"]
    refund_rate_cnt = summary["refund_rate_count_pct"]

    events_block = "\n".join([f"- {e}" for e in key_event_lines]) if key_event_lines else "- (No key events)"

    system_prompt = (
        "You are an analyst supporting a railway operator's enforcement decisions for suspicious ticketing behavior.\n"
        "Your job is to produce an operationally useful, evidence-based explanation.\n\n"
        "IMPORTANT:\n"
        "- You MUST do multi-step reasoning internally, but you MUST NOT reveal your chain-of-thought.\n"
        "- Do NOT mention model internals (scores, attention weights, feature names, embeddings).\n"
        "- Do NOT use definitive claims like 'confirmed fraud'. Use cautious language such as 'suggests', 'is consistent with', 'likely'.\n"
        "- Use concrete numbers and refer to specific events.\n"
    )

    user_prompt = (
        "Below is a summary of one user's 28-day ticketing window (up to the final purchase time).\n\n"
        "[1] Cumulative summary at final purchase time\n"
        f"- Purchase count: {purchase_cnt}\n"
        f"- Refund count: {refund_cnt}\n"
        f"- Refund rate : {fmt_pct(refund_rate_cnt)}\n"
        f"- Total purchase amount: {fmt_money_en(total_purchase_amt)}\n"
        f"- Total refund amount: {fmt_money_en(total_refund_amt)}\n\n"
        "[2] Key events (selected by model attention; chronological text may be mixed)\n"
        f"{events_block}\n\n"
        "Analyze internally using the following steps (DO NOT output these steps):\n"
        "Step 1) Interpret the cumulative metrics (refund rate/count/amount).\n"
        "Step 2) Interpret the key events with numbers.\n"
        "Step 3) Select ONE closest user type from the list.\n"
        "Step 4) Compare the user's behavior to the typical pattern of that type and identify concrete deviations.\n"
        "Step 5) Propose ONE most plausible intent consistent with the deviations.\n\n"
        "User types to choose from:\n"
        + "\n".join([f"- {x}" for x in USER_TYPE_LIST_EN]) +
        "\n\n"
        "Now output ONLY the following two sections (and nothing else):\n"
        "A) [CUMULATIVE]\n"
        "B) [INTERPRETATION]\n\n"
        "Formatting requirements:\n"
        "- [CUMULATIVE] must be 1–2 lines with the refund rate and refund count (optionally mention amounts).\n"
        "- [INTERPRETATION] must be 4–7 lines total:\n"
        "  * Line 1: chosen user type (one of the given types)\n"
        "  * Next lines: interpret each key event (one line per event) with concrete numbers and why it is abnormal/deviating\n"
        "  * Final line: ONE most plausible intent (cautious language) + short bullet-like justification in the same line\n"
        "\nEvent line MUST include:\n"
        "- transaction timestamp (YYYY-MM-DD HH:MM:SS)\n"
        "- departure date (YYYY-MM-DD)\n"
        "- weekday/time if available (e.g., Thu 15:00)\n"
        "Use a compact prefix such as: [<timestamp> <PURCHASE/REFUND>] For <dep_date> (<Dow HH:MM>) <Origin→Destination> ...\n"
    )
    return "System:\n" + system_prompt + "\nUser:\n" + user_prompt

# =========================================================
# 7) Build prompts and save jsonl (NO LLM)
# =========================================================
@torch.no_grad()
def build_and_save_jsonl(
    model,
    loader,
    device,
    station_map,
    out_jsonl_path: str,
    threshold: float = 0.5,
    max_records: int = 2000,
):
    # pdb.set_trace()
    model.eval()
    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)

    saved = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Build prompts (no-LLM)"):
            cat = batch["cat"].to(device, non_blocking=True)
            num = batch["num"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            meta = batch["meta"]
            seq_paths = batch.get("seq_path", None)  # ✅ 추가

            use_amp = bool(getattr(model, "cfg", None) and model.cfg.use_amp and device.type == "cuda")
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits, attn_last = forward_with_optional_attn(
                    model, cat, num, delta, pad_mask, return_attn=True
                )

            probs = torch.softmax(logits, dim=1)[:, 1]
            B = cat.size(0)

            pdb.set_trace()
            topk_idxs_per_sample = select_topk_events_by_cls_attention(attn_last, pad_mask, k=3)
            pdb.set_trace()
            # ✅ 메모리 압박 줄이기: attn 참조 즉시 끊기(프롬프트 길이는 유지)
            try:
                model.encoder.layers[-1].last_attn_weights = None
            except:
                pass
            attn_last = None

            for i in range(B):
                score = float(probs[i].item())
                risk_level = "High" if score >= threshold else "Low"
                true = int(y[i].item())
                this_path = seq_paths[i] if seq_paths is not None else None  # ✅ 추가

                summary = compute_window_summary(meta[i])

                topk_idx = topk_idxs_per_sample[i]
                key_events = [
                    summarize_event_row_en(meta[i][t], station_map)
                    for t in topk_idx
                    if 0 <= t < len(meta[i]) and meta[i][t] is not None
                ]

                prompt = build_llm_prompt_en(summary, key_events)

                rec = {
                    "test_path" : this_path,
                    "y_true": true,
                    "risk_score": score,
                    "risk_level": risk_level,
                    "window_summary": summary,
                    "key_events": key_events,
                    "prompt": prompt,
                    "attn_topk_indices_meta": topk_idx,
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                saved += 1
                if saved >= max_records:
                    break

            if saved >= max_records:
                break

    print(f"[DONE] saved prompts jsonl to: {out_jsonl_path}")

def main():
    # Transformer는 이 스크립트에서만 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CFG()

    save_dir = "/home/srt/ml_results/transformer/2026-01-25_16-04"
    best_pth = os.path.join(save_dir, "transformer_best.pth")
    assert os.path.exists(best_pth), f"best model not found: {best_pth}"

    station_map = load_station_map("/home/srt/Dataset/feature/station_code_map.csv")

    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()

    POS_LABEL = 1
    filtered = [(p, y) for p, y in zip(test_paths, test_labels) if int(y) == POS_LABEL]
    # pdb.set_trace()
    if len(filtered) == 0:
        raise RuntimeError("[ERROR] No positive samples (label=1) found in test set. Check labels.")
    test_paths, test_labels = zip(*filtered)
    test_paths, test_labels = list(test_paths), list(test_labels)
    print(f"[INFO] Filtered test set to positives only: {len(test_paths)} samples")

    vocabs = load_vocabs_if_exists(save_dir)
    if vocabs is None:
        vocabs = build_vocabs_on_the_fly(train_paths, limit=20000)

    test_ds = TransformerDatasetWithMeta(test_paths, test_labels, vocabs, cfg)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn_with_meta,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = SRTTransformerClassifier(vocabs, len(NUMERIC_COLS), cfg).to(device)
    sd = torch.load(best_pth, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    patch_encoder_layers_to_save_attn(model)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_jsonl = os.path.join(save_dir, f"llm_prompts_test_attn_en_{ts}.jsonl")

    build_and_save_jsonl(
        model=model,
        loader=test_loader,
        device=device,
        station_map=station_map,
        out_jsonl_path=out_jsonl,
        threshold=0.5,
        max_records=2000,
    )

if __name__ == "__main__":
    main()
