# transformer_result_attention_prompting.py
# ---------------------------------------------------------
# 목적:
# 1) Transformer(best)로 test set을 돌려 risk score 산출
# 2) 사용자별 window(=sequence) 전체를 기준으로 누적 반환율/반환금액/반환건수 계산
#    - refund rate = 환불횟수 / 발매횟수
# 3) "설명에 유의미한 이벤트 K개"를 attention 기반 Top-K로 선택 (CLS→event)
# 4) 영어 프롬프트 생성
# 5) (옵션) LLaMA-3.1-70B-Instruct 로 실제 출력까지 생성하여 jsonl에 함께 저장
#
# 주의:
# - 최종 출력에는 chain-of-thought(중간 추론)을 노출하지 않도록 프롬프트로 강제
# - 모델 내부 정보(attention, feature name, score 등)는 LLM에 노출하지 않도록 설계
# ---------------------------------------------------------

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

# Force use of the known valid token to avoid environment variable overrides
os.environ["HF_TOKEN"] = VALID_TOKEN
hf_token = VALID_TOKEN

# User requested to use GPU 0 (Quantized) because others are busy
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print(f"[INFO] Enforcing HF_TOKEN: {hf_token[:5]}... (Gated Model Access OK)")
print(f"[INFO] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
# 2) Dataset: model input + meta(원본 row) 반환
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
            return {"cat": cat, "num": num, "delta": delta, "meta": meta, "y": torch.tensor(label, dtype=torch.long)}

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
        return {"cat": cat, "num": num, "delta": delta, "meta": meta, "y": torch.tensor(label, dtype=torch.long)}


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

    for i, x in enumerate(batch):
        T = x["cat"].shape[0]
        cat_pad[i, :T] = x["cat"]
        num_pad[i, :T] = x["num"]
        delta_pad[i, :T] = x["delta"]
        pad_mask[i, :T] = False
        ys[i] = x["y"]
        for t in range(T):
            meta_pad[i][t] = x["meta"][t]

    return {"cat": cat_pad, "num": num_pad, "delta": delta_pad, "pad_mask": pad_mask, "y": ys, "meta": meta_pad}


# =========================================================
# 3) Attention extraction: runtime patch (NO forward signature changes)
# =========================================================
def patch_encoder_layers_to_save_attn(model) -> bool:
    """
    모델의 encoder.layers[*]._sa_block을 런타임으로 패치하여
    MultiheadAttention의 attention weights(B,H,S,S)를 layer.last_attn_weights에 저장.
    """
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        print("[WARN] model.encoder.layers not found. Skip patch.")
        return False

    patched_any = False
    for li, layer in enumerate(model.encoder.layers):
        if not hasattr(layer, "self_attn") or not hasattr(layer, "dropout1"):
            continue

        # 이미 패치됨
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
                average_attn_weights=False,  # (B,H,S,S)
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
    """
    model.forward()가 return_attn 인자를 지원하지 않아도
    런타임 patch로 저장된 last_attn_weights를 읽어서 반환.
    """
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
        pdb.set_trace()
        attn_last = None

    return logits, attn_last


# =========================================================
# ✅ FIXED: CLS 포함 모델에 맞는 Top-K 선택
# =========================================================
def select_topk_events_by_cls_attention(
    attn_last: Optional[torch.Tensor],
    pad_mask: torch.Tensor,
    k: int = 3
) -> List[List[int]]:
    """
    attn_last: (B,H,S,S)  where S = T+1 (CLS + events) in your model
    pad_mask : (B,T)      True=PAD for events only (no CLS)

    Return:
      - per sample: list of top-k indices in "meta/event index space" (0..T-1)
        (즉 summarize_event_row_en(meta[i][idx])에 바로 쓸 수 있는 인덱스)
    """

    B, T = pad_mask.shape
    topk_idxs: List[List[int]] = []

    # fallback if no attention
    if attn_last is None or not isinstance(attn_last, torch.Tensor):
        for b in range(B):
            valid_evt = torch.where(~pad_mask[b])[0].tolist()
            topk_idxs.append(valid_evt[-k:] if len(valid_evt) >= k else valid_evt)
        return topk_idxs

    # attn_last: (B,H,S,S)
    S = attn_last.size(-1)

    # ✅ your model prepends CLS, so typically S == T+1
    if S == T + 1:
        # pad_mask_aligned: (B,S) with CLS=False at front
        cls_col = torch.zeros((B, 1), dtype=torch.bool, device=pad_mask.device)
        pad_mask_aligned = torch.cat([cls_col, pad_mask], dim=1)  # (B,T+1)
        has_cls = True
    elif S == T:
        pad_mask_aligned = pad_mask
        has_cls = False
    else:
        # robust: trim to min length
        minL = min(S, T)
        pad_mask_aligned = pad_mask[:, :minL]
        attn_last = attn_last[:, :, :minL, :minL]
        S = minL
        has_cls = False

    # CLS query -> key tokens
    cls_attn = attn_last[:, :, 0, :]     # (B,H,S)
    cls_attn = cls_attn.mean(dim=1)      # (B,S) head-avg

    scores = cls_attn.clone()
    scores[pad_mask_aligned] = -1e9

    # CLS itself should not be selected
    if has_cls and S > 0:
        scores[:, 0] = -1e9

    for b in range(B):
        # number of valid tokens in aligned space
        valid_count = int((~pad_mask_aligned[b]).sum().item())
        effective_valid = valid_count - (1 if has_cls else 0)
        kk = min(k, max(0, effective_valid))

        if kk <= 0:
            valid_evt = torch.where(~pad_mask[b])[0].tolist()
            topk_idxs.append(valid_evt[-k:] if len(valid_evt) >= k else valid_evt)
            continue

        idx_token = torch.topk(scores[b], k=kk, largest=True).indices.tolist()

        if has_cls:
            # token index 1..T maps to meta index 0..T-1
            mapped = [t - 1 for t in idx_token if t > 0]
        else:
            mapped = idx_token

        mapped = [t for t in mapped if 0 <= t < T]
        topk_idxs.append(mapped)

    return topk_idxs


# =========================================================
# 4) Summarize a single event (LLM-friendly, English)
# =========================================================
def summarize_event_row_en(m: Dict[str, Any], station_map: dict) -> str:
    if m is None:
        return "No event data."

    # 1) event time
    ts = (m.get("timestamp") or "").strip()
    event_dt = fast_parse_ts(ts)

    # 2) action
    action = safe_int(m.get("action_type", ""), default=-1)
    action_str = "PURCHASE" if action == 0 else ("REFUND" if action == 1 else "OTHER")

    # 3) lead times (minutes before departure at the time of the event)
    ttb = safe_float(m.get("lead_time_buy", ""), 0.0)
    ttr = safe_float(m.get("lead_time_ref", ""), 0.0)

    # 4) departure datetime reconstruction (event_time + lead_time_{buy/ref})
    dep_dt = None
    if event_dt is not None:
        if action == 0 and ttb > 0:
            dep_dt = event_dt + datetime.timedelta(minutes=ttb)
        elif action == 1 and ttr > 0:
            dep_dt = event_dt + datetime.timedelta(minutes=ttr)
        else:
            # if lead time is 0 or missing, we can't infer reliably
            dep_dt = None

    # 5) optional: align hour to dep_hour feature if present (safer to keep dep_dt minute/second as computed)
    dep_dow = safe_int(m.get("dep_dow", ""), default=-1)
    dep_hour = safe_int(m.get("dep_hour", ""), default=-1)
    if dep_dt is not None and dep_hour >= 0:
        # keep computed date, but snap to dep_hour:00 to match your feature definition
        dep_dt = dep_dt.replace(hour=int(dep_hour), minute=0, second=0, microsecond=0)

    dep_date = safe_date_str(dep_dt)
    dow_str = DOW_EN.get(dep_dow, str(dep_dow)) if dep_dow >= 0 else ""
    dep_time_str = f"{dow_str} {dep_hour}:00 departure" if (dow_str and dep_hour >= 0) else (
        f"{dep_hour}:00 departure" if dep_hour >= 0 else "departure time unknown"
    )

    # stations
    dep = fmt_station_name(m.get("dep_station_id", ""), station_map)
    arr = fmt_station_name(m.get("arr_station_id", ""), station_map)

    # other fields
    seat = safe_int(m.get("seat_cnt", ""), default=0)
    buy_amt = safe_float(m.get("buy_amt", ""), 0.0)
    refund_amt = safe_float(m.get("refund_amt", ""), 0.0)
    fee = safe_float(m.get("cancel_fee", ""), 0.0)

    hold = safe_float(m.get("hold_time", ""), 0.0)

    # Intentional Context (short)
    route_buy_cnt = safe_int(m.get("route_buy_cnt", ""), default=0)
    rev_buy_cnt = safe_int(m.get("rev_buy_cnt", ""), default=0)
    rev_ratio = safe_float(m.get("rev_ratio", ""), default=0.0)
    unique_route_cnt = safe_int(m.get("unique_route_cnt", ""), default=0)
    rev_return_gap = safe_float(m.get("rev_return_gap", ""), default=0.0)

    # Fraud
    overlap = safe_int(m.get("overlap_cnt", ""), 0)
    same_route = safe_int(m.get("same_route_cnt", ""), 0)
    repeat_gap = safe_float(m.get("repeat_interval", ""), 0.0)
    recent_ref_cnt = safe_int(m.get("recent_ref_cnt", ""), 0)
    recent_ref_amt = safe_float(m.get("recent_ref_amt", ""), 0.0)
    recent_ref_rate = safe_float(m.get("recent_ref_rate", ""), 0.0)
    adj_flag = safe_int(m.get("adj_seat_refund_flag", ""), 0)

    # money/time description (event-relative lead times)
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

    # [CHANGED] dep_date is now reconstructed, not event_date
    return (
        f"[{ts}] (dep_date={dep_date}) {dep}→{arr} ({dep_time_str}) | {action_str} | seats={seat} | "
        f"{money_part} | {time_part} | "
        f"usage-context: {intent_part} | anomaly-signals: {fraud_part}"
    )


# =========================================================
# 5) Window-level summary (refund_rate = refund_count / purchase_count)
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
# 6) English prompt
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
        "Write an operationally useful, evidence-based explanation in English.\n\n"
        "Rules:\n"
        "- Think step-by-step internally; DO NOT reveal chain-of-thought.\n"
        "- Do NOT mention model internals (attention/scores/features/embeddings).\n"
        "- Do NOT claim confirmed fraud; use cautious language (suggests/likely/consistent with).\n"
        "- Ground every claim in the given numbers/events.\n"
        "- Use concrete numbers and specific timestamps/dates.\n"
    )

    user_prompt = (
        "28-day ticketing window summary (up to final purchase).\n\n"
        "INPUT\n"
        "[1] Numerical summary\n"
        f"- Purchase: {fmt_money_en(total_purchase_amt)} across {purchase_cnt}\n"
        f"- Refund: {fmt_money_en(total_refund_amt)} across {refund_cnt}\n"
        "[2] Key events (may be out of order; treat as evidence)\n"
        f"{events_block}\n\n"
        "Choose exactly ONE user type and ONE bulk-refund pattern label.\n\n"
        "User types:\n"
        "- Commuter (weekday commute, fixed route/time)\n"
        "- Weekend home-and-return (Fri outbound, Sun/Mon return)\n"
        "- Weekend/leisure traveler (Fri–Sun, diverse routes/times)\n"
        "- Weekend hospital visitor (Sat/Sun or fixed day, hospital-area routes)\n"
        "- One-time user (short-term, little repetition)\n"
        "- Corporate booking agent (books/cancels for others)\n\n"
        "Bulk-refund patterns:\n"
        "- Adjacent-seat holding + last-minute refunds\n"
        "- Card-spend targeting via bulk purchase/refund\n"
        "- Macro-assisted booking for resale\n"
        "- Bulk seat-holding + selective refunds\n\n"
        "TASK (do NOT output steps):\n"
        "1) Pick one user type + one pattern label.\n"
        "2) Justify BRB risk using the strongest event facts.\n"
        "3) Explain deviations from the chosen type; infer ONE plausible intent (cautious).\n\n"
        "OUTPUT (no extra text; keep headings exactly):\n"
        "=== NUMERICAL SUMMARY ===\n"
        "- <purchase amount> across <purchase count>\n"
        "- <refund amount> across <refund count> \n\n"
        "=== BRB RISK JUSTIFICATION (<bulk-refund pattern label>) ===\n"
        "4–6 sentences. Use concrete counts/amounts/timing; mention last-minute/holding/adjacent seats if present.\n\n"
        "=== BOOKING INTENT INTERPRETATION (<user type>) ===\n"
        "3–5 sentences. Start: Selected user type: <...>. End: A plausible intent is that ... (cautious).\n\n"
        "=== DETAILED EVENT SUMMARY ===\n"
        "Chronological (earliest→latest).\n"
        "For EACH event:\n"
        "- Describe the event factually (time, route, action, amount).\n"
        "- THEN select ONLY the most relevant 2–4 behavioral signals that help interpret\n"
        "  (a) bulk-refund risk or (b) booking intent.\n"
        "- Signals may include timing (e.g., how long held, how close to departure),\n"
        "  repetition or overlap patterns, seat-holding behavior, or recent refund concentration.\n"
        "- Do NOT list all available features; include only those that materially support interpretation.\n\n"
        "Each line format (flexible, but concise):\n"
        "[YYYY-MM-DD HH:MM:SS ACTION] For YYYY-MM-DD (Dow HH:MM) Origin→Destination | seats=<n> | amount=<...> | "
        "key signals: <signal A>; <signal B>; <signal C>\n"
    )


    return "System:\n" + system_prompt + "\nUser:\n" + user_prompt


# =========================================================
# 7) Optional: run LLM (vLLM preferred; fallback transformers)
# =========================================================
def generate_with_llm(
    prompts: List[str],
    model_name: str ="meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_new_tokens: int = 300,
    temperature: float = 0.2,
) -> List[str]:
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        llm = LLM(
            model=model_name,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enforce_eager=True,
            gpu_memory_utilization=0.95,
            max_model_len=65536,
            tensor_parallel_size=1
        )
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=0.95,
        )
        outs = llm.generate(prompts, params)
        return [o.outputs[0].text if (o.outputs and len(o.outputs) > 0) else "" for o in outs]
    except Exception as e_vllm:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                token=hf_token
            )
            model.eval()

            texts = []
            for p in prompts:
                inputs = tok(p, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=0.95,
                    )
                gen = tok.decode(out[0], skip_special_tokens=True)
                anchor = "A) [CUMULATIVE]"
                if anchor in gen:
                    gen = gen.split(anchor, 1)[-1]
                    gen = anchor + gen
                texts.append(gen)
            return texts
        except Exception as e_tf:
            print("[WARN] LLM generation unavailable. Install vllm or transformers.")
            print("  vLLM error:", str(e_vllm)[:300])
            print("  transformers error:", str(e_tf)[:300])
            return ["" for _ in prompts]


# =========================================================
# 8) Main: load best -> build prompts -> (optional) run LLM -> save jsonl
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
    run_llm: bool = False,
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    llm_batch_size: int = 8,
):
    model.eval()
    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)

    pending_prompts: List[str] = []
    pending_records: List[Dict[str, Any]] = []

    def flush_llm_and_write(fh):
        nonlocal pending_prompts, pending_records
        if not pending_records:
            return
        outputs = generate_with_llm(
            pending_prompts,
            model_name=llm_model_name,
            max_new_tokens=260,
            temperature=0.2,
        )
        for rec, out_text in zip(pending_records, outputs):
            rec["llm_output"] = out_text
            print(out_text)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pending_prompts = []
        pending_records = []

    saved = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Build prompts (best model)"):
            cat = batch["cat"].to(device, non_blocking=True)
            num = batch["num"].to(device, non_blocking=True)
            delta = batch["delta"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)  # (B,T) events only
            y = batch["y"].to(device, non_blocking=True)
            meta = batch["meta"]  # List[List[dict|None]]

            use_amp = bool(getattr(model, "cfg", None) and model.cfg.use_amp and device.type == "cuda")

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits, attn_last = forward_with_optional_attn(
                    model, cat, num, delta, pad_mask, return_attn=True
                )

            probs = torch.softmax(logits, dim=1)[:, 1]
            B = cat.size(0)

            # ✅ Top-K event indices in "meta index space" (0..T-1)
            topk_idxs_per_sample = select_topk_events_by_cls_attention(attn_last, pad_mask, k=5)

            for i in range(B):
                score = float(probs[i].item())
                risk_level = "High" if score >= threshold else "Low"
                true = int(y[i].item())

                summary = compute_window_summary(meta[i])

                topk_idx = topk_idxs_per_sample[i]
                key_events = [
                    summarize_event_row_en(meta[i][t], station_map)
                    for t in topk_idx
                    if 0 <= t < len(meta[i]) and meta[i][t] is not None
                ]

                prompt = build_llm_prompt_en(summary, key_events)

                rec = {
                    "y_true": true,
                    "risk_score": score,
                    "risk_level": risk_level,
                    "window_summary": summary,
                    "key_events": key_events,
                    "prompt": prompt,
                    "attn_topk_indices_meta": topk_idx,  # meta index 기준(재현용)
                }

                if run_llm:
                    pending_prompts.append(prompt)
                    pending_records.append(rec)
                    if len(pending_records) >= llm_batch_size:
                        flush_llm_and_write(f)
                else:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                saved += 1
                if saved >= max_records:
                    break

            if saved >= max_records:
                break

        if run_llm:
            flush_llm_and_write(f)

    print(f"[DONE] saved jsonl to: {out_jsonl_path}")


def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    cfg = CFG()

    # pdb.set_trace()
    save_dir = "/home/srt/ml_results/transformer/2026-01-25_16-04"
    best_pth = os.path.join(save_dir, "transformer_best.pth")
    assert os.path.exists(best_pth), f"best model not found: {best_pth}"

    # RUN_LLM = True
    RUN_LLM = False
    LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    station_map = load_station_map("/home/srt/Dataset/feature/station_code_map.csv")

    train_paths, train_labels = load_train_data()
    test_paths, test_labels = load_test_data()


    # 추가 : 부정 반환자(Label=1)만 필터링
    POS_LABEL = 1  # 부정 반환자 라벨이 1이라고 가정
    filtered = [(p, y) for p, y in zip(test_paths, test_labels) if int(y) == POS_LABEL]
    if len(filtered) == 0:
        raise RuntimeError("[ERROR] No positive samples (label=1) found in test set. Check labels.")

    test_paths, test_labels = zip(*filtered)
    test_paths, test_labels = list(test_paths), list(test_labels)

    print(f"[INFO] Filtered test set to positives only: {len(test_paths)} samples")


    vocabs = load_vocabs_if_exists(save_dir)
    if vocabs is None:
        vocabs = build_vocabs_on_the_fly(train_paths, limit=20000)

    test_ds = TransformerDatasetWithMeta(test_paths, test_labels, vocabs, cfg)
    # 부정 반환자에 대해서만 뽑으면 되는 거 아닌가?
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_with_meta,
        pin_memory=getattr(cfg, "pin_memory", True),
        persistent_workers=False
    )

    model = SRTTransformerClassifier(vocabs, len(NUMERIC_COLS), cfg).to(device)
    sd = torch.load(best_pth, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ✅ attention 저장 패치
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
        run_llm=RUN_LLM,
        llm_model_name=LLM_MODEL_NAME,
        llm_batch_size=8,
    )


if __name__ == "__main__":
    main()


