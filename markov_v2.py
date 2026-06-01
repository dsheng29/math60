#!/usr/bin/env python3
"""
WildChat Markov Chain Analysis — v2
-------------------------------------
5 categories (Informational, Creative, Revision, Explanation, Frustration).
Paired state: each exchange (Human turn, AI turn) = one state → 25 possible states (5×5).
Minimum 3 complete exchanges per conversation required.
English-only, shuffled sampling. Claude Sonnet classifier with context for AI turns.

Usage:
  python markov_v2.py                   # 1000 conversations
  python markov_v2.py --samples 200     # smaller test run
  python markov_v2.py --no-cache        # force re-classification
  python markov_v2.py --show-examples 5 # print N example conversation classifications
"""

import argparse
import asyncio
import os
import pickle
import re
import textwrap
import warnings
from collections import Counter
from pathlib import Path
import anthropic
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

# ── Categories ────────────────────────────────────────────────────────────────
# Same label set for both human and AI turns.
# Human framing: what is the human asking for?
# AI framing:    what kind of response is the AI giving?

SHORT_LABELS = [
    "Informational",
    "Creative",
    "Revision",
    "Explanation",
    "Frustration",
]
ABBREV = ["IF", "Cr", "Rv", "Ex", "Fr"]

N_CATS   = len(SHORT_LABELS)
N_STATES = N_CATS * N_CATS          # 25 paired states (5×5)
CAT_IDX  = {s: i for i, s in enumerate(SHORT_LABELS)}

MIN_EXCHANGES = 3                   # minimum Human-AI pairs per conversation
DEFAULT_SAMPLES = 1000
_HERE      = Path(__file__).parent  # always markov_v2/, regardless of cwd
CACHE_FILE = _HERE / "classified_v2_sonnet_5cat_v8.pkl"
OUTPUT_DIR = _HERE / "output"


# ── Paired state helpers ──────────────────────────────────────────────────────

def pair_idx(h: int, a: int) -> int:
    return h * N_CATS + a

def idx_to_pair(idx: int) -> tuple:
    return idx // N_CATS, idx % N_CATS

# Short labels for all 25 pair states, e.g. "IF+Cr"
PAIR_LABELS = [f"{ABBREV[h]}+{ABBREV[a]}"
               for h in range(N_CATS) for a in range(N_CATS)]

PAIR_LABELS_LONG = [f"{SHORT_LABELS[h]} / {SHORT_LABELS[a]}"
                    for h in range(N_CATS) for a in range(N_CATS)]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_wildchat(n: int) -> list:
    print(f"Streaming {n} English conversations (shuffled) from allenai/WildChat-1M …")
    ds = load_dataset("allenai/WildChat-1M", split="train",
                      streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    rows = []
    skipped = 0
    for rec in tqdm(ds, total=n, desc="Loading"):
        if rec.get("language") != "English":
            skipped += 1
            continue
        rows.append(rec)
        if len(rows) >= n:
            break
    print(f"  (skipped {skipped} non-English conversations)")
    return rows


# ── Classification (Claude Sonnet) ───────────────────────────────────────────

SYSTEM_PROMPT = """Classify a conversation turn. First write one sentence of reasoning starting with "Think:", then write "Answer: [category]".

Categories: Informational, Creative, Revision, Explanation, Frustration

RULES — check in this exact order, stop at the first match:
1. Frustration — human reacts to AI's previous output (corrects it, questions it, points out an error, asks it to redo). AI apologises or self-corrects.
2. Creative — human asks AI to generate, produce, write, list, create, or invent anything new. AI produces a generated list, story, or novel content.
3. Revision — human provides existing text to paraphrase/fix/evaluate, OR message is mostly a block of text with no direct question. AI rephrases or transforms provided content.
4. Informational — asks what a specific named thing is, what it does, or what its purpose is. Answerable by Google. For AI: delivering factual information about specific named topics, organisms, products, tests, diseases, etc.
5. Explanation — LAST RESORT only. Pure conceptual "why/how does X work" with nothing to generate and no named thing to look up. For AI: this is RARE — only when genuinely walking through abstract concepts with no factual lookup involved.

WORKED EXAMPLES — memorise these:

"give a word example for each character in the alphabet"
Think: asks AI to produce a list of words — generation task.
Answer: Creative

"now give examples which end with the character"
Think: follow-up asking AI to generate more examples.
Answer: Creative

"now provide words which begin and end with the same character"
Think: asks AI to generate a new set of words.
Answer: Creative

"some of your words does not obey the rule, find them and give the example again"
Think: human is pointing out the AI made errors in its previous output.
Answer: Frustration

"there is still a wrong word, tell me which one is wrong"
Think: human is challenging the AI's previous output.
Answer: Frustration

"is 'Iris' correct?"
Think: human is questioning whether the AI's previous answer was right.
Answer: Frustration

"What if Goku married Lilac From Freedom Planet?"
Think: hypothetical creative scenario — AI must generate a story.
Answer: Creative

"What if Goku married Lilac part 3, Raditz saga"
Think: continuing a creative hypothetical narrative.
Answer: Creative

"What's the US Public Sector Vibration Test?"
Think: asking what a specific named test is — lookupable.
Answer: Informational

"Please introduce certificates like CB/CSA/BSMI/BIS"
Think: asking what specific named certificates are — lookupable.
Answer: Informational

"What is the purpose of the USPS Test?"
Think: asking about a specific named test's purpose — lookupable.
Answer: Informational

[long academic/scientific paragraph pasted with no explicit question]
Think: block of existing text with no question — implicit paraphrase task.
Answer: Revision

"Colloquial: [text]. Classical: [text]."
Think: human is providing text in one style to be converted to another.
Answer: Revision

"Why does inflation happen?"
Think: pure conceptual understanding, no named thing, no generation.
Answer: Explanation

AI response about a specific organism, disease, product, or named scientific thing:
Think: AI is delivering factual information about a specific named topic — lookupable.
Answer: Informational

AI response rephrasing or paraphrasing a paragraph the human provided:
Think: AI is transforming existing text — revision task.
Answer: Revision"""

CONCURRENCY = 20

# ── Rule-based pre-classifier ─────────────────────────────────────────────────
# Applied before sending to Claude. Deterministic — handles clear-cut cases.

_FRUSTRATION_HUMAN = [
    r"does not obey", r"don'?t obey", r"doesn'?t obey",
    r"still wrong", r"still (a ?)?(wrong|error|mistake)",
    r"find (them|the) (wrong|error|mistake|word)",
    r"try again", r"not quite", r"that'?s? not (right|what|correct)",
    r"is [\"'].+[\"'] correct",
    r"are you sure", r"you missed", r"not what i (asked|meant|wanted)",
    r"incorrect(ly)?", r"(some|one) of your (words?|examples?)",
]
_FRUSTRATION_AI = [
    r"^apolog(ize|ies|y)", r"^i apologize", r"^i'?m sorry",
    r"^sorry for", r"^you'?re right,? let me", r"^i made a mistake",
    r"^(my )?apologies",
]
# AI response is clearly generating a list or set of examples
_CREATIVE_AI = [
    r"^[A-Za-z] ?[-–] \w",        # "A - Apple B - Banana" format
    r"^\d+\. ",                   # "1. First item" numbered list
    r"^(here are|here is a list|here's a list)",
    r"^(•|-) \w",                 # bullet list
]
# AI response is clearly paraphrasing/transforming existing content
_REFINEMENT_AI = [
    r"^(here is|here's) (the |a )?(rephrased|paraphrased|rewritten|revised|"
    r"simplified|formal|informal|shorter|longer|translated)",
    r"^(rephrased|paraphrased|rewritten|revised|translated):",
]
_CREATIVE_START = [
    r"^generate\b", r"^write\b", r"^create\b", r"^draft\b",
    r"^compose\b", r"^invent\b", r"^brainstorm\b", r"^imagine\b",
    r"^(now )?(give|provide|list|show)\b",
    r"^come up with\b",
    r"\bwhat if\b",
    r"for each (character|letter|word).{0,40}(give|provide|list|show|example)",
    r"(give|provide|list|show).{0,30}for each (character|letter|word)",
    r"^(give|provide).{0,60}(example|list|word|name|idea|expression|phrase)",
]
_REFINEMENT_EXPLICIT = [
    r"\b(rephrase|paraphrase|rewrite|summari[sz]e|translate|proofread|"
    r"fix this|improve this|edit this)\b",
]
_INFORMATIONAL_START = [
    r"^what'?s?\b", r"^what (is|are|does|do|did)\b",
    r"^who'?s?\b", r"^who (is|are|was|were)\b",
    r"^when (did|was|is|are)\b", r"^where (is|are|was|were)\b",
    r"^how (many|much|long|old|far)\b",
]


def _rule_classify(text: str, role: str):
    """Return a SHORT_LABEL if a clear rule fires, else None → send to Claude."""
    t = text.lower().strip()

    if role == "user":
        for pat in _FRUSTRATION_HUMAN:
            if re.search(pat, t):
                return "Frustration"
        for pat in _CREATIVE_START:
            if re.search(pat, t[:300]):
                return "Creative"
        for pat in _REFINEMENT_EXPLICIT:
            if re.search(pat, t):
                return "Revision"
        # Long block with no interrogative start → implicit paraphrase/edit task
        words = text.split()
        starts_with_question = any(
            t.startswith(q) for q in
            ("what", "why", "how", "who", "when", "where",
             "is ", "are ", "do ", "does ", "can ", "could ", "should ")
        )
        if len(words) > 70 and not starts_with_question:
            return "Revision"
        for pat in _INFORMATIONAL_START:
            if re.match(pat, t):
                return "Informational"

    else:  # assistant
        for pat in _FRUSTRATION_AI:
            if re.match(pat, t):
                return "Frustration"
        for pat in _CREATIVE_AI:
            if re.match(pat, t):
                return "Creative"
        for pat in _REFINEMENT_AI:
            if re.match(pat, t):
                return "Revision"

    return None  # ambiguous — let Claude decide


def _fuzzy_match(raw_label: str) -> str:
    """Map Claude's response to a SHORT_LABEL, with fallback."""
    raw = raw_label.strip().rstrip(".").lower()
    for sl in SHORT_LABELS:
        if sl.lower() == raw:
            return sl
    for sl in SHORT_LABELS:
        if sl.lower() in raw or raw in sl.lower():
            return sl
    # partial word match
    for sl in SHORT_LABELS:
        first_word = sl.split("/")[0].lower()
        if first_word in raw:
            return sl
    return "Explanation"      # genuine fallback after all matching attempts


async def _classify_one(text: str, role: str, context: str,
                        client: anthropic.AsyncAnthropic,
                        sem: asyncio.Semaphore) -> str:
    if not text.strip():
        return "Explanation"
    # Fast deterministic path — no API call needed
    rule_result = _rule_classify(text, role)
    if rule_result is not None:
        return rule_result
    async with sem:
        if role == "assistant" and context:
            user_content = (
                f"Classify the AI response based on what type of response it is.\n\n"
                f"Human asked: {context[:600]}\n\n"
                f"AI responded: {text[:900]}"
            )
        else:
            user_content = f"Classify this message:\n\n{text[:1500]}"
        try:
            msg = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=80,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_content}],
            )
            raw = msg.content[0].text
            # Extract "Answer: X" from chain-of-thought response
            match = re.search(r"Answer:\s*([A-Za-z/]+)", raw)
            return _fuzzy_match(match.group(1) if match else raw)
        except Exception:
            return "Explanation"


async def _classify_all(items: list) -> list:
    """items: list of (text, role, context_text)"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem    = asyncio.Semaphore(CONCURRENCY)
    bar    = tqdm(total=len(items), desc="Classifying")

    async def _tracked(text, role, ctx):
        result = await _classify_one(text, role, ctx, client, sem)
        bar.update(1)
        return result

    labels = await asyncio.gather(*[_tracked(t, r, c) for t, r, c in items])
    bar.close()
    return list(labels)


def classify_conversations(raw: list) -> list:
    # Build (text, role, context_text) triples.
    # AI turns receive the preceding human text as context so Claude can
    # classify the response independently (not just mirroring the human).
    items, locs = [], []
    for ci, rec in enumerate(raw):
        prev_human = ""
        for ti, turn in enumerate(rec["conversation"]):
            content = (turn.get("content") or "").strip()
            role    = turn["role"]
            ctx     = prev_human if role == "assistant" else ""
            items.append((content, role, ctx))
            locs.append((ci, ti))
            if role == "user":
                prev_human = content

    print(f"Classifying {len(items)} turns with Claude Sonnet "
          f"({CONCURRENCY} concurrent requests) …")
    labels = asyncio.run(_classify_all(items))

    classified = [[] for _ in raw]
    for (ci, ti), label in zip(locs, labels):
        turn    = raw[ci]["conversation"][ti]
        content = (turn.get("content") or "").strip()
        classified[ci].append({
            "role":     turn["role"],
            "category": label,
            "length":   len(content.split()),
            "preview":  content[:200],
        })
    return classified


# ── Conversation parsing ──────────────────────────────────────────────────────

def extract_exchanges(conv: list) -> list:
    """
    Return a list of (human_category, ai_category) pairs for each
    consecutive user→assistant turn pair.
    """
    exchanges = []
    i = 0
    while i < len(conv) - 1:
        if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
            h_cat = conv[i]["category"]
            a_cat = conv[i + 1]["category"]
            if h_cat in CAT_IDX and a_cat in CAT_IDX:
                exchanges.append((CAT_IDX[h_cat], CAT_IDX[a_cat]))
            i += 2
        else:
            i += 1
    return exchanges


def filter_min_exchanges(classified: list, min_ex: int = MIN_EXCHANGES) -> list:
    return [conv for conv in classified
            if len(extract_exchanges(conv)) >= min_ex]


# ── Markov chain ──────────────────────────────────────────────────────────────

def build_paired_matrix(classified: list) -> tuple:
    """
    25×25 transition matrix over paired states.
    Row = current exchange pair, column = next exchange pair.
    """
    counts = np.zeros((N_STATES, N_STATES), dtype=float)
    for conv in classified:
        exchanges = extract_exchanges(conv)
        for (h1, a1), (h2, a2) in zip(exchanges[:-1], exchanges[1:]):
            counts[pair_idx(h1, a1), pair_idx(h2, a2)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.where(
        row_sums == 0,
        1.0 / N_STATES,
        counts / np.where(row_sums == 0, 1, row_sums),
    )
    return normalized, counts


def stationary_distribution(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(mat.T)
    idx = np.argmin(np.abs(vals - 1.0))
    stat = np.real(vecs[:, idx])
    stat = np.abs(stat)
    return stat / stat.sum()


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_pair_frequency(classified: list, out_dir: Path):
    """Bar chart of how often each pair state occurs."""
    pair_counts = Counter()
    for conv in classified:
        for h, a in extract_exchanges(conv):
            pair_counts[pair_idx(h, a)] += 1

    labels = [PAIR_LABELS[i] for i in range(N_STATES)]
    values = [pair_counts.get(i, 0) for i in range(N_STATES)]

    # Sort by frequency
    order = np.argsort(values)[::-1]
    labels_sorted = [labels[i] for i in order]
    values_sorted = [values[i] for i in order]

    # Only show non-zero states
    nonzero = [(l, v) for l, v in zip(labels_sorted, values_sorted) if v > 0]
    if not nonzero:
        return
    nl, nv = zip(*nonzero)

    fig, ax = plt.subplots(figsize=(max(10, len(nl) * 0.4), 5))
    ax.bar(nl, nv, color="steelblue", edgecolor="white")
    ax.set_title("Frequency of Paired States  (Human category + AI category)", fontsize=12)
    ax.set_ylabel("Count")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.tight_layout()
    out = out_dir / "pair_state_frequency.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_top_transitions(counts: np.ndarray, top_n: int, out_dir: Path):
    """
    Transition heatmap restricted to the top_n most-observed pair states.
    Makes the 25×25 matrix legible by filtering to the states that actually
    appear enough to estimate reliably.
    """
    state_totals = counts.sum(axis=1) + counts.sum(axis=0)
    top_idx = np.argsort(state_totals)[::-1][:top_n]
    top_idx = sorted(top_idx)          # keep canonical order

    sub_counts = counts[np.ix_(top_idx, top_idx)]
    row_sums = sub_counts.sum(axis=1, keepdims=True)
    sub_norm = np.where(row_sums == 0, 0,
                        sub_counts / np.where(row_sums == 0, 1, row_sums))

    tick_labels = [PAIR_LABELS[i] for i in top_idx]

    annot = np.empty_like(sub_norm, dtype=object)
    for i in range(len(top_idx)):
        for j in range(len(top_idx)):
            annot[i, j] = f"{sub_norm[i,j]:.2f}\n(n={int(sub_counts[i,j])})"

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.7), max(8, top_n * 0.6)))
    sns.heatmap(sub_norm, xticklabels=tick_labels, yticklabels=tick_labels,
                annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                ax=ax, linewidths=0.4, linecolor="white")
    ax.set_title(f"Paired State Transitions  (top {top_n} states by frequency)\n"
                 f"Abbrev: IF=Informational, Cr=Creative, Rv=Revision, Ex=Explanation, Fr=Frustration",
                 fontsize=10)
    ax.set_xlabel("To (next exchange)", fontsize=10)
    ax.set_ylabel("From (current exchange)", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out = out_dir / f"matrix_paired_top{top_n}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_marginal_matrix(counts: np.ndarray, out_dir: Path):
    """
    Collapse the 25×25 matrix to a 5×5 by summing over the AI dimension
    (human marginal) and the human dimension (AI marginal).
    Shows how the *human's* category in exchange n predicts the *human's*
    category in exchange n+1, integrating over whatever AI did.
    """
    # Human marginal: sum rows/cols over AI category
    h_counts = np.zeros((N_CATS, N_CATS), dtype=float)
    for from_idx in range(N_STATES):
        for to_idx in range(N_STATES):
            h_from, _ = idx_to_pair(from_idx)
            h_to,   _ = idx_to_pair(to_idx)
            h_counts[h_from, h_to] += counts[from_idx, to_idx]

    # AI marginal
    a_counts = np.zeros((N_CATS, N_CATS), dtype=float)
    for from_idx in range(N_STATES):
        for to_idx in range(N_STATES):
            _, a_from = idx_to_pair(from_idx)
            _, a_to   = idx_to_pair(to_idx)
            a_counts[a_from, a_to] += counts[from_idx, to_idx]

    for label, cnt, fname in [
        ("Human category  n → n+1  (marginalised over AI)", h_counts, "matrix_human_marginal.png"),
        ("AI category  n → n+1  (marginalised over Human)",  a_counts, "matrix_ai_marginal.png"),
    ]:
        row_sums = cnt.sum(axis=1, keepdims=True)
        norm = np.where(row_sums == 0, 0, cnt / np.where(row_sums == 0, 1, row_sums))

        annot = np.empty_like(norm, dtype=object)
        for i in range(N_CATS):
            for j in range(N_CATS):
                annot[i, j] = f"{norm[i,j]:.2f}\n(n={int(cnt[i,j])})"

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(norm, xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
                    annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                    ax=ax, linewidths=0.5, linecolor="white")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Next exchange category", fontsize=10)
        ax.set_ylabel("Current exchange category", fontsize=10)
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        out = out_dir / fname
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved → {out}")


def plot_stationary_bar(stat: np.ndarray, out_dir: Path):
    labels = PAIR_LABELS
    order  = np.argsort(stat)[::-1]

    # Show only non-negligible states
    threshold = 1 / N_STATES / 2
    visible = [(labels[i], stat[i]) for i in order if stat[i] > threshold]
    if not visible:
        visible = [(labels[i], stat[i]) for i in order[:15]]
    vl, vv = zip(*visible)

    fig, ax = plt.subplots(figsize=(max(10, len(vl) * 0.45), 5))
    ax.bar(vl, vv, color="teal", edgecolor="white")
    ax.set_title("Stationary Distribution of Paired States", fontsize=12)
    ax.set_ylabel("Probability")
    plt.xticks(rotation=55, ha="right", fontsize=8)
    plt.tight_layout()
    out = out_dir / "stationary_paired.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ── Time-homogeneity ─────────────────────────────────────────────────────────

TURN_BUCKETS = {
    "Early (exchanges 1–2)": (0, 2),
    "Mid (exchanges 3–5)":   (2, 5),
    "Late (exchanges 6+)":   (5, None),
}


def build_marginal_by_turn(classified: list) -> dict:
    """
    Build a human-marginal 5×5 transition matrix for each turn bucket.
    A transition from exchange i → exchange i+1 belongs to bucket based on i.
    """
    result = {}
    for name, (lo, hi) in TURN_BUCKETS.items():
        counts = np.zeros((N_CATS, N_CATS), dtype=float)
        for conv in classified:
            exchanges = extract_exchanges(conv)
            for i, (from_ex, to_ex) in enumerate(
                    zip(exchanges[:-1], exchanges[1:])):
                h1, h2 = from_ex[0], to_ex[0]
                in_bucket = (i >= lo) and (hi is None or i < hi)
                if in_bucket:
                    counts[h1, h2] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        norm = np.where(row_sums == 0, 0,
                        counts / np.where(row_sums == 0, 1, row_sums))
        result[name] = (norm, counts)
    return result


def plot_time_homogeneity(classified: list, out_dir: Path):
    buckets = build_marginal_by_turn(classified)
    n_buckets = len(buckets)

    fig, axes = plt.subplots(1, n_buckets, figsize=(9 * n_buckets, 7))

    mats = {}
    for ax, (name, (norm, counts)) in zip(axes, buckets.items()):
        annot = np.empty_like(norm, dtype=object)
        total = int(counts.sum())
        for i in range(N_CATS):
            for j in range(N_CATS):
                annot[i, j] = f"{norm[i,j]:.2f}\n(n={int(counts[i,j])})"
        sns.heatmap(norm, xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
                    annot=annot, fmt="", cmap="Purples", vmin=0, vmax=1,
                    ax=ax, linewidths=0.5, linecolor="white")
        ax.set_title(f"{name}\n({total} transitions)", fontsize=11)
        ax.set_xlabel("Next exchange (human category)", fontsize=9)
        ax.set_ylabel("Current exchange (human category)", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        mats[name] = norm

    # Frobenius distance between earliest and latest bucket as homogeneity diagnostic
    keys = list(mats.keys())
    frob = np.linalg.norm(mats[keys[0]] - mats[keys[-1]], "fro")
    verdict = "suggests non-homogeneity" if frob > 0.4 else "roughly homogeneous"
    fig.suptitle(
        f"Time-Homogeneity Check — Human Category Transition Matrices\n"
        f"Frobenius distance (early vs late): {frob:.3f}  ({verdict})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    out = out_dir / "time_homogeneity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
    print(f"  Frobenius dist (early vs late): {frob:.4f}  ({verdict})")


# ── Diagonal (aligned) 5×5 matrix ────────────────────────────────────────────

def plot_diagonal_matrix(classified: list, out_dir: Path):
    """
    Simplified 5-state Markov chain.

    Only counts exchanges where human and AI landed in the same category
    (the 'aligned' pairs: IF+IF, Cr+Cr, …). Records transitions between
    consecutive aligned states, skipping mismatched exchanges in between.

    Answers: given the conversation was clearly in mode X, what mode
    does it move into next (the next time both sides agree)?
    """
    counts = np.zeros((N_CATS, N_CATS), dtype=float)

    for conv in classified:
        exchanges = extract_exchanges(conv)
        # Keep only exchanges where human and AI share the same category
        aligned = [h for h, a in exchanges if h == a]
        for prev, nxt in zip(aligned[:-1], aligned[1:]):
            counts[prev, nxt] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    norm = np.where(
        row_sums == 0,
        1.0 / N_CATS,
        counts / np.where(row_sums == 0, 1, row_sums),
    )

    annot = np.empty_like(norm, dtype=object)
    for i in range(N_CATS):
        for j in range(N_CATS):
            annot[i, j] = f"{norm[i,j]:.2f}\n(n={int(counts[i,j])})"

    # Also report what fraction of exchanges are aligned
    total_ex = sum(len(extract_exchanges(c)) for c in classified)
    aligned_ex = int(counts.sum())
    pct_aligned = 100 * aligned_ex / max(total_ex, 1)
    print(f"  Aligned (same-category) exchanges: {aligned_ex}/{total_ex} ({pct_aligned:.1f}%)")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(norm, xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
                annot=annot, fmt="", cmap="Greens", vmin=0, vmax=1,
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title(
        "Simplified 5-State Markov Chain\n"
        "(transitions between exchanges where Human & AI share the same category)",
        fontsize=11,
    )
    ax.set_xlabel("Next aligned state", fontsize=10)
    ax.set_ylabel("Current aligned state", fontsize=10)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    out = out_dir / "matrix_diagonal_5state.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ── Example outputs ───────────────────────────────────────────────────────────

def print_examples(classified: list, n: int, out_dir: Path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"CLASSIFICATION EXAMPLES  (first {n} conversations after filtering)")
    lines.append("=" * 70)

    for ci, conv in enumerate(classified[:n]):
        exchanges = extract_exchanges(conv)
        lines.append(f"\n── Conversation {ci+1}  ({len(exchanges)} exchanges) ──")
        for turn in conv:
            role    = "Human" if turn["role"] == "user" else "AI   "
            cat     = turn["category"]
            preview = textwrap.shorten(turn["preview"], width=120, placeholder="…")
            lines.append(f"  [{role}] ({cat:18s})  {preview}")
        lines.append(
            f"  Exchange sequence: "
            + " → ".join(f"({ABBREV[h]}/{ABBREV[a]})"
                         for h, a in exchanges)
        )

    text = "\n".join(lines)
    print(text)

    out = out_dir / "classification_examples.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",       type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--show-examples", type=int, default=5,
                        help="Number of example conversations to print")
    parser.add_argument("--top-states",    type=int, default=15,
                        help="Number of top states to show in filtered heatmap")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── 1. Classify ───────────────────────────────────────────────────────────
    if CACHE_FILE.exists() and not args.no_cache:
        print(f"Loading cached classifications from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            classified_all = pickle.load(f)
    else:
        raw = load_wildchat(args.samples)
        classified_all = classify_conversations(raw)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(classified_all, f)
        print(f"Cached to {CACHE_FILE}")

    # ── 2. Filter to minimum exchanges ────────────────────────────────────────
    classified = filter_min_exchanges(classified_all, MIN_EXCHANGES)
    n_dropped  = len(classified_all) - len(classified)
    print(f"\n{len(classified_all)} conversations loaded, "
          f"{n_dropped} dropped (< {MIN_EXCHANGES} exchanges), "
          f"{len(classified)} kept.")
    total_exchanges = sum(len(extract_exchanges(c)) for c in classified)
    total_transitions = sum(max(0, len(extract_exchanges(c)) - 1) for c in classified)
    print(f"{total_exchanges} total exchanges, {total_transitions} transitions.\n")

    # ── 3. Example outputs ────────────────────────────────────────────────────
    if args.show_examples > 0:
        print_examples(classified, args.show_examples, OUTPUT_DIR)

    # ── 4. Pair state frequency ───────────────────────────────────────────────
    print("\n=== Pair State Frequency ===")
    pair_counts_flat = Counter()
    for conv in classified:
        for h, a in extract_exchanges(conv):
            pair_counts_flat[pair_idx(h, a)] += 1

    print(f"\nTop 10 most common paired states:")
    for idx, cnt in pair_counts_flat.most_common(10):
        h, a = idx_to_pair(idx)
        pct  = 100 * cnt / max(total_exchanges, 1)
        print(f"  {PAIR_LABELS[idx]:10s}  ({SHORT_LABELS[h]:18s} + {SHORT_LABELS[a]:18s})  "
              f"{cnt:5d}  ({pct:.1f}%)")

    plot_pair_frequency(classified, OUTPUT_DIR)

    # ── 5. Paired transition matrix ───────────────────────────────────────────
    print("\n=== Building Paired Transition Matrix (25×25) ===")
    paired_mat, paired_counts = build_paired_matrix(classified)

    plot_top_transitions(paired_counts, args.top_states, OUTPUT_DIR)
    plot_marginal_matrix(paired_counts, OUTPUT_DIR)

    # ── 6. Top transitions ────────────────────────────────────────────────────
    print("\n=== Top 15 Most Common Transitions ===")
    transition_list = []
    for fi in range(N_STATES):
        for ti in range(N_STATES):
            if paired_counts[fi, ti] > 0:
                transition_list.append((int(paired_counts[fi, ti]),
                                        paired_mat[fi, ti], fi, ti))
    transition_list.sort(reverse=True)
    print(f"  {'From':12s}  →  {'To':12s}  {'Count':>6}  {'P(to|from)':>10}")
    print(f"  {'-'*55}")
    for cnt, prob, fi, ti in transition_list[:15]:
        print(f"  {PAIR_LABELS[fi]:12s}  →  {PAIR_LABELS[ti]:12s}  {cnt:6d}  {prob:10.3f}")

    # ── 7. Stationary distribution ────────────────────────────────────────────
    print("\n=== Stationary Distribution ===")
    stat = stationary_distribution(paired_mat)
    print("Top 10 states in stationary distribution:")
    top_stat = sorted(enumerate(stat), key=lambda x: -x[1])
    for idx, p in top_stat[:10]:
        h, a = idx_to_pair(idx)
        print(f"  {PAIR_LABELS[idx]:10s}  ({SHORT_LABELS[h]:18s} + {SHORT_LABELS[a]:18s})  {p:.4f}")
    plot_stationary_bar(stat, OUTPUT_DIR)

    # ── 8. Time-homogeneity ───────────────────────────────────────────────────
    print("\n=== Time-Homogeneity Check ===")
    plot_time_homogeneity(classified, OUTPUT_DIR)

    # ── 9. Diagonal (simplified) 5-state matrix ───────────────────────────────
    print("\n=== Diagonal 5-State Markov Chain ===")
    plot_diagonal_matrix(classified, OUTPUT_DIR)

    # ── 10. Category distribution (marginal) ─────────────────────────────────
    print("\n=== Marginal Category Distribution ===")
    all_turns   = [t for conv in classified for t in conv]
    human_turns = [t for t in all_turns if t["role"] == "user"]
    ai_turns    = [t for t in all_turns if t["role"] == "assistant"]
    for role_label, turns in [("Human", human_turns), ("AI", ai_turns)]:
        ctr = Counter(t["category"] for t in turns)
        print(f"\n{role_label} ({len(turns)} turns):")
        for cat in SHORT_LABELS:
            pct = 100 * ctr.get(cat, 0) / max(len(turns), 1)
            bar = "█" * int(pct / 2)
            print(f"  {cat:20s} {ctr.get(cat,0):5d}  ({pct:5.1f}%)  {bar}")

    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
