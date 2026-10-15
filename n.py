"""
Spelling + grammar corrector trained FROM SCRATCH in PyTorch (no pretrained weights).

TWO STAGES
  Stage 1  SPELLER  - a character-level Transformer (encoder-decoder) that turns a misspelled
           word into the right word. It learns character edit patterns, so it can fix words it
           never saw in training. It is trained on your word pairs + spelling errors found in
           your sentence pairs + synthetic dyslexia-style noise. A spell-checker library
           (pyspellchecker) acts as dictionary / fallback, and a rule layer restores negations
           ("dont" -> "don't").
  Stage 2  TAGGER   - a Transformer encoder that reads the WHOLE sentence and gives every token
           one label: keep it, delete it, or apply a small edit ("add -ed", "replace with went").
           Because it can only keep/delete/edit a token, it cannot rewrite a sentence freely,
           which protects meaning. Negation guards add a second layer of protection.

ANTI-LEAKAGE
  Everything is split by the CORRECT text (all misspellings of one word stay together) on RAW
  pairs, before anything is expanded. Synthetic noise never uses val/test word-file words.

INSTALL
  pip install pandas numpy rapidfuzz pyspellchecker
  pip install torch      (pick the CUDA build from pytorch.org for an NVIDIA GPU; torch >= 2.3)
"""
import json
import math
import os
import random
import re
import string
import time
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein
from spellchecker import SpellChecker


# =============================================================================
# 1. CONFIGURATION (everything tunable lives here)
# =============================================================================
@dataclass
class Config:
    # ---- data ----
    words_csv: str = r"C:\Users\amrrrr\Downloads\projects\Depi Final Project\nlp\data_all.csv"
    sentences_csv: str = r"C:\Users\amrrrr\Downloads\projects\Depi Final Project\nlp\Grammar Correction.csv"
    wrong_col: str = "incorrect"
    correct_col: str = "correct"
    output_dir: str = "saved_model"
    seed: int = 42
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # ---- tokenization ----
    max_tokens: int = 64            # longest sentence (in tokens) processed at once
    max_word_len: int = 20          # longest word (in characters) the speller handles
    max_edit_ratio: float = 0.6     # drop sentence pairs where >60% of tokens change (rewrites)

    # ---- tagger input features: hashed character n-grams + whole-word hash ----
    ngram_min: int = 1
    ngram_max: int = 5
    max_token_chars: int = 16
    ngram_buckets: int = 2 ** 16
    word_buckets: int = 2 ** 15

    # ---- speller model (big: this is where your data is plentiful) ----
    spell_d_model: int = 256
    spell_heads: int = 8
    spell_layers: int = 4           # encoder layers AND decoder layers
    spell_ff: int = 1024
    spell_dropout: float = 0.1
    spell_batch: int = 512
    spell_epochs: int = 25
    spell_patience: int = 5
    spell_lr: float = 1e-3
    spell_val_cap: int = 4000       # max validation words decoded per epoch
    spell_eval_cap: int = 1000      # max test words for the (slow) library-only comparison

    # ---- speller data ----
    synthetic_pairs: int = 200_000  # fake (misspelled, correct) pairs from a word list
    synthetic_clean_fraction: float = 0.1   # share of synthetic pairs left uncorrupted (teaches "keep")
    synthetic_top_words: int = 40_000
    real_repeat: int = 3            # real pairs are repeated so synthetic noise doesn't drown them

    # ---- speller inference rules ----
    spell_min_len: int = 3
    spell_max_edit: int = 3         # never accept a correction further than this from the input
    spell_accept_prob: float = 0.5  # model confidence needed when its output is not in the dictionary
    rare_accept_prob: float = 0.9   # confidence needed to "fix" a word that IS in the dictionary
    rare_word_ratio: float = 50.0   # ...and the fix must be this many times more frequent

    # ---- tagger model (smaller: your sentence file is small, big models memorize it) ----
    tag_d_model: int = 256
    tag_heads: int = 4
    tag_layers: int = 4
    tag_ff: int = 1024
    tag_dropout: float = 0.25
    word_dropout: float = 0.15      # randomly hides whole-word identity so n-grams/context matter
    tag_batch: int = 64
    tag_epochs: int = 60
    tag_patience: int = 10
    tag_lr: float = 5e-4
    synthetic_variants: int = 6     # corrupted copies made from each correct training sentence
    synthetic_error_prob: float = 0.15
    min_class_count: int = 3        # labels rarer than this are ignored in training
    copy_loss_weight: float = 0.4   # <1 down-weights the (very common) "keep" label

    # ---- shared training ----
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.1
    label_smoothing: float = 0.05
    grad_clip: float = 1.0

    # ---- tagger inference guards ----
    default_edit_threshold: float = 0.5   # replaced by a value tuned on validation data
    delete_extra_confidence: float = 0.1  # deleting a word needs extra confidence
    max_negation_edit: int = 2
    max_eval_sentences: int = 5000

    @property
    def max_ngrams(self) -> int:
        padded_len = self.max_token_chars + 2
        return sum(max(padded_len - n + 1, 0) for n in range(self.ngram_min, self.ngram_max + 1))


PAD, BOS, EOS, UNK = 0, 1, 2, 3          # special character ids for the speller
COPY, DEL = "<copy>", "<del>"            # special tagger labels

_NEG = {"not", "no", "never", "cannot", "don't", "didn't", "doesn't", "can't", "won't",
        "isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", "hadn't",
        "shouldn't", "couldn't", "wouldn't"}
NEGATIONS = _NEG | {w.replace("'", "") for w in _NEG}   # also the apostrophe-less spellings

DEMO_SENTENCES = [
    "i dident wnat to go to the doctr",
    "Yesterday I go to the market and buy some bread.",
    "She dont like coffe becuase it is to bitter.",
    "We was happy when they arived.",
    "I did not want to go to the doctor.",                 # already correct: must stay unchanged
    "He have alot of freinds in there school.",
    "Their going to the libary tomorow.",
    "Wen I was yung I never liked vegtables.",
    "Last week my brother buyed a new bike and ridded it to scool.",
    "The childrens was playing outside becuase it wasnt raining.",
]

# Groups of words that are commonly confused. Used ONLY to make synthetic training sentences:
# a correct sentence gets one word swapped for a group-mate, and the tagger learns to undo it.
CONFUSION_GROUPS = [
    ("their", "there", "they're"), ("your", "you're"), ("its", "it's"), ("to", "too"),
    ("then", "than"), ("were", "where", "we're"), ("whose", "who's"), ("affect", "effect"),
    ("loose", "lose"), ("accept", "except"), ("quiet", "quite"), ("weather", "whether"),
    ("a", "an"), ("is", "are"), ("was", "were"), ("has", "have"), ("does", "do"),
    ("go", "went", "gone", "goes"), ("buy", "bought", "buys"), ("eat", "ate", "eaten", "eats"),
    ("see", "saw", "seen", "sees"), ("take", "took", "taken", "takes"),
    ("come", "came", "comes"), ("give", "gave", "given", "gives"),
    ("write", "wrote", "written", "writes"), ("run", "ran", "runs"), ("make", "made", "makes"),
    ("know", "knew", "known", "knows"), ("think", "thought", "thinks"),
    ("bring", "brought", "brings"), ("get", "got", "gets"), ("say", "said", "says"),
    ("tell", "told", "tells"), ("find", "found", "finds"), ("feel", "felt", "feels"),
    ("walk", "walked", "walks"), ("play", "played", "plays"), ("want", "wanted", "wants"),
]


def build_confusion_lookup(groups) -> dict:
    lookup = {}
    for group in groups:
        for word in group:
            if word in NEGATIONS:                      # never touch negations
                continue
            lookup.setdefault(word, set()).update(w for w in group if w != word and w not in NEGATIONS)
    return {w: sorted(alts) for w, alts in lookup.items() if alts}


CONFUSION = build_confusion_lookup(CONFUSION_GROUPS)


# =============================================================================
# 2. TEXT UTILITIES
# =============================================================================
# A token is a word (letters/digits/_, optionally with inner ' or -) or ONE punctuation mark.
_TOKEN_RE = re.compile(r"\w+(?:['\-]\w+)*|[^\w\s]")
levenshtein = Levenshtein.distance      # fast C implementation (number of single-char edits)


def normalize_text(text: str) -> str:
    """Unicode-normalize and straighten curly quotes so "didn’t" == "didn't"."""
    text = unicodedata.normalize("NFKC", str(text))
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def tokens_cased(text: str) -> list:
    return _TOKEN_RE.findall(normalize_text(text))


def tok(text: str) -> list:
    """Lowercase tokens (the models work on lowercase; casing is restored afterwards)."""
    return [t.lower() for t in tokens_cased(text)]


def detok(tokens: list) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:)\]%])", r"\1", text)   # no space BEFORE closing punctuation
    return re.sub(r"([(\[])\s+", r"\1", text)          # no space AFTER an opening bracket


def restore_casing(text: str) -> str:
    text = re.sub(r"\bi\b", "I", text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def match_case(original: str, new: str) -> str:
    """Give `new` the capitalisation style of `original` (names, sentence starts, ALL CAPS)."""
    if not new:
        return new
    if len(original) > 1 and original.isupper():
        return new.upper()
    if original[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def has_negation(text: str) -> bool:
    return any(w in NEGATIONS for w in text.split())


# =============================================================================
# 3. LABELS: how a token's correction is written down for the tagger
# =============================================================================
# <copy>        keep the token
# <del>         delete it
# SUF|k|text    cut the last k characters, then append text   (walk -> walked = SUF|0|ed)
# REPL|text     replace the token with text (may be several words: "alot" -> "a lot")
# Suffix labels are shared across words, so "walked", "jumped", "played" all use ONE class.
def to_label(token: str, output: str) -> str:
    if output == token:
        return COPY
    if output == "":
        return DEL
    if " " not in output and token.isalpha() and output.isalpha():
        prefix = 0
        for a, b in zip(token, output):
            if a != b:
                break
            prefix += 1
        strip, add = len(token) - prefix, output[prefix:]
        if prefix >= 2 and strip <= 4 and len(add) <= 4:
            return f"SUF|{strip}|{add}"
    return f"REPL|{output}"


def apply_label(token: str, label: str) -> str:
    if label == COPY:
        return token
    if label == DEL:
        return ""
    kind, _, rest = label.partition("|")
    if kind == "SUF":
        strip_text, _, add = rest.partition("|")
        strip = int(strip_text)
        return token[:len(token) - strip] + add if strip <= len(token) else token
    return rest


def align_outputs(wrong: list, correct: list) -> list:
    """
    Returns, for every WRONG token, the text it should become ("" = deleted).
    Works when the two sentences have different lengths (run-together / split words).
    """
    outputs = list(wrong)
    matcher = SequenceMatcher(a=wrong, b=correct, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        new = correct[j1:j2]
        if tag == "delete":
            for i in range(i1, i2):
                outputs[i] = ""
        elif tag == "replace":
            if i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    outputs[i1 + k] = new[k]
            else:                                   # 1 -> many, many -> 1, ...
                outputs[i1] = " ".join(new)
                for i in range(i1 + 1, i2):
                    outputs[i] = ""
        elif tag == "insert":
            phrase = " ".join(new)
            if i1 > 0:
                outputs[i1 - 1] = (outputs[i1 - 1] + " " + phrase).strip()
            else:
                outputs[0] = (phrase + " " + outputs[0]).strip()
    return outputs


# =============================================================================
# 4. DATA: loading, leakage-free splitting, synthetic data
# =============================================================================
def load_pairs(path: str, kind: str, cfg: Config) -> list:
    df = pd.read_csv(path, dtype=str, encoding_errors="replace").dropna(subset=[cfg.wrong_col, cfg.correct_col])
    pairs = [(str(w).strip(), str(c).strip(), kind) for w, c in zip(df[cfg.wrong_col], df[cfg.correct_col])]
    return list(dict.fromkeys(p for p in pairs if p[0] and p[1]))   # drop empty rows and duplicates


def group_key(pair: tuple) -> str:
    """All rows with the same CORRECT text belong to one group."""
    return " ".join(tok(pair[1]))


def group_split(pairs: list, val_fraction: float, test_fraction: float, seed: int):
    groups = {}
    for pair in pairs:
        groups.setdefault(group_key(pair), []).append(pair)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test, n_val = int(len(keys) * test_fraction), int(len(keys) * val_fraction)
    test_keys, val_keys = set(keys[:n_test]), set(keys[n_test:n_test + n_val])
    train, val, test = [], [], []
    for key, members in groups.items():
        (test if key in test_keys else val if key in val_keys else train).extend(members)
    return train, val, test


def remove_cross_leakage(train: list, val: list, test: list):
    """After merging the word and sentence files, no group may sit in two splits."""
    test_keys = {group_key(p) for p in test}
    val = [p for p in val if group_key(p) not in test_keys]
    val_keys = {group_key(p) for p in val}
    train = [p for p in train if group_key(p) not in (test_keys | val_keys)]
    assert not ({group_key(p) for p in train} & (val_keys | test_keys)), "train/val/test leakage!"
    return train, val, test


def single_token_pairs(pairs: list, cfg: Config) -> list:
    """(wrong_word, correct_word) for pairs that are one token on both sides."""
    out = []
    for wrong, correct, _ in pairs:
        tw, tc = tok(wrong), tok(correct)
        if len(tw) == 1 and len(tc) == 1 and tw[0] != tc[0] and max(len(tw[0]), len(tc[0])) <= cfg.max_word_len:
            out.append((tw[0], tc[0]))
    return out


def spelling_pairs_from_sentences(pairs: list, lib: SpellChecker, cfg: Config) -> list:
    """Real spelling mistakes hiding inside the sentence file: 1-to-1 swaps to a non-word."""
    out = []
    for wrong, correct, _ in pairs:
        tw, tc = tok(wrong), tok(correct)
        for tag, i1, i2, j1, j2 in SequenceMatcher(a=tw, b=tc, autojunk=False).get_opcodes():
            if tag == "replace" and i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    w, c = tw[i1 + k], tc[j1 + k]
                    if (w.isalpha() and c.isalpha() and len(w) <= cfg.max_word_len and len(c) <= cfg.max_word_len
                            and not lib.known([w]) and levenshtein(w, c) <= 3):
                        out.append((w, c))
    return out


# --- synthetic dyslexia-style noise for single words ---
_PHONETIC = [("ph", "f"), ("f", "ph"), ("ck", "k"), ("k", "c"), ("c", "k"), ("s", "c"), ("c", "s"),
             ("ie", "ei"), ("ei", "ie"), ("tion", "shun"), ("sion", "shun"), ("ough", "uff"),
             ("wh", "w"), ("ght", "t"), ("ai", "ay"), ("ay", "ai"), ("ea", "ee"), ("ee", "ea"),
             ("ou", "ow"), ("ow", "ou"), ("er", "ar"), ("ar", "er"), ("z", "s"), ("s", "z"),
             ("ll", "l"), ("ss", "s"), ("tt", "t"), ("mm", "m"), ("nn", "n"), ("gh", "f"),
             ("qu", "kw"), ("x", "ks"), ("y", "i"), ("i", "y"), ("a", "e"), ("e", "a"), ("o", "u")]
_NOISE_OPS = ["swap", "drop", "double", "insert", "phonetic", "phonetic", "vowel"]


def corrupt_word(word: str, rng: random.Random) -> str:
    """Letter swaps, omissions, doubled letters, phonetic spellings, vowel confusions."""
    for _ in range(10):                         # retry until the result really differs
        w = word
        for _ in range(rng.choice([1, 1, 1, 1, 2, 2])):
            op = rng.choice(_NOISE_OPS)
            if op == "swap" and len(w) >= 2:
                i = rng.randrange(len(w) - 1)
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "drop" and len(w) >= 3:
                i = rng.randrange(len(w))
                w = w[:i] + w[i + 1:]
            elif op == "double" and w:
                i = rng.randrange(len(w))
                w = w[:i] + w[i] + w[i:]
            elif op == "insert":
                i = rng.randrange(len(w) + 1)
                w = w[:i] + rng.choice(string.ascii_lowercase) + w[i:]
            elif op == "phonetic":
                options = [(a, b) for a, b in _PHONETIC if a in w]
                if options:
                    a, b = rng.choice(options)
                    w = w.replace(a, b, 1)
            elif op == "vowel":
                spots = [i for i, ch in enumerate(w) if ch in "aeiou"]
                if spots:
                    i = rng.choice(spots)
                    w = w[:i] + rng.choice("aeiou") + w[i + 1:]
        if w and w != word:
            return w
    return word


def make_synthetic_spelling(cfg: Config, lib: SpellChecker, forbidden: set, extra: set, seed: int) -> list:
    """
    Fake (misspelled, correct) pairs. Source words = frequent dictionary words + training words.
    `forbidden` (val/test words of the word file) are NEVER used, so the test stays honest.
    """
    freq = lib.word_frequency.dictionary
    ranked = sorted(freq, key=freq.get, reverse=True)[:cfg.synthetic_top_words]
    pool = [w for w in dict.fromkeys(ranked + sorted(extra))
            if w.isalpha() and w.isascii() and 3 <= len(w) <= cfg.max_word_len and w not in forbidden]
    rng = random.Random(seed)
    pairs = []
    for _ in range(cfg.synthetic_pairs):
        word = rng.choice(pool)
        if rng.random() < cfg.synthetic_clean_fraction:
            pairs.append((word, word))
        else:
            pairs.append((corrupt_word(word, rng), word))
    return pairs


def build_sentence_examples(pairs: list, spell, cfg: Config):
    """
    Raw sentence pairs -> (spell-corrected wrong tokens, one label per token).
    The tagger is trained on spell-corrected input, exactly what it sees at inference time.
    """
    examples, stats = [], Counter()
    for wrong, correct, _ in pairs:
        wt, ct = tok(wrong), tok(correct)
        if not wt or not ct:
            stats["skipped_empty"] += 1
            continue
        if len(wt) > cfg.max_tokens:
            stats["skipped_too_long"] += 1
            continue
        wt = spell.fix_tokens(wt)
        labels = [to_label(t, o) for t, o in zip(wt, align_outputs(wt, ct))]
        if sum(l != COPY for l in labels) / len(labels) > cfg.max_edit_ratio:
            stats["skipped_rewrite"] += 1
            continue
        examples.append((wt, labels))
        stats["used"] += 1
    return examples, stats


def build_synthetic_examples(train_pairs: list, cfg: Config, seed: int) -> list:
    """Corrupt correct TRAIN sentences with confusion-set swaps (their/there, go/went, has/have...)."""
    rng = random.Random(seed)
    sentences = {}
    for _, correct, kind in train_pairs:
        if kind == "sentence":
            tokens = tok(correct)
            sentences[" ".join(tokens)] = tokens
    examples = []
    for tokens in sentences.values():
        if not 3 <= len(tokens) <= cfg.max_tokens:
            continue
        for _ in range(cfg.synthetic_variants):
            wrong, changed = [], False
            for t in tokens:
                alts = CONFUSION.get(t)
                if alts and rng.random() < cfg.synthetic_error_prob:
                    wrong.append(rng.choice(alts))
                    changed = True
                else:
                    wrong.append(t)
            if changed:
                examples.append((wrong, [to_label(w, t) for w, t in zip(wrong, tokens)]))
    return examples


# =============================================================================
# 5. VECTORIZATION FOR THE TAGGER: deterministic hashed character n-grams
# =============================================================================
@lru_cache(maxsize=None)
def token_features(token, ngram_min, ngram_max, max_chars, ngram_buckets, word_buckets, max_ngrams):
    """(padded tuple of n-gram ids, whole-word id). zlib.crc32 is stable across runs (hash() is not)."""
    padded = "<" + token[:max_chars] + ">"
    ids = []
    for n in range(ngram_min, ngram_max + 1):
        for i in range(len(padded) - n + 1):
            ids.append(1 + zlib.crc32(padded[i:i + n].encode("utf-8")) % ngram_buckets)
    ids = ids[:max_ngrams]
    ids += [0] * (max_ngrams - len(ids))
    return tuple(ids), zlib.crc32(("w|" + token).encode("utf-8")) % word_buckets


class TokenTable:
    """Distinct token -> row number (row 0 = padding). Features are looked up on the GPU."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.index = {"<pad>": 0}
        self.ngram_rows = [tuple([0] * cfg.max_ngrams)]
        self.word_ids = [0]

    def add(self, token: str) -> int:
        idx = self.index.get(token)
        if idx is None:
            c = self.cfg
            ngrams, word_id = token_features(token, c.ngram_min, c.ngram_max, c.max_token_chars,
                                             c.ngram_buckets, c.word_buckets, c.max_ngrams)
            idx = len(self.index)
            self.index[token] = idx
            self.ngram_rows.append(ngrams)
            self.word_ids.append(word_id)
        return idx

    def encode_sentences(self, sentences: list, length: int) -> torch.Tensor:
        out = np.zeros((len(sentences), length), dtype=np.int64)
        for i, tokens in enumerate(sentences):
            for j, t in enumerate(tokens[:length]):
                out[i, j] = self.add(t)
        return torch.from_numpy(out)

    def to_device(self, device):
        ngram = torch.from_numpy(np.array(self.ngram_rows, dtype=np.int32)).to(device)
        word = torch.from_numpy(np.array(self.word_ids, dtype=np.int64)).to(device)
        return ngram, word


# =============================================================================
# 6. MODELS
# =============================================================================
def sinusoidal_positions(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CharSpeller(nn.Module):
    """
    Character-level Transformer encoder-decoder (pre-LayerNorm = stable gradients in deep stacks).
    Input and output embeddings are tied, which cuts parameters and helps on small vocabularies.
    """
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        d = cfg.spell_d_model
        self.d = d
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.max_word_len + 2, d)
        enc_layer = nn.TransformerEncoderLayer(d, cfg.spell_heads, cfg.spell_ff, cfg.spell_dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(d, cfg.spell_heads, cfg.spell_ff, cfg.spell_dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.spell_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.spell_layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, vocab_size, bias=False)
        self.out.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, std=d ** -0.5)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    def embed(self, ids):
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.emb(ids) * math.sqrt(self.d) + self.pos(positions)

    def encode(self, src):
        pad = src == PAD
        return self.encoder(self.embed(src), src_key_padding_mask=pad), pad

    def decode(self, tgt_in, memory, src_pad):
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.decoder(self.embed(tgt_in), memory, tgt_mask=causal,
                         tgt_key_padding_mask=(tgt_in == PAD), memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)


@torch.no_grad()
def greedy_decode(model: CharSpeller, src: torch.Tensor, max_steps: int):
    """Returns (output ids [B, <=max_steps], probability of the whole output sequence [B])."""
    model.eval()
    B = src.size(0)
    memory, src_pad = model.encode(src)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
    log_prob = torch.zeros(B, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_steps):
        logits = model.decode(ys, memory, src_pad)[:, -1].float()
        best_lp, nxt = F.log_softmax(logits, dim=-1).max(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        log_prob = log_prob + torch.where(done, torch.zeros_like(best_lp), best_lp)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        done = done | (nxt == EOS)
        if bool(done.all()):
            break
    return ys[:, 1:], log_prob.exp()


class SentenceTagger(nn.Module):
    """
    Token features (char n-gram bag + word hash) -> Transformer encoder over the whole sentence
    -> one label per token. Pre-LayerNorm, GELU, dropout, word-dropout against memorization.
    """
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        d = cfg.tag_d_model
        self.word_dropout = cfg.word_dropout
        self.ngram_bag = nn.EmbeddingBag(cfg.ngram_buckets + 1, d, mode="mean", padding_idx=0)
        self.word_emb = nn.Embedding(cfg.word_buckets, d)
        self.embed_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.tag_dropout)
        self.register_buffer("pos", sinusoidal_positions(512, d), persistent=False)
        layer = nn.TransformerEncoderLayer(d, cfg.tag_heads, cfg.tag_ff, cfg.tag_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.tag_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.classifier = nn.Linear(d, num_classes)
        nn.init.normal_(self.ngram_bag.weight, std=0.1)
        with torch.no_grad():
            self.ngram_bag.weight[0].zero_()
        nn.init.normal_(self.word_emb.weight, std=0.1)

    def forward(self, ngram_ids, word_ids, pad_mask):
        B, L, N = ngram_ids.shape
        chars = self.ngram_bag(ngram_ids.reshape(B * L, N)).view(B, L, -1)
        words = self.word_emb(word_ids)
        if self.training and self.word_dropout > 0:
            keep = (torch.rand(B, L, 1, device=words.device) >= self.word_dropout).to(words.dtype)
            words = words * keep
        x = self.drop(self.embed_norm(chars + words) + self.pos[:L])
        return self.classifier(self.encoder(x, src_key_padding_mask=pad_mask))


# =============================================================================
# 7. TRAINING HELPERS
# =============================================================================
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_setup(device):
    """bf16 on recent NVIDIA GPUs (no loss scaling needed), fp16 + scaler on older ones, else off."""
    if device.type != "cuda":
        return None, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def autocast_ctx(device, amp_dtype):
    return torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                          dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None)


def make_optimizer(model, lr: float, cfg: Config, total_steps: int):
    """AdamW + OneCycle (warm-up, then decay): small early steps avoid unstable starts."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(total_steps, 1),
                                                pct_start=cfg.warmup_fraction)
    return opt, sched


def encode_words(words: list, stoi: dict, width: int, with_bos: bool) -> torch.Tensor:
    arr = np.zeros((len(words), width), dtype=np.int64)
    for i, w in enumerate(words):
        seq = ([BOS] if with_bos else []) + [stoi.get(ch, UNK) for ch in w[:width - 2]] + [EOS]
        arr[i, :len(seq)] = seq
    return torch.from_numpy(arr)


def ids_to_text(ids: list, vocab: list) -> str:
    out = []
    for i in ids:
        if i in (EOS, PAD):
            break
        if i > UNK:
            out.append(vocab[i])
    return "".join(out)


# =============================================================================
# 8. SPELLER TRAINING
# =============================================================================
def save_speller(path, model, cfg, vocab, lexicon, epoch, metrics):
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "char_vocab": vocab,
                "lexicon_extra": sorted(lexicon), "epoch": int(epoch), "val_metrics": metrics}, path)


@torch.no_grad()
def speller_validate(model, v_src, v_tgt, vocab_size, width):
    model.eval()
    n = len(v_src)
    loss_sum, counted, correct = 0.0, 0, 0
    for s in range(0, n, 2048):
        x, y = v_src[s:s + 2048], v_tgt[s:s + 2048]
        logits = model(x, y[:, :-1]).float()
        loss_sum += float(F.cross_entropy(logits.reshape(-1, vocab_size), y[:, 1:].reshape(-1),
                                          ignore_index=PAD, reduction="sum"))
        counted += int((y[:, 1:] != PAD).sum())
        pred, _ = greedy_decode(model, x, width - 1)
        pred = F.pad(pred, (0, width - 1 - pred.size(1)), value=PAD)
        correct += int((pred == y[:, 1:]).all(1).sum())
    return {"loss": loss_sum / max(counted, 1), "exact": correct / max(n, 1)}


def train_speller(cfg, vocab, lexicon, train_pairs, val_pairs, device, out_dir):
    stoi = {c: i for i, c in enumerate(vocab)}
    width = cfg.max_word_len + 2
    V = len(vocab)
    src = encode_words([w for w, _ in train_pairs], stoi, width, False).to(device)
    tgt = encode_words([c for _, c in train_pairs], stoi, width, True).to(device)
    val_pairs = val_pairs[:cfg.spell_val_cap]
    v_src = encode_words([w for w, _ in val_pairs], stoi, width, False).to(device)
    v_tgt = encode_words([c for _, c in val_pairs], stoi, width, True).to(device)

    model = CharSpeller(cfg, V).to(device)
    print(f"[speller] parameters: {sum(p.numel() for p in model.parameters()):,} | "
          f"train pairs: {len(train_pairs):,} | val pairs: {len(val_pairs):,}")
    N = len(train_pairs)
    steps = (N + cfg.spell_batch - 1) // cfg.spell_batch
    opt, sched = make_optimizer(model, cfg.spell_lr, cfg, cfg.spell_epochs * steps)
    amp_dtype, use_scaler = amp_setup(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)

    history, best_exact, bad = [], -1.0, 0
    for epoch in range(1, cfg.spell_epochs + 1):
        model.train()
        start = time.time()
        order = torch.randperm(N, device=device)
        loss_sum = torch.zeros((), device=device)     # accumulated ON the GPU: no per-step sync
        grad_sum = torch.zeros((), device=device)
        for s in range(0, N, cfg.spell_batch):
            idx = order[s:s + cfg.spell_batch]
            x, y = src[idx], tgt[idx]
            with autocast_ctx(device, amp_dtype):
                logits = model(x, y[:, :-1])
            loss = criterion(logits.float().reshape(-1, V), y[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach()
            grad_sum += grad_norm.detach()
        val = speller_validate(model, v_src, v_tgt, V, width)
        record = {"epoch": epoch, "train_loss": float(loss_sum) / steps,
                  "mean_grad_norm": float(grad_sum) / steps,
                  "val_loss": float(val["loss"]), "val_exact": float(val["exact"])}
        history.append(record)
        save_speller(out_dir / "speller_last.pt", model, cfg, vocab, lexicon, epoch, val)
        improved = val["exact"] > best_exact
        if improved:
            best_exact, bad = val["exact"], 0
            save_speller(out_dir / "speller_best.pt", model, cfg, vocab, lexicon, epoch, val)
        else:
            bad += 1
        print(f"[speller] epoch {epoch:02d} | {time.time() - start:5.1f}s | train_loss {record['train_loss']:.4f} | "
              f"val_loss {val['loss']:.4f} | val_exact {val['exact']:.3f} | "
              f"grad_norm {record['mean_grad_norm']:.2f}{'  <- best' if improved else ''}")
        if bad >= cfg.spell_patience:
            print(f"[speller] early stopping (no improvement for {cfg.spell_patience} epochs)")
            break
    return history


# =============================================================================
# 9. SPELL STAGE (neural model + dictionary library + negation rules)
# =============================================================================
class SpellStage:
    """
    Decision order for each distinct word (results are cached):
      1. punctuation / numbers / very short words        -> unchanged
      2. "dont", "wasnt", ...                             -> "don't", "wasn't" (rule)
      3. unknown word within 1 edit of a negation         -> that negation ("dident" -> "didn't")
      4. neural proposal, accepted if the dictionary knows it OR the model is confident
         (for words the dictionary already knows, only if it is a clearly commoner word)
      5. otherwise the pyspellchecker suggestion
    A correction may never add or remove a negation.
    """
    def __init__(self, ckpt_path, device, lib=None):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.cfg = Config(**ckpt["config"])
        self.vocab = ckpt["char_vocab"]
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.extra = set(ckpt["lexicon_extra"])
        self.epoch = ckpt["epoch"]
        self.device = device
        self.lib = lib if lib is not None else SpellChecker(distance=2)
        self.model = CharSpeller(self.cfg, len(self.vocab)).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.plain_to_neg = {w.replace("'", ""): w for w in NEGATIONS if "'" in w}
        self.cache = {}

    # ---- helpers ----
    def is_known(self, t: str) -> bool:
        return t in self.extra or bool(self.lib.known([t]))

    def freq(self, t: str) -> int:
        return self.lib.word_frequency.dictionary.get(t, 0)

    def library_correct(self, t: str) -> str:
        best = self.lib.correction(t)
        return best if best else t

    @torch.no_grad()
    def model_predict(self, tokens: list) -> list:
        """Neural proposal + sequence probability for every token (batched on the GPU)."""
        results = [(t, 0.0) for t in tokens]
        usable = [i for i, t in enumerate(tokens) if 0 < len(t) <= self.cfg.max_word_len]
        width = self.cfg.max_word_len + 2
        for s in range(0, len(usable), 2048):
            part = usable[s:s + 2048]
            src = encode_words([tokens[i] for i in part], self.stoi, width, False).to(self.device)
            ids, probs = greedy_decode(self.model, src, width - 1)
            for i, row, p in zip(part, ids.tolist(), probs.tolist()):
                results[i] = (ids_to_text(row, self.vocab), p)
        return results

    def _choose(self, token: str, proposal: str, prob: float) -> str:
        known = self.is_known(token)
        same_negation = lambda w: (w in NEGATIONS) == (token in NEGATIONS)
        usable = (proposal and proposal != token and proposal.isalpha()
                  and levenshtein(token, proposal) <= self.cfg.spell_max_edit and same_negation(proposal))
        if usable:
            if known:   # a real word: only replace it with a MUCH commoner word, and only if sure
                if (prob >= self.cfg.rare_accept_prob and self.is_known(proposal)
                        and self.freq(proposal) >= self.cfg.rare_word_ratio * max(self.freq(token), 1)):
                    return proposal
                return token
            if self.is_known(proposal) or prob >= self.cfg.spell_accept_prob:
                return proposal
        if known:
            return token
        fallback = self.library_correct(token)
        if fallback != token and same_negation(fallback):
            return fallback
        return token

    def prime(self, tokens):
        """Computes and caches the correction of every distinct token, using batched GPU calls."""
        todo = []
        for t in dict.fromkeys(tokens):
            if t in self.cache:
                continue
            if t in self.plain_to_neg:
                self.cache[t] = self.plain_to_neg[t]
            elif not (t.isalpha() and len(t) >= self.cfg.spell_min_len):
                self.cache[t] = t
            elif not self.is_known(t) and len(t) >= 4 and any(
                    levenshtein(t, plain) <= 1 for plain in self.plain_to_neg):
                self.cache[t] = next(neg for plain, neg in self.plain_to_neg.items() if levenshtein(t, plain) <= 1)
            else:
                todo.append(t)
        if todo:
            for t, (proposal, prob) in zip(todo, self.model_predict(todo)):
                self.cache[t] = self._choose(t, proposal, prob)

    def fix_tokens(self, tokens: list) -> list:
        self.prime(tokens)
        return [self.cache[t] for t in tokens]


# =============================================================================
# 10. TAGGER TRAINING
# =============================================================================
def tensorize(table: TokenTable, examples: list, class_to_id: dict, length: int, device):
    tok_ids = table.encode_sentences([e[0] for e in examples], length)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    for i, (_, labs) in enumerate(examples):
        for j, lab in enumerate(labs[:length]):
            labels[i, j] = class_to_id.get(lab, -100)        # unknown/rare label -> ignored
    lens = (tok_ids != 0).sum(1)                              # stays on the CPU
    return tok_ids.to(device), labels.to(device), lens


def iter_batches(tensors, order: torch.Tensor, batch: int):
    """Slices GPU-resident data; each batch is trimmed to its longest sentence."""
    tok_ids, labels, lens = tensors
    for s in range(0, len(order), batch):
        idx_cpu = order[s:s + batch]
        longest = max(int(lens[idx_cpu].max()), 1)
        idx = idx_cpu.to(tok_ids.device)
        yield tok_ids[idx, :longest], labels[idx, :longest]


@torch.no_grad()
def collect_tagger_outputs(model, tensors, tables, copy_id: int, batch: int = 256):
    """Per-token truth id, best non-copy label id and its probability, for tokens with a known label."""
    model.eval()
    ngram_tab, word_tab = tables
    truth, lab, conf = [], [], []
    loss_sum, counted = 0.0, 0
    for t, y in iter_batches(tensors, torch.arange(len(tensors[0])), batch):
        logits = model(ngram_tab[t].long(), word_tab[t], t == 0).float()
        valid = y != -100
        if not bool(valid.any()):
            continue
        loss_sum += float(F.cross_entropy(logits[valid], y[valid], reduction="sum"))
        counted += int(valid.sum())
        probs = logits.softmax(-1)
        probs[..., copy_id] = 0.0                              # only "edit" candidates compete
        c, l = probs.max(-1)
        truth.append(y[valid]); lab.append(l[valid]); conf.append(c[valid])
    if not truth:
        return None
    return {"loss": loss_sum / counted, "truth": torch.cat(truth).cpu().numpy(),
            "label": torch.cat(lab).cpu().numpy(), "conf": torch.cat(conf).cpu().numpy()}


def best_threshold(out: dict, copy_id: int, beta: float = 0.5):
    """
    Picks the confidence threshold with the best F0.5 (precision counts double) on validation data.
    This replaces guessing a threshold by hand, and is the main lever for fewer false corrections.
    """
    truth, label, conf = out["truth"], out["label"], out["conf"]
    needs = truth != copy_id
    best = {"threshold": 0.5, "f_beta": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    for thr in np.arange(0.10, 0.96, 0.05):
        pred = np.where(conf >= thr, label, copy_id)
        made = pred != copy_id
        tp = int(((pred == truth) & needs).sum())
        precision, recall = tp / max(int(made.sum()), 1), tp / max(int(needs.sum()), 1)
        denom = beta ** 2 * precision + recall
        f = (1 + beta ** 2) * precision * recall / denom if denom > 0 else 0.0
        if f > best["f_beta"]:
            best = {"threshold": float(round(thr, 2)), "f_beta": float(f), "precision": float(precision),
                    "recall": float(recall), "accuracy": float((pred == truth).mean())}
    return best


def save_tagger(path, model, cfg, class_names, epoch, metrics, history, threshold):
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "class_names": class_names,
                "epoch": int(epoch), "val_metrics": metrics, "history": history,
                "edit_threshold": float(threshold)}, path)


def train_tagger(cfg, model, class_names, train_t, val_t, tables, device, out_dir):
    copy_id = class_names.index(COPY)
    N = len(train_t[0])
    steps = (N + cfg.tag_batch - 1) // cfg.tag_batch
    opt, sched = make_optimizer(model, cfg.tag_lr, cfg, cfg.tag_epochs * steps)
    amp_dtype, use_scaler = amp_setup(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    weights = torch.ones(len(class_names), device=device)
    weights[copy_id] = cfg.copy_loss_weight
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100, label_smoothing=cfg.label_smoothing)
    ngram_tab, word_tab = tables

    history, best_key, bad = [], (-1.0, 0.0), 0
    for epoch in range(1, cfg.tag_epochs + 1):
        model.train()
        start = time.time()
        order = torch.randperm(N)
        loss_sum = torch.zeros((), device=device)
        grad_sum = torch.zeros((), device=device)
        for t, y in iter_batches(train_t, order, cfg.tag_batch):
            with autocast_ctx(device, amp_dtype):
                logits = model(ngram_tab[t].long(), word_tab[t], t == 0)
            loss = criterion(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach()
            grad_sum += grad_norm.detach()

        out = collect_tagger_outputs(model, val_t, tables, copy_id)
        tuned = best_threshold(out, copy_id) if out else {"threshold": cfg.default_edit_threshold, "f_beta": 0.0,
                                                        "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
        val_loss = out["loss"] if out else float("nan")
        record = {"epoch": epoch, "train_loss": float(loss_sum) / steps, "mean_grad_norm": float(grad_sum) / steps,
                  "val_loss": float(val_loss), **{f"val_{k}": v for k, v in tuned.items()}}
        history.append(record)
        save_tagger(out_dir / "tagger_last.pt", model, cfg, class_names, epoch, tuned, history, tuned["threshold"])
        key = (tuned["f_beta"], -val_loss if val_loss == val_loss else 0.0)   # F0.5 first, then loss
        improved = key > best_key
        if improved:
            best_key, bad = key, 0
            save_tagger(out_dir / "tagger_best.pt", model, cfg, class_names, epoch, tuned, history, tuned["threshold"])
        else:
            bad += 1
        print(f"[tagger] epoch {epoch:02d} | {time.time() - start:5.1f}s | train_loss {record['train_loss']:.4f} | "
              f"val_loss {val_loss:.4f} | val_F0.5 {tuned['f_beta']:.3f} (thr {tuned['threshold']:.2f}, "
              f"P {tuned['precision']:.2f} R {tuned['recall']:.2f}) | "
              f"grad_norm {record['mean_grad_norm']:.2f}{'  <- best' if improved else ''}")
        if bad >= cfg.tag_patience:
            print(f"[tagger] early stopping (no improvement for {cfg.tag_patience} epochs)")
            break
    return history


# =============================================================================
# 11. INFERENCE
# =============================================================================
class TaggerPredictor:
    """Loads a tagger checkpoint; applies the tuned threshold plus the meaning guards."""
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.cfg = Config(**ckpt["config"])
        self.class_names = ckpt["class_names"]
        self.class_set = set(self.class_names)
        self.copy_id = self.class_names.index(COPY)
        self.epoch = ckpt["epoch"]
        self.threshold = ckpt.get("edit_threshold", self.cfg.default_edit_threshold)
        self.device = device
        self.model = SentenceTagger(self.cfg, len(self.class_names)).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _decide(self, token: str, label_id: int, conf: float) -> str:
        label = self.class_names[label_id]
        needed = self.threshold + (self.cfg.delete_extra_confidence if label == DEL else 0.0)
        if conf < needed:
            return COPY                                        # not sure enough -> do nothing
        out = apply_label(token, label)
        if out == token:
            return COPY
        if token in NEGATIONS and not has_negation(out):
            return COPY                                        # guard 1: never remove a negation
        if has_negation(out) and token not in NEGATIONS and levenshtein(token, out) > self.cfg.max_negation_edit:
            return COPY                                        # guard 2: never invent a negation
        return label

    @torch.no_grad()
    def predict_labels(self, sentences: list) -> list:
        """sentences: list of token lists -> list of guarded label lists (same lengths)."""
        L = self.cfg.max_tokens
        chunks, owners = [], []
        for i, tokens in enumerate(sentences):               # long sentences are processed in chunks
            for s in range(0, len(tokens), L):
                chunks.append(tokens[s:s + L])
                owners.append(i)
        results = [[] for _ in sentences]
        if not chunks:
            return results
        table = TokenTable(self.cfg)
        ids = table.encode_sentences(chunks, L).to(self.device)
        ngram_tab, word_tab = table.to_device(self.device)
        for s in range(0, len(chunks), 256):
            part = chunks[s:s + 256]
            longest = max(len(c) for c in part)
            t = ids[s:s + 256, :longest]
            probs = self.model(ngram_tab[t].long(), word_tab[t], t == 0).float().softmax(-1)
            probs[..., self.copy_id] = 0.0
            conf, lab = probs.max(-1)
            conf, lab = conf.cpu().tolist(), lab.cpu().tolist()
            for r, chunk in enumerate(part):
                results[owners[s + r]].extend(
                    self._decide(tk, lab[r][j], conf[r][j]) for j, tk in enumerate(chunk))
        return results


class Corrector:
    """The full pipeline: spell stage, then (optionally) the tagger, then casing restored."""
    def __init__(self, spell: SpellStage, tagger=None):
        self.spell, self.tagger = spell, tagger

    def correct_texts(self, texts: list) -> list:
        cased = [tokens_cased(t) for t in texts]
        lower = [[w.lower() for w in c] for c in cased]
        spelled = [self.spell.fix_tokens(l) for l in lower]
        labels = self.tagger.predict_labels(spelled) if self.tagger else [[COPY] * len(s) for s in spelled]
        outputs = []
        for orig, low, sp, labs in zip(cased, lower, spelled, labels):
            pieces = []
            for o, l, s, lab in zip(orig, low, sp, labs):
                final = apply_label(s, lab)
                if final:
                    pieces.append(o if final == l else match_case(o, final))   # untouched words keep their case
            outputs.append(restore_casing(detok(pieces)))
        return outputs


def load_corrector(out_dir="saved_model", which="best", device=None) -> Corrector:
    """Reuse the trained pipeline anywhere:  load_corrector().correct_texts(["i dont no"])"""
    device = device or pick_device()
    out = Path(out_dir)
    return Corrector(SpellStage(out / "speller_best.pt", device), TaggerPredictor(out / f"tagger_{which}.pt", device))


# =============================================================================
# 12. EVALUATION
# =============================================================================
def evaluate_speller(stage: SpellStage, pairs: list, cfg: Config) -> dict:
    """Word-level: neural only vs dictionary library only vs the hybrid used in the pipeline."""
    if not pairs:
        return {}
    pairs = random.Random(cfg.seed).sample(pairs, min(len(pairs), cfg.spell_eval_cap))
    wrongs, rights = [w for w, _ in pairs], [c for _, c in pairs]
    n = len(pairs)
    acc = lambda out: sum(o == r for o, r in zip(out, rights)) / n
    model_only = [m for m, _ in stage.model_predict(wrongs)]
    library_only = [stage.library_correct(w) for w in wrongs]
    hybrid = stage.fix_tokens(wrongs)
    kept = stage.fix_tokens(rights)
    return {"n": n, "do_nothing": acc(wrongs), "model_only": acc(model_only), "library_only": acc(library_only),
            "hybrid": acc(hybrid), "hybrid_keeps_correct_words": sum(k == r for k, r in zip(kept, rights)) / n}


def evaluate_tagger(predictor: TaggerPredictor, examples: list) -> dict:
    if not examples:
        return {}
    sentences, truths = [e[0] for e in examples], [e[1] for e in examples]
    preds = predictor.predict_labels(sentences)
    tokens = [t for s in sentences for t in s]
    truth = [l for ls in truths for l in ls]
    pred = [l for ls in preds for l in ls]
    n = len(truth)
    needs = np.array([t != COPY for t in truth])
    made = np.array([p != COPY for p in pred])
    right = np.array([p == t for p, t in zip(pred, truth)])
    clean = int((~needs).sum())
    tp = int((right & needs).sum())
    broken = sum(tk in NEGATIONS and not has_negation(apply_label(tk, p)) for tk, p in zip(tokens, pred))
    return {
        "n_tokens": n,
        "accuracy": float(right.mean()),
        "do_nothing_accuracy": clean / n,
        "edit_recall": tp / max(int(needs.sum()), 1),
        "edit_precision": tp / max(int(made.sum()), 1),
        "false_edit_rate": int(((~needs) & made).sum()) / max(clean, 1),
        "target_reachable": sum(t in predictor.class_set for t in truth) / n,
        "negations_broken": int(broken),
    }


def evaluate_sentences(fn, pairs: list, limit: int) -> dict:
    """Whole-sentence metrics on RAW test pairs. fn: list[str] -> list[str]."""
    pairs = pairs[:limit]
    if not pairs:
        return {}
    wrongs, corrects = [p[0] for p in pairs], [p[1] for p in pairs]
    outputs = fn(wrongs)
    norm = lambda s: " ".join(tok(s))
    chars = max(sum(len(norm(c)) for c in corrects), 1)
    return {
        "n_sentences": len(pairs),
        "exact_match": sum(norm(o) == norm(c) for o, c in zip(outputs, corrects)) / len(pairs),
        "do_nothing_exact": sum(norm(w) == norm(c) for w, c in zip(wrongs, corrects)) / len(pairs),
        "char_error_rate": sum(levenshtein(norm(o), norm(c)) for o, c in zip(outputs, corrects)) / chars,
        "do_nothing_cer": sum(levenshtein(norm(w), norm(c)) for w, c in zip(wrongs, corrects)) / chars,
    }


def print_table(results: dict, title: str):
    names = list(results)
    flat = {n: {f"{split}/{k}": v for split, d in results[n].items() for k, v in d.items()} for n in names}
    keys = sorted(set().union(*[set(f) for f in flat.values()]))
    print(f"\n=== {title} ===\n{'metric':<40}" + "".join(f"{n:>16}" for n in names))
    for k in keys:
        print(f"{k:<40}" + "".join(f"{flat[n].get(k, float('nan')):>16.4f}" for n in names))


# =============================================================================
# 13. MAIN
# =============================================================================
def main(cfg: Config = None):
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = pick_device()
    torch.set_num_threads(os.cpu_count() or 1)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"device: {device}  (no CUDA GPU found: install the CUDA build of PyTorch to use the GPU)")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. load RAW pairs and split by group BEFORE expanding anything ----
    words = load_pairs(cfg.words_csv, "word", cfg)
    sentences = load_pairs(cfg.sentences_csv, "sentence", cfg)
    print(f"loaded {len(words)} word pairs, {len(sentences)} sentence pairs")
    parts = [group_split(d, cfg.val_fraction, cfg.test_fraction, cfg.seed) for d in (words, sentences)]
    train_pairs, val_pairs, test_pairs = remove_cross_leakage(
        parts[0][0] + parts[1][0], parts[0][1] + parts[1][1], parts[0][2] + parts[1][2])
    random.Random(cfg.seed).shuffle(train_pairs)
    print(f"pairs -> train {len(train_pairs)} | val {len(val_pairs)} | test {len(test_pairs)}")

    # ---- 2. SPELLER data: real word pairs + real spelling errors from sentences + synthetic noise ----
    lib = SpellChecker(distance=2)
    extra = {t for _, c, _ in train_pairs for t in tok(c) if t.isalpha()}          # training vocabulary only
    forbidden = {t for _, c, k in val_pairs + test_pairs if k == "word" for t in tok(c)}
    real = (single_token_pairs([p for p in train_pairs if p[2] == "word"], cfg)
            + spelling_pairs_from_sentences([p for p in train_pairs if p[2] == "sentence"], lib, cfg))
    synthetic = make_synthetic_spelling(cfg, lib, forbidden, extra, cfg.seed)
    speller_train = real * cfg.real_repeat + synthetic
    random.Random(cfg.seed).shuffle(speller_train)
    speller_val = single_token_pairs([p for p in val_pairs if p[2] == "word"], cfg)
    test_word_pairs = single_token_pairs([p for p in test_pairs if p[2] == "word"], cfg)
    print(f"speller data: {len(real)} real + {len(synthetic)} synthetic (real x{cfg.real_repeat})")
    chars = sorted(set(string.ascii_lowercase + "'-") | {ch for w, c in speller_train + speller_val for ch in w + c})
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars

    # ---- 3. train + test the speller (best AND last) ----
    speller_history = train_speller(cfg, vocab, extra, speller_train, speller_val, device, out_dir)
    spell_stages, speller_results = {}, {}
    for name in ("best", "last"):
        stage = SpellStage(out_dir / f"speller_{name}.pt", device, lib)
        label = f"{name}(ep{stage.epoch})"
        spell_stages[name] = stage
        speller_results[label] = {"words": evaluate_speller(stage, test_word_pairs, cfg)}
    print_table(speller_results, "SPELLER on held-out WORD pairs (words the model never saw)")
    spell = spell_stages["best"]            # the pipeline uses the best speller

    # ---- 4. TAGGER data: spell-corrected sentences + synthetic confusion-set sentences ----
    all_tokens = {t for p in train_pairs + val_pairs + test_pairs if p[2] == "sentence" for t in tok(p[0])}
    spell.prime(list(all_tokens))
    train_s = [p for p in train_pairs if p[2] == "sentence"]
    real_train, stats = build_sentence_examples(train_s, spell, cfg)
    val_ex, _ = build_sentence_examples([p for p in val_pairs if p[2] == "sentence"], spell, cfg)
    test_ex, _ = build_sentence_examples([p for p in test_pairs if p[2] == "sentence"], spell, cfg)
    synth_train = build_synthetic_examples(train_pairs, cfg, cfg.seed)
    train_ex = real_train + synth_train
    random.Random(cfg.seed).shuffle(train_ex)
    print(f"tagger sentences -> real train {len(real_train)} + synthetic {len(synth_train)} | "
          f"val {len(val_ex)} | test {len(test_ex)} | alignment stats {dict(stats)}")
    counts = Counter(l for _, ls in train_ex for l in ls)
    class_names = [COPY, DEL] + sorted(l for l, c in counts.items()
                                       if c >= cfg.min_class_count and l not in (COPY, DEL))
    class_to_id = {n: i for i, n in enumerate(class_names)}
    print(f"tagger classes: {len(class_names)} | tokens needing an edit (train): "
          f"{1 - counts[COPY] / max(sum(counts.values()), 1):.1%}")

    table = TokenTable(cfg)
    train_t = tensorize(table, train_ex, class_to_id, cfg.max_tokens, device)
    val_t = tensorize(table, val_ex, class_to_id, cfg.max_tokens, device)
    tables = table.to_device(device)

    # ---- 5. train the tagger ----
    tagger = SentenceTagger(cfg, len(class_names)).to(device)
    print(f"[tagger] parameters: {sum(p.numel() for p in tagger.parameters()):,}")
    tagger_history = train_tagger(cfg, tagger, class_names, train_t, val_t, tables, device, out_dir)

    # ---- 6. test BOTH tagger checkpoints (and spell-only) end to end ----
    test_sentence_pairs = [p for p in test_pairs if p[2] == "sentence"]
    predictors, pipelines, results = {}, {"spell_only": Corrector(spell, None)}, {"spell_only": {}}
    results["spell_only"]["sentences"] = evaluate_sentences(
        pipelines["spell_only"].correct_texts, test_sentence_pairs, cfg.max_eval_sentences)
    for name in ("best", "last"):
        predictor = TaggerPredictor(out_dir / f"tagger_{name}.pt", device)
        label = f"{name}(ep{predictor.epoch},thr{predictor.threshold:.2f})"
        predictors[label] = predictor
        pipelines[label] = Corrector(spell, predictor)
        results[label] = {
            "sentence_tokens": evaluate_tagger(predictor, test_ex),
            "sentences": evaluate_sentences(pipelines[label].correct_texts, test_sentence_pairs,
                                            cfg.max_eval_sentences),
        }
    print_table(results, "FULL PIPELINE on held-out SENTENCE pairs")

    # ---- 7. qualitative demo ----
    outputs = {label: p.correct_texts(DEMO_SENTENCES) for label, p in pipelines.items()}
    for i, sentence in enumerate(DEMO_SENTENCES):
        print(f"\nIN  : {sentence}")
        for label in pipelines:
            print(f"{label[:22]:<22}: {outputs[label][i]}")

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "speller_results": speller_results, "pipeline_results": results,
                   "speller_history": speller_history, "tagger_history": tagger_history,
                   "demo": {"inputs": DEMO_SENTENCES, "outputs": outputs}}, f, indent=2)
    print(f"\nSaved in {out_dir.resolve()}: speller_best/last.pt, tagger_best/last.pt, report.json")


if __name__ == "__main__":
    main()
