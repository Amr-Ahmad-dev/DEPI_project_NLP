
import re, random
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPClassifier
import os
import joblib
from pathlib import Path

# ----------------------------- CONFIG (edit these) -----------------------------
WORDS_CSV = r"C:\Users\amrrrr\Downloads\projects\Depi Final Project\nlp\data_all.csv"
SENTENCES_CSV = r"C:\Users\amrrrr\Downloads\projects\Depi Final Project\nlp\Grammar Correction.csv"

WRONG_COL, CORRECT_COL = "incorrect", "correct"

HIDDEN = (512, 256, 128)
MAX_ITER = 40
ALPHA = 1e-4
MIN_CONF = 0.30
SEED = 42

BATCH_SIZE = 128

MODEL_DIR = Path("saved_model")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "grammar_corrector.joblib"

NEGATIONS = {
    "not", "no", "never", "don't", "didn't", "doesn't",
    "can't", "won't", "isn't", "wasn't", "aren't",
    "weren't", "cannot"
}

random.seed(SEED)
np.random.seed(SEED)


os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())   # set BEFORE importing numpy/sklearn
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

# 1. float32 halves memory and speeds up the math
def hv(n): return HashingVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                                    n_features=n, alternate_sign=False,
                                    norm="l2", dtype=np.float32)

# 2. smaller feature space = smaller first layer (biggest single win)
hv_cur, hv_ctx = hv(2**13), hv(2**11)     # ~12k features instead of ~49k

# 3. bigger batches = far fewer Adam updates per epoch
clf = MLPClassifier(hidden_layer_sizes=HIDDEN, batch_size=512, max_iter=MAX_ITER, alpha=ALPHA,
                    random_state=SEED, verbose=True, early_stopping=False)



# ----------------------------- TOKENIZATION -----------------------------
# Words and punctuation become separate tokens. Everything is lowercased (see notes).
def tok(s): return re.findall(r"\w+(?:'\w+)?|[^\w\s]", str(s).lower())

def detok(tokens):
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))

# ----------------------------- BUILD TOKEN-LEVEL EXAMPLES -----------------------------
# Each example: (prev_token, current_token, next_token, corrected_token)
def examples_from_words(path):
    df = pd.read_csv(path, dtype=str).dropna(subset=[WRONG_COL, CORRECT_COL])
    out = []
    for w, c in zip(df[WRONG_COL], df[CORRECT_COL]):
        tw, tc = tok(w), tok(c)
        if len(tw) == 1 and len(tc) == 1:
            out.append(("<s>", tw[0], "</s>", tc[0]))
    return out

def examples_from_sentences(path):
    df = pd.read_csv(path, dtype=str).dropna(subset=[WRONG_COL, CORRECT_COL])
    out, skipped = [], 0
    for w, c in zip(df[WRONG_COL], df[CORRECT_COL]):
        tw, tc = tok(w), tok(c)
        if len(tw) != len(tc):          # needs word-by-word alignment
            skipped += 1; continue
        padded = ["<s>"] + tw + ["</s>"]
        for i in range(len(tw)):
            out.append((padded[i], padded[i + 1], padded[i + 2], tc[i]))
    print(f"sentence pairs skipped (different token counts): {skipped}/{len(df)}")
    return out

def split(ex, val=0.1, test=0.1):
    ex = ex[:]; random.shuffle(ex)
    n = len(ex); a, b = int(n * test) , int(n * (test + val))
    return ex[b:], ex[a:b], ex[:a]         # train, val, test


wtr, wva, wte = split(examples_from_words(WORDS_CSV))#idk lek
str_, sva, ste = split(examples_from_sentences(SENTENCES_CSV))#same)
train, val = wtr+str_, wva+sva#Just try moving the word from the vocabulary. It does not make any sense and potentially hurts performance. 
random.shuffle(train)
print(f"train={len(train)} val={len(val)} test_words={len(wte)} test_sent_tokens={len(ste)}")

# ----------------------------- VECTORIZATION -----------------------------
# Character n-grams (1-4) of the current token, plus char n-grams of the previous
# and next tokens, hashed into fixed-size sparse vectors.
def hv(n): return HashingVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                                    n_features=n, alternate_sign=False, norm="l2")
hv_cur , hv_ctx = hv(2**15) , hv(2**13)

def featurize(rows):
    prev = [r[0] for r in rows]; cur = [r[1] for r in rows]; nxt = [r[2] for r in rows]
    return hstack([hv_cur.transform(cur), hv_ctx.transform(prev), hv_ctx.transform(nxt)]).tocsr()

Xtr, ytr = featurize(train), [r[3] for r in train]
Xva, yva = featurize(val),   [r[3] for r in val]

# ----------------------------- MODEL + TRAINING -----------------------------
# Output classes = every corrected token seen in training.
# (early_stopping is off: it needs stratified splits, which fail on rare classes.)


clf = MLPClassifier(hidden_layer_sizes=HIDDEN, alpha=ALPHA, max_iter=MAX_ITER,
                    batch_size=128, random_state=SEED, verbose=True)

clf.fit(Xtr, ytr)

print("classes:", len(clf.classes_))

print("val token accuracy:", np.mean(clf.predict(Xva) == np.array(yva)))

# ----------------------------- INFERENCE (with meaning guards) -----------------------------
def lev(a, b):

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def predict_tokens(rows):

    P = clf.predict_proba(featurize(rows))

    idx , out = P.argmax(1) , []

    for (_, cur, _, *_), k, p in zip(rows, idx, P.max(1)):

        pred = clf.classes_[k]

        keep = p < MIN_CONF

        # Guard 1: a negation in the input must stay a negation.

        if cur in NEGATIONS and pred not in NEGATIONS: keep = True

        # Guard 2: never introduce a negation that isn't a near-spelling of the input.

        if pred in NEGATIONS and cur not in NEGATIONS and lev(cur, pred) > 2: keep = True

        out.append(cur if keep else pred)

    return out



def correct_text(text):

    t = tok(text); padded = ["<s>"] + t + ["</s>"]

    rows = [(padded[i], padded[i + 1], padded[i + 2]) for i in range(len(t))]

    return detok(predict_tokens(rows)) if rows else ""


# ----------------------------- EVALUATION -----------------------------

def evaluate(data, name):
    if not data: return

    preds = predict_tokens(data)

    refs, srcs = [r[3] for r in data], [r[1] for r in data]

    acc  = np.mean([p == r for p, r in zip(preds, refs)])

    base = np.mean([s == r for s, r in zip(srcs, refs)])

    known = np.mean([r in set(clf.classes_) for r in refs])

    # Meaning check: negation status of every token must match the reference.

    neg_ok = np.mean([(p in NEGATIONS) == (r in NEGATIONS) for p, r in zip(preds, refs)])

    broke = sum((s in NEGATIONS) and (p not in NEGATIONS) for s, p in zip(srcs, preds))

    print(f"[{name}] n={len(data)} acc={acc:.3f} (do-nothing={base:.3f}) "
          
          f"ref_in_vocab={known:.3f} negation_match={neg_ok:.3f} negations_broken={broke}")

evaluate(wte, "test/words")
evaluate(ste, "test/sentence-tokens")



print(correct_text("i dident wnat to go to the doctr"))



joblib.dump(
    {
        "model": clf,
        "hv_cur": hv_cur,
        "hv_ctx": hv_ctx,
        "min_conf": MIN_CONF,
        "negations": NEGATIONS,
        "hidden": HIDDEN,
        "max_iter": MAX_ITER,
        "alpha": ALPHA,
        "seed": SEED
    },
    MODEL_PATH,
    compress=3
)

print(f"Model saved to: {MODEL_PATH.resolve()}")