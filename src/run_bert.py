"""BERT baseline on the leakage-free hold-out split (Protocol C).

Each student is turned into a sentence ("gender is female. age at enrollment is 19. ...") and a BERT model is fine-tuned
for 3-class classification with class-weighted cross-entropy. The split is IDENTICAL to run_protocol_c.py (stratified 80/20,
seed 42); a validation part (10% of the training part) selects the best epoch, the 885 hold-out students are used once.
For a paired comparison the proposed 4-model ensemble is re-fitted on the same training part (LightGBM parameters from
results/optuna_best.json) and compared with exact McNemar and a bootstrap interval.

Usage:   python run_bert.py [--model bert-base-uncased] [--seeds 3] [--epochs 5] [--fast] [--tiny-test]
  --fast       small subset, 1 epoch (smoke test; needs the pretrained model download)
  --tiny-test  random tiny BERT + locally trained tokenizer (no download; only checks that the code runs)
Needs a GPU for reasonable speed (Colab T4: roughly 3-6 min per epoch for bert-base).
Output: results/bert_holdout.csv, results/bert_vs_ensemble.csv, results/bert_confusion.json, results/bert_text_examples.txt
"""
import argparse, json, time
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max_len", type=int, default=384)
ap.add_argument("--fast", action="store_true"); ap.add_argument("--tiny-test", action="store_true")
args = ap.parse_args()
if args.fast or args.tiny_test: args.epochs, args.seeds = 1, 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ text serialisation
YESNO = {0: "no", 1: "yes"}
CATMAP = {"Gender": {0: "female", 1: "male"}, "Daytime/evening attendance": {0: "evening", 1: "daytime"},
          "Displaced": YESNO, "Educational special needs": YESNO, "Debtor": YESNO, "Tuition fees up to date": YESNO,
          "Scholarship holder": YESNO, "International": YESNO}

def to_text(row):
    parts = []
    for c, v in row.items():
        name = c.replace("Curricular units", "units").replace("sem (", "semester ").replace("(", "").replace(")", "").lower().strip()
        mp = CATMAP.get(c.strip())
        val = mp.get(int(v), str(v)) if mp else (f"{v:.1f}" if isinstance(v, float) and not float(v).is_integer() else str(int(v)) if float(v).is_integer() else f"{v:.1f}")
        parts.append(f"{name} is {val}")
    return ". ".join(parts) + "."

_, X, y = load_data()
texts = np.array([to_text(r) for _, r in X.iterrows()], dtype=object)
(RESULTS / "bert_text_examples.txt").write_text("\n\n".join(texts[:3]))
print("example:", texts[0][:300], "...")

# same split as run_protocol_c.py
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
if args.fast or args.tiny_test:                                     # smoke test on a small subset of the training part
    idx_tr = np.random.RandomState(0).choice(idx_tr, 400, replace=False)
tr_sub, va_sub = train_test_split(np.arange(len(idx_tr)), test_size=0.1, stratify=y[idx_tr], random_state=SEED)
i_fit, i_val = idx_tr[tr_sub], idx_tr[va_sub]
print("fit", len(i_fit), "val", len(i_val), "hold-out", len(idx_te), "device", DEV)

# ------------------------------------------------------------------ tokenizer and model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup
if args.tiny_test:
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertTokenizerFast
    import tempfile, os
    d = tempfile.mkdtemp(); open(f"{d}/corpus.txt", "w").write("\n".join(texts))
    _wp = BertWordPieceTokenizer(); _wp.train([f"{d}/corpus.txt"], vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]); _wp.save_model(d)
    tok = BertTokenizerFast(vocab_file=f"{d}/vocab.txt", do_lower_case=True)
    make_model = lambda: BertForSequenceClassification(BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, num_labels=3))
else:
    tok = AutoTokenizer.from_pretrained(args.model)
    make_model = lambda: AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

enc = tok(list(texts), truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
n_tok = (enc["attention_mask"].sum(1)).numpy()
print(f"token length: mean {n_tok.mean():.0f}, max {n_tok.max()}, truncated at {args.max_len}: {(n_tok >= args.max_len).mean():.1%}")

def batches(idx, shuffle, rng):
    idx = np.array(idx); idx = rng.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), args.batch):
        b = idx[s:s + args.batch]
        yield b, {k: v[b].to(DEV) for k, v in enc.items()}, torch.tensor(y[b], device=DEV)

@torch.no_grad()
def predict(model, idx):
    model.eval(); out = []
    for _, inp, _ in batches(idx, False, None):
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            out.append(torch.softmax(model(**inp).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)

def run_seed(seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model = make_model().to(DEV)
    cw = torch.tensor([class_weights(y[i_fit], "balanced")[i] for i in range(3)], dtype=torch.float, device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(i_fit) / args.batch)); sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda"); best, best_state = -1, None
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        for _, inp, lab in batches(i_fit, True, rng):
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**inp).logits.float(), lab, weight=cw)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        mv = metrics(y[i_val], predict(model, i_val).argmax(1))["macro_f1"]
        print(f"seed {seed} epoch {ep + 1}/{args.epochs} val macro-F1 {mv:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if mv > best: best, best_state = mv, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model, idx_te)

probas = [run_seed(s) for s in range(args.seeds)]
rows = []
for s, p in enumerate(probas):
    rows.append(dict(model=f"{args.model} (seed {s})", **metrics(y[idx_te], p.argmax(1))))
df = pd.DataFrame(rows); mean = df.drop(columns="model").mean(); sd = df.drop(columns="model").std(ddof=1) if len(df) > 1 else mean * 0
summary = pd.concat([df, pd.DataFrame([dict(model=f"{args.model} MEAN over {args.seeds} seeds", **mean), dict(model="SD over seeds", **sd)])])

# ------------------------------------------------------------------ paired comparison with the proposed ensemble
opt = json.load(open(RESULTS / "optuna_best.json")) if (RESULTS / "optuna_best.json").exists() else dict(params=BASE_LGB, class_weight="balanced")
Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
ens = np.mean([m.fit(Xtr, ytr).predict_proba(Xte) for m in ensemble_members(opt["params"], opt["class_weight"])], axis=0).argmax(1)
bert0 = probas[0].argmax(1)
summary = pd.concat([summary, pd.DataFrame([dict(model="Proposed ensemble (same split)", **metrics(yte, ens))])])
summary.round(5).to_csv(RESULTS / "bert_holdout.csv", index=False)

fa = lambda yt, yp: metrics(yt, yp)
rng = np.random.RandomState(0); d = {k: [] for k in ["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]}
for _ in range(1000 if not (args.fast or args.tiny_test) else 100):
    ix = rng.randint(0, len(yte), len(yte)); a, b = fa(yte[ix], ens[ix]), fa(yte[ix], bert0[ix])
    for k in d: d[k].append(a[k] - b[k])
ens_ok, bert_ok = ens == yte, bert0 == yte
b01, b10 = int((ens_ok & ~bert_ok).sum()), int((~ens_ok & bert_ok).sum())
p_mc = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0
comp = pd.DataFrame([dict(metric=k, ensemble_minus_bert=float(np.mean(v)), ci_low=float(np.percentile(v, 2.5)), ci_high=float(np.percentile(v, 97.5)), mcnemar_exact_p_accuracy=p_mc) for k, v in d.items()])
comp.round(5).to_csv(RESULTS / "bert_vs_ensemble.csv", index=False)
json.dump(dict(bert_seed0=confusion_matrix(yte, bert0, labels=[0, 1, 2]).tolist(), ensemble=confusion_matrix(yte, ens, labels=[0, 1, 2]).tolist()), open(RESULTS / "bert_confusion.json", "w"))
print(summary.round(4).to_string(index=False)); print(comp.round(4).to_string(index=False))
