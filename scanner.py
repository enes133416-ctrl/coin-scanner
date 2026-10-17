"""Her coin icin 7 gun icinde %20 yukselme olasiligini hesaplar -> docs/data.json"""
import json, sys, time, datetime as dt
from pathlib import Path
import requests, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASE = "https://data-api.binance.vision/api/v3"
TARGET, DAYS = 0.20, 7        # hedef: 7 gunde %20
TRAIN_N, MIN_VOL = 250, 2_000_000
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","EUR","TRY","BUSD","USDE","AEUR","XUSD","USD1","PYUSD","EURI","RLUSD","U"}
TR_URLS = ["https://www.trbinance.com/open/v1/common/symbols",
           "https://www.binance.tr/open/v1/common/symbols"]
FEATS = ["r1","r5","r30","r90","r180","r365","ath_dd","hi365","lo365","rsi","e20","e50","e200",
         "atr","vr","rv30","bbw_rank","brk20","maxup30","btc5","btc30"]

def tr_coins():
    for u in TR_URLS:
        try:
            lst = requests.get(u, timeout=20).json()["data"]["list"]
            out = {(x.get("baseAsset") or str(x.get("symbol","")).split("_")[0]).upper() for x in lst}
            if len(out) > 20:
                print("TR listesi API:", len(out)); return out
        except Exception as e:
            print("TR API hatasi", e)
    f = Path(__file__).parent / "tr_coins.txt"
    if f.exists():
        out = {c.strip().upper() for c in f.read_text().replace(",", "\n").split() if c.strip()}
        if out:
            print("TR listesi dosya:", len(out)); return out
    sys.exit("HATA: Binance TR listesi alinamadi. tr_coins.txt ekle.")

def get(path, **p):
    for _ in range(3):
        try:
            r = requests.get(f"{BASE}/{path}", params=p, timeout=20)
            if r.status_code == 200: return r.json()
        except requests.RequestException: pass
        time.sleep(2)
    return None

def daily(sym):
    d = get("klines", symbol=sym, interval="1d", limit=1000)
    if not d or len(d) < 250: return None
    df = pd.DataFrame(d).iloc[:, :6]; df.columns = ["t","o","h","l","c","v"]
    df = df.astype(float); df.index = df.t.astype("int64"); return df

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
def atr_pct(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h-df.l, (df.h-pc).abs(), (df.l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean() / df.c * 100

def feats(d, btc):
    c, f = d.c, pd.DataFrame(index=d.index)
    for n in (1, 5, 30, 90, 180, 365): f[f"r{n}"] = c.pct_change(n) * 100
    f["ath_dd"] = (c / d.h.cummax() - 1) * 100
    f["hi365"] = (c / d.h.rolling(365, min_periods=90).max() - 1) * 100
    f["lo365"] = (c / d.l.rolling(365, min_periods=90).min() - 1) * 100
    f["rsi"] = rsi(c)
    for n in (20, 50, 200): f[f"e{n}"] = (c / ema(c, n) - 1) * 100
    f["atr"] = atr_pct(d)
    f["vr"] = d.v.rolling(7).mean() / d.v.rolling(30).mean()
    f["rv30"] = c.pct_change().rolling(30).std() * 100
    bbw = 4 * c.rolling(20).std() / c.rolling(20).mean()
    f["bbw_rank"] = bbw.rolling(120, min_periods=30).rank(pct=True)
    f["brk20"] = (c / d.h.shift(1).rolling(20).max() - 1) * 100
    f["maxup30"] = c.pct_change().rolling(30).max() * 100
    f["btc5"], f["btc30"] = btc["r5"].reindex(d.index), btc["r30"].reindex(d.index)
    fut = d.h.rolling(DAYS).max().shift(-DAYS)
    f["y"] = np.where(fut.notna(), (fut / c - 1 >= TARGET).astype(float), np.nan)
    return f.replace([np.inf, -np.inf], np.nan)

def mk():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                          l2_regularization=1.0, random_state=0)

def main():
    tick = get("ticker/24hr") or []
    tr = tr_coins()
    rows = [x for x in tick if x["symbol"].endswith("USDT") and x["symbol"][:-4] not in STABLE
            and float(x["quoteVolume"]) >= 300_000]
    rows.sort(key=lambda x: -float(x["quoteVolume"]))
    syms = [x["symbol"] for x in rows if float(x["quoteVolume"]) >= MIN_VOL][:TRAIN_N]
    syms += [x["symbol"] for x in rows if x["symbol"][:-4] in tr and x["symbol"] not in syms]
    b = daily("BTCUSDT")
    btc = pd.DataFrame({"r5": b.c.pct_change(5)*100, "r30": b.c.pct_change(30)*100})
    frames, latest = [], []
    for s in syms:
        d = daily(s); time.sleep(0.1)
        if d is None or atr_pct(d).median() < 0.6: continue
        f = feats(d, btc); base = s[:-4]
        frames.append(f.dropna(subset=["r90", "atr", "rsi", "y"]).assign(sym=base))
        if base in tr:
            last = f.iloc[-1]
            latest.append((base, d.c.iloc[-1], last, d.c.iloc[-1] / d.h.max() - 1))
    data = pd.concat(frames)
    tmax = data.index.max(); cut = tmax - 90 * 86400000
    train, test = data[data.index < cut - 7 * 86400000], data[data.index >= cut]
    m = mk().fit(train[FEATS], train.y)
    p = m.predict_proba(test[FEATS])[:, 1]
    thr = np.quantile(p, 0.95)
    metrics = dict(base_rate=round(float(test.y.mean()) * 100, 2),
                   auc=round(float(roc_auc_score(test.y, p)), 3),
                   top5_hit=round(float(test.y[p >= thr].mean()) * 100, 2),
                   test_rows=int(len(test)), coins=int(data.sym.nunique()))
    print(metrics)
    m = mk().fit(data[FEATS], data.y)
    items = []
    for base, price, last, _ in latest:
        x = last[FEATS].to_frame().T.astype(float)
        if x[["r90", "atr", "rsi"]].isna().any(axis=None): continue
        pr = float(m.predict_proba(x)[:, 1][0]) * 100
        g = lambda k: None if pd.isna(last[k]) else round(float(last[k]), 2)
        items.append(dict(symbol=base, price=round(float(price), 8), prob=round(pr, 1),
                          r1=g("r1"), r5=g("r5"), r30=g("r30"), r90=g("r90"), r180=g("r180"),
                          r365=g("r365"), ath_dd=g("ath_dd"), rsi=g("rsi"), atr=g("atr")))
    items.sort(key=lambda i: -i["prob"])
    out = Path(__file__).parent / "docs" / "data.json"
    out.write_text(json.dumps({"updated": dt.datetime.now(dt.timezone.utc).isoformat(),
                               "metrics": metrics, "target": int(TARGET*100), "days": DAYS,
                               "items": items}, ensure_ascii=False))
    print(len(items), "coin yazildi")

if __name__ == "__main__":
    main()
