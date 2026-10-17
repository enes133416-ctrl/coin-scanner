"""Her coin icin 7 gun icinde %20 yukselme olasiligini hesaplar -> docs/data.json"""
import json, sys, time, datetime as dt
from pathlib import Path
import requests, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

BASE = "https://data-api.binance.vision/api/v3"
TARGET, DAYS = 0.20, 7        # hedef: 7 gunde %20
TRAIN_N, MIN_VOL = 250, 2_000_000
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","EUR","TRY","BUSD","USDE","AEUR","XUSD","USD1","PYUSD","EURI","RLUSD","U"}
TR_URLS = ["https://www.trbinance.com/open/v1/common/symbols",
           "https://www.binance.tr/open/v1/common/symbols"]
STOP, FEE = 10.0, 0.2          # zarar kes %10, komisyon+kayma %0.2
FL = {"r1":"1G getiri","r5":"5G getiri","r7":"7G getiri","r30":"1A getiri","r90":"3A getiri","r180":"6A getiri",
      "r365":"1Y getiri","ath_dd":"ATH uzaklığı","hi365":"1Y zirve uzaklığı","lo365":"1Y dip üstü","rsi":"RSI",
      "e20":"EMA20 farkı","e50":"EMA50 farkı","e200":"EMA200 farkı","atr":"Volatilite","vr":"Hacim artışı",
      "rv30":"30G oynaklık","bbw_rank":"Bant sıkışması","brk20":"20G kırılım","maxup30":"30G en büyük gün",
      "btc5":"BTC 5G","btc30":"BTC 30G","cpos":"Gün kapanış konumu","uw":"Üst fitil","body":"Gün mumu",
      "up5":"Son 5G yeşil gün","rng5":"5G aralık","volz":"Hacim sapması","dd30":"30G zirveden düşüş","sl50":"EMA50 eğimi"}
FEATS = list(FL)

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
    for n in (1, 5, 7, 30, 90, 180, 365): f[f"r{n}"] = c.pct_change(n) * 100
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
    f["cpos"] = (c - d.l) / (d.h - d.l).replace(0, np.nan)
    f["uw"] = (d.h - np.maximum(d.o, c)) / c * 100
    f["body"] = (c - d.o) / d.o * 100
    f["up5"] = (c.diff() > 0).rolling(5).sum()
    f["rng5"] = (d.h.rolling(5).max() / d.l.rolling(5).min() - 1) * 100
    f["volz"] = (d.v - d.v.rolling(30).mean()) / d.v.rolling(30).std()
    f["dd30"] = (c / d.h.rolling(30).max() - 1) * 100
    f["sl50"] = ema(c, 50).pct_change(10) * 100
    fh, fl = d.h.rolling(DAYS).max().shift(-DAYS), d.l.rolling(DAYS).min().shift(-DAYS)
    f["fh"], f["fl"] = (fh / c - 1) * 100, (fl / c - 1) * 100
    f["c7"] = (c.shift(-DAYS) / c - 1) * 100
    f["yu"] = np.where(fh.notna(), (f.fh >= TARGET * 100).astype(float), np.nan)
    f["yd"] = np.where(fh.notna(), (f.fl <= -TARGET * 100).astype(float), np.nan)
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
        frames.append(f.dropna(subset=["r90", "atr", "rsi", "yu"]).assign(sym=base))
        if base in tr:
            last = f.iloc[-1]
            latest.append((base, d.c.iloc[-1], last, d.c.iloc[-1] / d.h.max() - 1))
    data = pd.concat(frames); D = 86400000
    tmax = data.index.max(); oos = []
    for k in (180, 120, 60):             # ileri yürüyen test: model hiç görmediği dönemi tahmin eder
        st = tmax - k * D
        trn, te = data[data.index < st - 7 * D], data[(data.index >= st) & (data.index < st + 60 * D)]
        mu, md = mk().fit(trn[FEATS], trn.yu), mk().fit(trn[FEATS], trn.yd)
        oos.append(te.assign(pu=mu.predict_proba(te[FEATS])[:, 1], pdn=md.predict_proba(te[FEATS])[:, 1]))
    oos = pd.concat(oos)
    thr = np.quantile(oos.pu, 0.95)
    t = oos[oos.sym.isin(tr)].copy(); t["day"] = t.index // D; t["sc"] = t.pu - t.pdn
    t["ret"] = np.where(t.fl <= -STOP, -STOP, np.where(t.fh >= TARGET * 100, TARGET * 100, t.c7)) - FEE
    pk = t.sort_values("sc", ascending=False).groupby("day").head(3)
    metrics = dict(base_rate=round(float(oos.yu.mean()) * 100, 2), auc=round(float(roc_auc_score(oos.yu, oos.pu)), 3),
                   top5_hit=round(float(oos.yu[oos.pu >= thr].mean()) * 100, 2),
                   bt_trades=int(len(pk)), bt_win=round(float((pk.ret > 0).mean()) * 100, 1),
                   bt_avg=round(float(pk.ret.mean()), 2), bt_bench=round(float(t.ret.mean()), 2),
                   bt_target=round(float(((pk.fh >= TARGET * 100) & (pk.fl > -STOP)).mean()) * 100, 1),
                   bt_stop=round(float((pk.fl <= -STOP).mean()) * 100, 1), coins=int(data.sym.nunique()))
    print(metrics)
    sm = oos.sample(min(15000, len(oos)), random_state=0)
    pi = permutation_importance(mu, sm[FEATS], sm.yu, scoring="roc_auc", n_repeats=2, random_state=0)
    imp = sorted(zip(FEATS, pi.importances_mean), key=lambda z: -z[1])[:6]
    r2 = lambda v: None if pd.isna(v) else round(float(v), 1)
    up, dn = data[(data.yu == 1) & (data.yd == 0)], data[(data.yd == 1) & (data.yu == 0)]
    prof = [dict(f=FL[k], up=r2(up[k].median()), down=r2(dn[k].median()), all=r2(data[k].median()))
            for k in ["r7", "r30", "r90", "r365", "rsi", "vr", "atr", "dd30", "ath_dd"]]
    lessons = dict(importance=[[FL[k], round(float(v), 4)] for k, v in imp], profile=prof,
                   n_up=int(len(up)), n_down=int(len(dn)))
    mu, md = mk().fit(data[FEATS], data.yu), mk().fit(data[FEATS], data.yd)
    med = data[FEATS].median(); items = []
    for base, price, last, _ in latest:
        row = last[FEATS].astype(float)
        if row[["r90", "atr", "rsi"]].isna().any(): continue
        X = pd.DataFrame([row.values] * (len(FEATS) + 1), columns=FEATS)
        for i, fn in enumerate(FEATS): X.loc[i, fn] = med[fn]
        pu_all = mu.predict_proba(X)[:, 1]; pu0 = pu_all[-1]; dl = pu0 - pu_all[:-1]
        order = np.argsort(dl)
        why = [f"{FL[FEATS[i]]} {row.iloc[i]:.1f}" for i in order[::-1][:3] if dl[i] > 0]
        risk = [f"{FL[FEATS[i]]} {row.iloc[i]:.1f}" for i in order[:2] if dl[i] < 0]
        pdn = float(md.predict_proba(X.iloc[[-1]])[:, 1][0]) * 100
        g = lambda k: None if pd.isna(last[k]) else round(float(last[k]), 2)
        items.append(dict(symbol=base, price=round(float(price), 8), pu=round(pu0 * 100, 1), pdn=round(pdn, 1),
                          score=round(pu0 * 100 - pdn, 1), why=why, risk=risk,
                          r1=g("r1"), r5=g("r5"), r7=g("r7"), r30=g("r30"), r90=g("r90"), r180=g("r180"), r365=g("r365")))
    items.sort(key=lambda i: -i["score"])
    out = Path(__file__).parent / "docs" / "data.json"
    out.write_text(json.dumps({"updated": dt.datetime.now(dt.timezone.utc).isoformat(), "metrics": metrics,
                               "lessons": lessons, "target": int(TARGET * 100), "days": DAYS, "stop": STOP,
                               "items": items}, ensure_ascii=False))
    print(len(items), "coin yazildi")

if __name__ == "__main__":
    main()
