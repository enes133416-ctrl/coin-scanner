"""Binance USDT paritelerini çok zaman dilimli puanlar, docs/data.json üretir."""
import json, time, datetime as dt
from pathlib import Path
import requests, numpy as np, pandas as pd

BASE = "https://data-api.binance.vision/api/v3"  # herkese açık veri, API anahtarı gerekmez
MIN_VOL = 5_000_000   # 24 saatlik minimum USDT hacmi
TOP_N = 150           # hacme göre ilk N coin analiz edilir
MAX_24H = 20          # son 24 saatte bu yüzdeden fazla gitmiş coinleri ele
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","EUR","TRY","BUSD","USDE","AEUR","XUSD","USD1","PYUSD","EURI"}

def get(path, **p):
    for _ in range(3):
        try:
            r = requests.get(f"{BASE}/{path}", params=p, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2)
    return None

def klines(sym, itv, limit):
    d = get("klines", symbol=sym, interval=itv, limit=limit)
    if not d or len(d) < 30:
        return None
    df = pd.DataFrame(d).iloc[:, :6]
    df.columns = ["t", "o", "h", "l", "c", "v"]
    return df.astype(float)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr_pct(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean() / df.c * 100

def pct(a, b): return (a / b - 1) * 100

def r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)

def analyze(sym):
    w = klines(sym, "1w", 1000)   # tüm zamanlar
    d = klines(sym, "1d", 1000)   # ~2.7 yıl, yıllık bakış
    h4 = klines(sym, "4h", 200)   # ~33 gün, aylık bakış
    h1 = klines(sym, "1h", 100)   # ~4 gün, günlük bakış
    if any(x is None for x in (w, d, h4, h1)) or len(d) < 210 or len(h1) < 30:
        return None  # çok yeni coinler (<210 gün) atlanır
    p = d.c.iloc[-1]
    sc, why = 0.0, []

    def chk(label, val, wt):
        nonlocal sc
        val = max(0.0, min(1.0, float(val)))
        sc += val * wt
        if val >= 0.6:
            why.append(label)

    # MAKRO (15) - haftalık
    chk("Haftalık trend yukarı", p > ema(w.c, 20).iloc[-1], 8)
    dd = pct(p, w.h.max())
    chk("ATH'den aşırı uzak değil", 1 if dd > -60 else 0.5 if dd > -80 else 0, 7)
    # YIL (20) - günlük
    e50, e200 = ema(d.c, 50), ema(d.c, 200)
    chk("200G ortalama üstünde", p > e200.iloc[-1], 10)
    chk("50G > 200G", e50.iloc[-1] > e200.iloc[-1], 5)
    chk("1Y zirvesine yakın", pct(p, d.h.iloc[-365:].max()) > -40, 5)
    # AY (30) - 4 saatlik + günlük hacim
    chk("4S EMA20 > EMA50", ema(h4.c, 20).iloc[-1] > ema(h4.c, 50).iloc[-1], 8)
    r4 = rsi(h4.c).iloc[-1]
    chk("4S RSI 50-70", 1 if 50 <= r4 <= 70 else 0.5 if (45 <= r4 < 50 or 70 < r4 <= 75) else 0, 7)
    vr = d.v.iloc[-7:].mean() / d.v.iloc[-37:-7].mean()
    chk("Hacim artışı", vr / 1.5, 8)
    hh = d.h.iloc[-21:-1].max()
    chk("20G direnç kırılımı/yakın", 1 if p >= hh else 0.5 if p >= hh * 0.97 else 0, 7)
    # GÜN (15) - saatlik
    r1 = rsi(h1.c).iloc[-1]
    chk("1S RSI 45-70", 1 if 45 <= r1 <= 70 else 0, 5)
    c24 = pct(p, h1.c.iloc[-25])
    chk("24S ılımlı yükseliş", 1 if 0 <= c24 <= 10 else 0.5 if 10 < c24 <= 20 else 0, 5)
    chk("1S EMA20 üstünde", p > ema(h1.c, 20).iloc[-1], 5)
    # VOLATİLİTE (10): %10-50 hareket için coin yeterince hareketli olmalı
    at = atr_pct(d).iloc[-1]
    chk("Yeterli volatilite", at / 4, 10)
    # SIKIŞMA (10): Bollinger bant genişliği yakın zamanda dar mıydı
    bw = (4 * d.c.rolling(20).std() / d.c.rolling(20).mean()).dropna().iloc[-120:]
    chk("Sıkışma sonrası hareket", (bw <= bw.iloc[-10:].min()).mean() < 0.25, 10)

    return dict(
        symbol=sym[:-4], price=r(p, 8), score=r(sc, 1),
        ch_all=r(pct(p, w.o.iloc[0])), ch_1y=r(pct(p, d.c.iloc[-366])) if len(d) > 366 else None,
        ch_1m=r(pct(p, d.c.iloc[-31])), ch_1d=r(c24),
        vol_ratio=r(vr), rsi4h=r(r4, 1), atr=r(at), ath_dd=r(dd, 1), why=why)

def main():
    tick = get("ticker/24hr") or []
    rows = [x for x in tick if x["symbol"].endswith("USDT")
            and x["symbol"][:-4] not in STABLE
            and float(x["quoteVolume"]) >= MIN_VOL
            and float(x["priceChangePercent"]) <= MAX_24H]
    rows.sort(key=lambda x: -float(x["quoteVolume"]))
    items = []
    for x in rows[:TOP_N]:
        try:
            res = analyze(x["symbol"])
        except Exception as e:
            print("hata", x["symbol"], e); continue
        if res:
            res["vol24h"] = r(float(x["quoteVolume"]), 0)
            items.append(res)
        time.sleep(0.15)
    items.sort(key=lambda i: -i["score"])
    out = Path(__file__).parent / "docs" / "data.json"
    out.write_text(json.dumps({"updated": dt.datetime.now(dt.timezone.utc).isoformat(),
                               "count": len(items), "items": items}, ensure_ascii=False))
    print(f"{len(items)} coin yazıldı")

if __name__ == "__main__":
    main()
