from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import math, random
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","NFLX","AMD","INTC"]
BASE_PRICES = {"AAPL":178,"MSFT":415,"GOOGL":175,"AMZN":195,"TSLA":248,"NVDA":880,"META":520,"NFLX":640,"AMD":175,"INTC":31}

def generate_data(ticker, days=200):
    s0 = BASE_PRICES.get(ticker, 100.0)
    seed = sum(ord(c) for c in ticker)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    mu, sigma = 0.0003, 0.018
    s = s0
    result = []
    base = datetime.now() - timedelta(days=days)
    for i in range(days):
        z = np_rng.standard_normal()
        s = max(s * math.exp((mu - 0.5*sigma**2) + sigma*z), 1.0)
        o = s*(1+rng.uniform(-0.005,0.005))
        c = s
        h = max(o,c)*(1+rng.uniform(0.001,0.02))
        l = min(o,c)*(1-rng.uniform(0.001,0.02))
        result.append({"date":(base+timedelta(days=i)).strftime("%Y-%m-%d"),
            "open":round(o,2),"high":round(h,2),"low":round(l,2),
            "close":round(c,2),"volume":int(rng.uniform(8e6,80e6))})
    return result

def compute_features(prices):
    closes = np.array([p["close"] for p in prices])
    highs  = np.array([p["high"]  for p in prices])
    lows   = np.array([p["low"]   for p in prices])
    vols   = np.array([p["volume"] for p in prices],dtype=float)
    feats = []
    for i in range(20, len(closes)):
        c = closes[i]
        ma5  = np.mean(closes[i-5:i])
        ma10 = np.mean(closes[i-10:i])
        ma20 = np.mean(closes[i-20:i])
        diffs = np.diff(closes[i-15:i+1])
        ag = np.mean(np.maximum(diffs[-14:],0))
        al = np.mean(np.abs(np.minimum(diffs[-14:],0)))+1e-9
        rsi = 100-(100/(1+ag/al))
        std20 = np.std(closes[i-20:i])
        bb_u = ma20+2*std20; bb_l = ma20-2*std20
        bb = (c-bb_l)/(bb_u-bb_l+1e-9)
        macd = np.mean(closes[max(0,i-12):i])-np.mean(closes[max(0,i-26):i])
        vr = vols[i]/(np.mean(vols[i-10:i])+1e-9)
        mom5  = (c-closes[i-5])/(closes[i-5]+1e-9)
        mom10 = (c-closes[i-10])/(closes[i-10]+1e-9)
        atr = np.mean(highs[i-14:i]-lows[i-14:i])
        feats.append([c/ma5-1,c/ma10-1,c/ma20-1,rsi/100,bb,macd/(c+1e-9),vr-1,mom5,mom10,atr/(c+1e-9)])
    return np.array(feats), closes[20:]

class LR:
    def __init__(self): self.w=None; self.b=0; self.mu=None; self.sd=None
    def fit(self, X, y, lr=0.01, ep=500):
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9
        Xn=(X-self.mu)/self.sd; n,d=Xn.shape
        self.w=np.zeros(d)
        for _ in range(ep):
            e=Xn@self.w+self.b-y
            self.w-=lr*(2/n)*(Xn.T@e); self.b-=lr*(2/n)*e.sum()
    def predict(self, X): return (X-self.mu)/self.sd@self.w+self.b

def predict_stock(ticker, fd=14):
    prices = generate_data(ticker, 200)
    feats, tgts = compute_features(prices)
    rets = np.diff(tgts)/(tgts[:-1]+1e-9)
    X,y = feats[:-1], rets
    sp = int(len(X)*0.8)
    m = LR(); m.fit(X[:sp], y[:sp])
    yp = m.predict(X[sp:])
    ss_r = np.sum((y[sp:]-yp)**2); ss_t = np.sum((y[sp:]-y[sp:].mean())**2)
    r2 = max(0,1-ss_r/(ss_t+1e-9))
    lc = prices[-1]["close"]
    ld = datetime.strptime(prices[-1]["date"],"%Y-%m-%d")
    fp,fd_list = [],[]
    cp = lc; cf = feats[-1].reshape(1,-1)
    np_rng = np.random.default_rng(42)
    for i in range(1,fd+1):
        pr = float(m.predict(cf)[0])
        cp *= (1+pr); fp.append(round(max(cp,0.01),2))
        fd_list.append((ld+timedelta(days=i)).strftime("%Y-%m-%d"))
        cf = cf+np_rng.normal(0,0.001,cf.shape)
    hstd = float(np.std(rets)*lc)
    upper=[round(p+hstd*math.sqrt(i+1)*1.2,2) for i,p in enumerate(fp)]
    lower=[round(max(p-hstd*math.sqrt(i+1)*1.2,0.01),2) for i,p in enumerate(fp)]
    closes=[p["close"] for p in prices[-15:]]
    diffs=[closes[i+1]-closes[i] for i in range(len(closes)-1)]
    ag=sum(max(d,0) for d in diffs)/14; al=sum(abs(min(d,0)) for d in diffs)/14+1e-9
    rsi=round(100-(100/(1+ag/al)),1)
    return {"ticker":ticker,"historical":prices[-60:],"forecast":{"dates":fd_list,"prices":fp,"upper":upper,"lower":lower},
        "metrics":{"r2":round(r2,4),"accuracy_pct":round(r2*100,1)},"signal":{"trend":"BULLISH" if fp[-1]>lc else "BEARISH",
        "change_pct":round((fp[-1]-lc)/lc*100,2),"rsi":rsi,"last_close":lc,"volume":prices[-1]["volume"]},"model":"Linear Regression · 10 Technical Indicators","trained_on":f"{sp} samples"}

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/tickers")
def api_tickers():
    out=[]
    for t in TICKERS:
        p=generate_data(t,5); prev=p[-2]["close"]; curr=p[-1]["close"]
        chg=round(curr-prev,2); pct=round(chg/prev*100,2)
        out.append({"ticker":t,"price":curr,"change":chg,"change_pct":pct,"volume":p[-1]["volume"],"bullish":chg>=0})
    return jsonify(out)

@app.route("/api/predict/<ticker>")
def api_predict(ticker):
    ticker=ticker.upper()
    if ticker not in TICKERS: return jsonify({"error":"Not found"}),404
    days=int(request.args.get("days",14))
    return jsonify(predict_stock(ticker, min(days,30)))

@app.route("/api/compare", methods=["POST"])
def api_compare():
    tickers=request.get_json(force=True).get("tickers",["AAPL","MSFT"])[:4]
    result={}
    for t in [x.upper() for x in tickers]:
        if t in TICKERS:
            r=predict_stock(t,14)
            result[t]={"historical_closes":[p["close"] for p in r["historical"]],"dates":[p["date"] for p in r["historical"]],
                "forecast_prices":r["forecast"]["prices"],"forecast_dates":r["forecast"]["dates"],"signal":r["signal"],"metrics":r["metrics"]}
    return jsonify(result)

@app.route("/api/market_overview")
def api_market():
    out=[]
    for t in TICKERS:
        p=generate_data(t,30); closes=[x["close"] for x in p]
        out.append({"ticker":t,"price":closes[-1],"weekly_chg":round((closes[-1]-closes[-5])/closes[-5]*100,2),
            "monthly_chg":round((closes[-1]-closes[0])/closes[0]*100,2),"sparkline":closes[-10:]})
    return jsonify(out)

if __name__=="__main__":
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   StockSense AI  →  localhost:5000   ║")
    print("  ╚══════════════════════════════════════╝\n")
    app.run(debug=True, port=5000)
