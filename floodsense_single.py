import os, re, math, time, threading, json, warnings
import requests

# Enable Windows system certificate truststore if available
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Suppress sklearn unpickling / version warnings cleanly
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify, Response

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "floodsense_model.joblib")
PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__)

# ---------------- SSL & session management ----------------
_session = requests.Session()
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def safe_request_get(url, params=None, timeout=25, headers=None):
    headers = headers or {"User-Agent": "FloodSenseAI/1.0"}
    try:
        return _session.get(url, params=params, timeout=timeout, headers=headers)
    except requests.exceptions.SSLError:
        return _session.get(url, params=params, timeout=timeout, headers=headers, verify=False)

# ---------------- cached upstream access ----------------
UPSTREAM = {
    "weather": "https://api.open-meteo.com",
    "flood": "https://flood-api.open-meteo.com",
    "geo": "https://photon.komoot.io",
    "nominatim": "https://nominatim.openstreetmap.org",
    "overpass": "https://overpass-api.de",
    "gdacs": "https://www.gdacs.org"
}
TTL = {"weather": 300, "flood": 1800, "geo": 86400, "nominatim": 86400, "overpass": 1800, "gdacs": 900}
_cache, _lock, _hits = {}, threading.Lock(), {}

def fetch_cached(svc, path, qs):
    key = f"{svc}/{path}?{qs}"
    with _lock:
        hit = _cache.get(key)
    if hit and hit[0] > time.time():
        return 200, hit[1]
    
    url = f"{UPSTREAM[svc]}/{path}"
    r = safe_request_get(url, params=qs or None, timeout=25, headers={"User-Agent": "FloodSenseAI/1.0"})
    if r.ok:
        with _lock:
            if len(_cache) > 1000:
                _cache.pop(next(iter(_cache)))
            _cache[key] = (time.time() + TTL.get(svc, 300), r.content)
    return r.status_code, r.content

def get_json(svc, path, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    status, body = fetch_cached(svc, path, qs)
    if status >= 400:
        raise RuntimeError(f"{svc} HTTP {status}")
    return json.loads(body)

def rate_ok(ip):
    now = time.time()
    with _lock:
        if len(_hits) > 500:
            stale = [k for k, v in _hits.items() if not v or now - v[-1] > 300]
            for k in stale:
                _hits.pop(k, None)
        a = [t for t in _hits.get(ip, []) if now - t < 60] + [now]
        _hits[ip] = a
    return len(a) <= 300

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

@app.route("/proxy/<svc>/<path:rest>", methods=["GET", "OPTIONS"])
def proxy(svc, rest):
    if request.method == "OPTIONS":
        return Response(status=200)
    if svc not in UPSTREAM:
        return jsonify(error="Unknown service"), 404
    if not rate_ok(request.remote_addr):
        return jsonify(error="Too many requests"), 429
    try:
        status, body = fetch_cached(svc, rest, request.query_string.decode())
    except requests.RequestException as e:
        return jsonify(error=f"Upstream unavailable: {e}"), 502
    return Response(body, status=status, content_type="application/json")

# ---------------- risk: formula + trained model ----------------
LEVELS = [
    (.75, "CRITICAL", "Evacuate low-lying blocks, deploy boats and pumps, close underpasses"),
    (.5, "HIGH", "Send SMS alert, pre-position pumps, divert traffic"),
    (.3, "WATCH", "Clear drains, notify ward officers, check gauges every 15 min"),
    (0, "NORMAL", "Routine monitoring")
]
level = lambda p: next(l for l in LEVELS if p >= l[0])
clamp = lambda v, a, b: min(max(v, a), b)

def formula(v):
    s = (max(v["rain"] - v["dr"] * .8, 0) / 20 + v["f3"] / 150
         + v["river"] / 6 * (1 - v["el"] / 35) + v["soil"] * .6 - v["el"] / 40)
    return 1 / (1 + math.exp(-(s - 1.4) / .14))

MODEL, MODEL_COLS, MODEL_ERR = None, [], None
try:
    import joblib
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        if isinstance(obj, dict):
            MODEL = obj.get("model") or obj.get("pipeline") or obj.get("clf")
            MODEL_COLS = list(obj.get("features") or obj.get("feature_names") or [])
        else:
            MODEL = obj
        if not MODEL_COLS:
            MODEL_COLS = list(getattr(MODEL, "feature_names_in_", []))
    else:
        MODEL_ERR = f"Model file not found at {MODEL_PATH}"
except Exception as e:
    MODEL_ERR = str(e)

def match_feature(name):
    n = name.lower()
    # Check drainage before rain to avoid matching "drain" with "rain"
    if "drain" in n: return "dr"
    if "pop" in n: return "pop"
    if "soil" in n or "moisture" in n: return "soil"
    if "river" in n or "discharge" in n: return "river"
    if "elev" in n or "height" in n or "altitude" in n: return "el"
    if n == "f3" or any(k in n for k in ("3h", "next3", "next_3", "f3", "forecast")): return "f3"
    if "rain" in n or "precip" in n: return "rain"
    return None

def model_predict(v):
    if MODEL is None:
        return None, f"model not loaded ({MODEL_ERR})"
    if not MODEL_COLS:
        return None, "model has no feature names; send the training columns"
    row, missing = {}, []
    for n in MODEL_COLS:
        k = match_feature(n)
        if k is None or k not in v:
            missing.append(n)
        else:
            row[n] = v[k]
    if missing:
        return None, "could not map features: " + ", ".join(missing)
    import pandas as pd
    X = pd.DataFrame([row], columns=MODEL_COLS)
    cls_idx = list(MODEL.classes_).index(1) if (hasattr(MODEL, "classes_") and 1 in MODEL.classes_) else -1
    p = float(MODEL.predict_proba(X)[0][cls_idx]) if hasattr(MODEL, "predict_proba") else float(MODEL.predict(X)[0])
    return clamp(p, 0.0, 1.0), None

def score(v):
    p, note = model_predict(v)
    src = "model"
    if p is None:
        p, src = formula(v), "formula"
    l = level(p)
    return {"probability": round(p, 3), "level": l[1], "action": l[2], "source": src, "note": note, "inputs": v}

def score_batch(items):
    if not items:
        return []
    
    use_model = (MODEL is not None and len(MODEL_COLS) > 0)
    col_mapping = {}
    if use_model:
        for n in MODEL_COLS:
            k = match_feature(n)
            if k is not None:
                col_mapping[n] = k
            else:
                use_model = False
                break
    
    if use_model:
        try:
            import pandas as pd
            rows = []
            for v in items:
                row = {n: v.get(col_mapping[n], 0) for n in MODEL_COLS}
                rows.append(row)
            X = pd.DataFrame(rows, columns=MODEL_COLS)
            cls_idx = list(MODEL.classes_).index(1) if (hasattr(MODEL, "classes_") and 1 in MODEL.classes_) else -1
            if hasattr(MODEL, "predict_proba"):
                probs = MODEL.predict_proba(X)[:, cls_idx]
            else:
                probs = MODEL.predict(X)
            
            results = []
            for i, p in enumerate(probs):
                clamped_p = clamp(float(p), 0.0, 1.0)
                l = level(clamped_p)
                v = items[i]
                res = {"probability": round(clamped_p, 3), "level": l[1], "action": l[2], "source": "model", "note": None, "inputs": v}
                if "id" in v: res["id"] = v["id"]
                results.append(res)
            return results
        except Exception:
            pass

    # Fallback to formula
    results = []
    for v in items:
        p = formula(v)
        l = level(p)
        res = {"probability": round(p, 3), "level": l[1], "action": l[2], "source": "formula", "note": "model fallback", "inputs": v}
        if "id" in v: res["id"] = v["id"]
        results.append(res)
    return results

def river_from(j):
    q = [x for x in ((j.get("daily") or {}).get("river_discharge") or []) if x is not None]
    if len(q) < 6: return None
    med = sorted(q)[len(q) // 2] or .001
    return .5 + 5.5 * clamp((max(q[-3:]) / med - .5) / 3.5, 0, 1)

def num(src, k, d):
    try: return float(src.get(k, d))
    except (TypeError, ValueError): return d

# ---------------- API Endpoints ----------------
@app.route("/api/predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return Response(status=200)
    src = request.get_json(silent=True) or request.args
    v = {k: num(src, k, d) for k, d in
         [("rain", 0), ("f3", 0), ("river", 2), ("soil", .35), ("el", 5), ("dr", 20), ("pop", 0)]}
    return jsonify(score(v))

@app.route("/api/batch-predict", methods=["POST", "OPTIONS"])
def batch_predict():
    if request.method == "OPTIONS":
        return Response(status=200)
    data = request.get_json(silent=True) or {}
    items = data.get("items") or []
    if not isinstance(items, list):
        return jsonify(error="'items' must be a list"), 400
    
    inputs_list = []
    for item in items:
        v = {k: num(item, k, d) for k, d in
             [("rain", 0), ("f3", 0), ("river", 2), ("soil", .35), ("el", 5), ("dr", 20), ("pop", 0)]}
        if "id" in item:
            v["id"] = item["id"]
        inputs_list.append(v)
    
    results = score_batch(inputs_list)
    return jsonify(predictions=results, count=len(results), model_loaded=MODEL is not None)

@app.route("/api/risk", methods=["GET", "OPTIONS"])
def risk_at():
    if request.method == "OPTIONS":
        return Response(status=200)
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
        assert abs(lat) <= 90 and abs(lon) <= 180
    except (KeyError, ValueError, AssertionError):
        return jsonify(error="lat and lon required"), 400
    if not rate_ok(request.remote_addr):
        return jsonify(error="Too many requests"), 429
    lat, lon, off = round(lat, 4), round(lon, 4), .008
    try:
        def weather(h):
            return get_json("weather", "v1/forecast", latitude=lat, longitude=lon,
                            hourly=h, forecast_hours=15, timezone="auto")
        try: w = weather("precipitation,soil_moisture_3_to_9cm")
        except Exception: w = weather("precipitation")
        try: rv = river_from(get_json("flood", "v1/flood", latitude=lat, longitude=lon,
                                      daily="river_discharge", past_days=30, forecast_days=3))
        except Exception: rv = None
        las = [lat, lat + off, lat - off, lat, lat]
        los = [lon, lon, lon, lon + off, lon - off]
        el_res = get_json("weather", "v1/elevation", latitude=",".join(map(str, las)),
                          longitude=",".join(map(str, los)))
        el = el_res.get("elevation", [10, 10, 10, 10, 10])
    except Exception as e:
        return jsonify(error=str(e)), 502
    hp = [x or 0 for x in w["hourly"]["precipitation"]]
    sm = (w["hourly"].get("soil_moisture_3_to_9cm") or [None])[0]
    rel_el = clamp(el[0] - min(el[1:]), 0, 35) if len(el) > 1 else 5
    v = {"rain": hp[0], "f3": sum(hp[1:4]), "river": 2 if rv is None else rv,
         "soil": clamp((sm if sm is not None else .35) / .5, 0, 1),
         "el": rel_el, "dr": 20, "pop": 0}
    out = score(v)
    out["location"] = {"lat": lat, "lon": lon, "ground_elevation_m": el[0]}
    return jsonify(out)

@app.route("/api/model-info", methods=["GET", "OPTIONS"])
def model_info():
    if request.method == "OPTIONS":
        return Response(status=200)
    importances = {}
    if MODEL is not None and hasattr(MODEL, "feature_importances_") and hasattr(MODEL, "feature_names_in_"):
        for name, imp in zip(MODEL.feature_names_in_, MODEL.feature_importances_):
            importances[name] = round(float(imp), 4)
    return jsonify(
        loaded=MODEL is not None,
        error=MODEL_ERR,
        type=type(MODEL).__name__ if MODEL is not None else None,
        features=MODEL_COLS,
        importances=importances,
        classes=list(map(int, getattr(MODEL, "classes_", []))) if MODEL is not None else [],
        mapped={n: match_feature(n) for n in MODEL_COLS}
    )

@app.route("/api/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return Response(status=200)
    return jsonify(
        ok=True,
        cached=len(_cache),
        model=MODEL is not None,
        model_type=type(MODEL).__name__ if MODEL is not None else None,
        features=MODEL_COLS
    )

# Monitored locations dataset
KARNATAKA_LOCATIONS = [
    {"name":"Bellandur","area":"Bengaluru East","region":"Bengaluru","lat":12.9304,"lon":77.6784,"el":2,"dr":16,"pop":90000},
    {"name":"Whitefield","area":"Bengaluru East","region":"Bengaluru","lat":12.9698,"lon":77.7500,"el":5,"dr":20,"pop":150000},
    {"name":"Marathahalli","area":"Bengaluru East","region":"Bengaluru","lat":12.9569,"lon":77.7011,"el":4,"dr":18,"pop":110000},
    {"name":"KR Puram","area":"Bengaluru East","region":"Bengaluru","lat":13.0050,"lon":77.6960,"el":3,"dr":17,"pop":100000},
    {"name":"Silk Board / HSR Layout","area":"Bengaluru South","region":"Bengaluru","lat":12.9177,"lon":77.6238,"el":3,"dr":17,"pop":120000},
    {"name":"Koramangala","area":"Bengaluru South","region":"Bengaluru","lat":12.9352,"lon":77.6245,"el":4,"dr":19,"pop":130000},
    {"name":"Bommanahalli","area":"Bengaluru South","region":"Bengaluru","lat":12.9081,"lon":77.6476,"el":4,"dr":18,"pop":100000},
    {"name":"Jayanagar","area":"Bengaluru South","region":"Bengaluru","lat":12.9250,"lon":77.5938,"el":10,"dr":26,"pop":140000},
    {"name":"Electronic City","area":"Bengaluru South","region":"Bengaluru","lat":12.8452,"lon":77.6602,"el":9,"dr":28,"pop":90000},
    {"name":"Hebbal","area":"Bengaluru North","region":"Bengaluru","lat":13.0358,"lon":77.5970,"el":3,"dr":18,"pop":100000},
    {"name":"Yelahanka","area":"Bengaluru North","region":"Bengaluru","lat":13.1007,"lon":77.5963,"el":8,"dr":25,"pop":120000},
    {"name":"Majestic (City Centre)","area":"Bengaluru Central","region":"Bengaluru","lat":12.9767,"lon":77.5713,"el":6,"dr":22,"pop":160000},
    {"name":"Rajajinagar","area":"Bengaluru West","region":"Bengaluru","lat":12.9916,"lon":77.5520,"el":9,"dr":25,"pop":120000},
    {"name":"Kengeri","area":"Bengaluru West","region":"Bengaluru","lat":12.9170,"lon":77.4830,"el":8,"dr":24,"pop":80000},
    {"name":"Mangaluru","area":"Dakshina Kannada","region":"Coastal","lat":12.9141,"lon":74.8560,"el":3,"dr":20,"pop":480000},
    {"name":"Udupi","area":"Udupi","region":"Coastal","lat":13.3409,"lon":74.7421,"el":3,"dr":20,"pop":170000},
    {"name":"Kundapura","area":"Udupi","region":"Coastal","lat":13.6231,"lon":74.6926,"el":2,"dr":17,"pop":40000},
    {"name":"Bhatkal","area":"Uttara Kannada","region":"Coastal","lat":13.9853,"lon":74.5569,"el":2,"dr":16,"pop":40000},
    {"name":"Karwar","area":"Uttara Kannada","region":"Coastal","lat":14.8137,"lon":74.1290,"el":3,"dr":18,"pop":80000},
    {"name":"Madikeri","area":"Kodagu","region":"Malnad","lat":12.4244,"lon":75.7382,"el":22,"dr":20,"pop":35000},
    {"name":"Sakleshpura","area":"Hassan","region":"Malnad","lat":12.9440,"lon":75.7860,"el":12,"dr":20,"pop":30000},
    {"name":"Chikkamagaluru","area":"Chikkamagaluru","region":"Malnad","lat":13.3161,"lon":75.7720,"el":15,"dr":26,"pop":120000},
    {"name":"Shivamogga","area":"Shivamogga","region":"Malnad","lat":13.9299,"lon":75.5681,"el":6,"dr":25,"pop":320000},
    {"name":"Thirthahalli","area":"Shivamogga","region":"Malnad","lat":13.6890,"lon":75.2410,"el":8,"dr":18,"pop":35000},
    {"name":"Sirsi","area":"Uttara Kannada","region":"Malnad","lat":14.6196,"lon":74.8354,"el":12,"dr":22,"pop":65000},
    {"name":"Mysuru","area":"Mysuru","region":"South Interior","lat":12.2958,"lon":76.6394,"el":8,"dr":28,"pop":920000},
    {"name":"Srirangapatna","area":"Mandya","region":"South Interior","lat":12.4140,"lon":76.6940,"el":2,"dr":16,"pop":25000},
    {"name":"Mandya","area":"Mandya","region":"South Interior","lat":12.5218,"lon":76.8951,"el":4,"dr":22,"pop":150000},
    {"name":"Hassan","area":"Hassan","region":"South Interior","lat":13.0068,"lon":76.1004,"el":8,"dr":24,"pop":160000},
    {"name":"Tumakuru","area":"Tumakuru","region":"South Interior","lat":13.3379,"lon":77.1173,"el":10,"dr":28,"pop":310000},
    {"name":"Kolar","area":"Kolar","region":"South Interior","lat":13.1357,"lon":78.1294,"el":12,"dr":26,"pop":140000},
    {"name":"Chikkaballapur","area":"Chikkaballapur","region":"South Interior","lat":13.4355,"lon":77.7315,"el":14,"dr":28,"pop":70000},
    {"name":"Ramanagara","area":"Ramanagara","region":"South Interior","lat":12.7159,"lon":77.2813,"el":10,"dr":24,"pop":100000},
    {"name":"Chamarajanagar","area":"Chamarajanagar","region":"South Interior","lat":11.9261,"lon":76.9437,"el":8,"dr":22,"pop":70000},
    {"name":"Chitradurga","area":"Chitradurga","region":"South Interior","lat":14.2251,"lon":76.3980,"el":12,"dr":28,"pop":150000},
    {"name":"Davanagere","area":"Davanagere","region":"South Interior","lat":14.4644,"lon":75.9218,"el":8,"dr":26,"pop":435000},
    {"name":"Hubballi","area":"Dharwad","region":"North Karnataka","lat":15.3647,"lon":75.1240,"el":6,"dr":25,"pop":940000},
    {"name":"Dharwad","area":"Dharwad","region":"North Karnataka","lat":15.4589,"lon":75.0078,"el":10,"dr":28,"pop":230000},
    {"name":"Belagavi","area":"Belagavi","region":"North Karnataka","lat":15.8497,"lon":74.4977,"el":6,"dr":24,"pop":490000},
    {"name":"Gadag","area":"Gadag","region":"North Karnataka","lat":15.4166,"lon":75.6290,"el":8,"dr":24,"pop":170000},
    {"name":"Haveri","area":"Haveri","region":"North Karnataka","lat":14.7951,"lon":75.4047,"el":8,"dr":24,"pop":70000},
    {"name":"Bagalkot","area":"Bagalkot","region":"North Karnataka","lat":16.1691,"lon":75.6615,"el":3,"dr":16,"pop":110000},
    {"name":"Vijayapura","area":"Vijayapura","region":"North Karnataka","lat":16.8302,"lon":75.7100,"el":6,"dr":22,"pop":330000},
    {"name":"Kalaburagi","area":"Kalaburagi","region":"North Karnataka","lat":17.3297,"lon":76.8343,"el":5,"dr":22,"pop":540000},
    {"name":"Yadgir","area":"Yadgir","region":"North Karnataka","lat":16.7700,"lon":77.1380,"el":5,"dr":20,"pop":70000},
    {"name":"Raichur","area":"Raichur","region":"North Karnataka","lat":16.2160,"lon":77.3566,"el":3,"dr":18,"pop":230000},
    {"name":"Koppal","area":"Koppal","region":"North Karnataka","lat":15.3459,"lon":76.1548,"el":6,"dr":22,"pop":70000},
    {"name":"Hosapete","area":"Vijayanagara","region":"North Karnataka","lat":15.2689,"lon":76.3909,"el":4,"dr":20,"pop":200000},
    {"name":"Ballari","area":"Ballari","region":"North Karnataka","lat":15.1394,"lon":76.9214,"el":8,"dr":26,"pop":410000},
    {"name":"Bidar","area":"Bidar","region":"North Karnataka","lat":17.9104,"lon":77.5199,"el":8,"dr":24,"pop":220000}
]

@app.route("/api/locations", methods=["GET", "OPTIONS"])
def locations():
    if request.method == "OPTIONS":
        return Response(status=200)
    return jsonify(locations=KARNATAKA_LOCATIONS, total=len(KARNATAKA_LOCATIONS))

# ---------------- frontend: reverse proxy routing patch ----------------
PATCH = """<script>
(function(){
 var base = location.pathname.replace(/[^\\/]*$/, "");
 var M = {
   "https://api.open-meteo.com/": "weather",
   "https://flood-api.open-meteo.com/": "flood",
   "https://photon.komoot.io/": "geo",
   "https://nominatim.openstreetmap.org/": "nominatim",
   "https://www.gdacs.org/": "gdacs"
 };
 var of = window.fetch.bind(window);
 window.fetch = function(u, o) {
   if (typeof u === "string") {
     for (var h in M) {
       if (u.indexOf(h) === 0) {
         u = base + "proxy/" + M[h] + "/" + u.slice(h.length);
         break;
       }
     }
   }
   return of(u, o);
 };
})();
</script>"""

@app.route("/")
def index():
    for folder in (BASE, os.path.join(BASE, "frontend")):
        for name in ("index.html", "eshan.html"):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    html = fh.read()
                html, n = re.subn(r"(<head[^>]*>)", lambda m: m.group(1) + PATCH, html, count=1, flags=re.I)
                return Response(html if n else PATCH + html, content_type="text/html; charset=utf-8")
    return "Put index.html or eshan.html next to this file (or in a 'frontend' folder).", 404

if __name__ == "__main__":
    print(f"FloodSense AI running on http://127.0.0.1:{PORT}")
    print(f"Model status: {'Loaded successfully (' + type(MODEL).__name__ + ')' if MODEL is not None else 'Not loaded: ' + str(MODEL_ERR)}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)