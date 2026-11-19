import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the model once at startup (linear.pkl must sit next to app.py)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Order MUST match the training columns
FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

# Change this to "₹", "€", etc. if your price column uses another currency
CURRENCY = "$"

# ---------------------------------------------------------------------------
# HTML page (kept inside app.py)
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Price Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --teal:#14b8a6; --gold:#fbbf24; --coral:#fb7185; --navy:#0b1d3a;
    --text:#f1f5f9; --muted:#94a3b8;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Outfit',sans-serif;color:var(--text);min-height:100vh;
    display:flex;align-items:center;justify-content:center;padding:28px 16px;
    background:linear-gradient(-45deg,#061226,#0b2a4a,#0f3d3e,#1b2559);
    background-size:400% 400%;animation:bg 18s ease infinite;overflow-x:hidden;
  }
  @keyframes bg{0%{background-position:0 50%}50%{background-position:100% 50%}100%{background-position:0 50%}}

  /* floating houses & glow orbs */
  .float{position:fixed;z-index:0;opacity:.12;font-size:60px;animation:drift 14s ease-in-out infinite;user-select:none}
  .f1{top:8%;left:6%} .f2{top:70%;left:10%;animation-delay:-4s;font-size:80px}
  .f3{top:15%;right:8%;animation-delay:-7s;font-size:70px} .f4{bottom:8%;right:12%;animation-delay:-2s}
  @keyframes drift{0%,100%{transform:translateY(0) rotate(-6deg)}50%{transform:translateY(-35px) rotate(8deg)}}
  .orb{position:fixed;border-radius:50%;filter:blur(80px);z-index:0;opacity:.35;animation:drift 16s ease-in-out infinite}
  .o1{width:300px;height:300px;background:var(--teal);top:-90px;right:-60px}
  .o2{width:260px;height:260px;background:var(--gold);bottom:-80px;left:-60px;animation-delay:-6s}

  .card{
    position:relative;z-index:1;width:100%;max-width:640px;padding:38px 34px;
    background:rgba(255,255,255,.07);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
    border:1px solid rgba(255,255,255,.16);border-radius:28px;
    box-shadow:0 30px 70px rgba(0,0,0,.5);animation:rise .9s cubic-bezier(.2,.8,.2,1) both;
  }
  @keyframes rise{from{opacity:0;transform:translateY(45px) scale(.96)}to{opacity:1;transform:none}}
  .logo{text-align:center;font-size:52px;animation:bob 3s ease-in-out infinite}
  @keyframes bob{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-9px) scale(1.06)}}
  h1{
    text-align:center;font-size:31px;font-weight:800;margin:4px 0;
    background:linear-gradient(90deg,var(--teal),var(--gold),var(--coral));background-size:200% 100%;
    -webkit-background-clip:text;background-clip:text;color:transparent;animation:slide 6s linear infinite;
  }
  @keyframes slide{to{background-position:200% 0}}
  .sub{text-align:center;color:var(--muted);font-size:14px;font-weight:300;margin-bottom:28px}

  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  .field{display:flex;flex-direction:column;gap:7px;animation:fade .6s ease both}
  .field:nth-child(1){animation-delay:.1s}.field:nth-child(2){animation-delay:.15s}
  .field:nth-child(3){animation-delay:.2s}.field:nth-child(4){animation-delay:.25s}
  .field:nth-child(5){animation-delay:.3s}.field:nth-child(6){animation-delay:.35s}
  .field:nth-child(7){animation-delay:.4s}
  @keyframes fade{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:none}}
  .full{grid-column:1/-1}
  label{font-size:12px;letter-spacing:.9px;text-transform:uppercase;color:var(--muted);font-weight:600;display:flex;justify-content:space-between}
  label b{color:var(--gold);font-weight:700}
  input[type=number]{
    width:100%;padding:13px 14px;font:inherit;font-size:15px;color:var(--text);
    background:rgba(255,255,255,.06);border:1.5px solid rgba(255,255,255,.14);
    border-radius:13px;outline:none;transition:all .25s ease;
  }
  input[type=number]:hover{border-color:rgba(255,255,255,.35)}
  input[type=number]:focus{
    border-color:var(--teal);background:rgba(20,184,166,.1);transform:translateY(-2px);
    box-shadow:0 0 0 4px rgba(20,184,166,.2),0 0 24px rgba(20,184,166,.35);
  }
  /* range slider */
  input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:8px;border-radius:8px;
    background:linear-gradient(90deg,var(--teal),var(--gold),var(--coral));outline:none;cursor:pointer}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;border-radius:50%;
    background:#fff;border:4px solid var(--teal);box-shadow:0 0 14px rgba(20,184,166,.8);transition:transform .2s}
  input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.25)}
  input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:#fff;border:4px solid var(--teal)}

  button{
    margin-top:26px;width:100%;padding:16px;border:none;border-radius:15px;cursor:pointer;
    font:inherit;font-size:17px;font-weight:700;color:#06231f;letter-spacing:.6px;
    background:linear-gradient(90deg,var(--teal),var(--gold),var(--teal));background-size:200% 100%;
    box-shadow:0 12px 32px rgba(20,184,166,.45);position:relative;overflow:hidden;
    transition:transform .2s,box-shadow .2s,background-position .6s;
  }
  button:hover{background-position:100% 0;transform:translateY(-3px);box-shadow:0 18px 42px rgba(251,191,36,.5)}
  button:active{transform:scale(.97)}
  button::after{content:"";position:absolute;top:0;left:-120%;width:55%;height:100%;
    background:linear-gradient(120deg,transparent,rgba(255,255,255,.6),transparent);transform:skewX(-20deg)}
  button:hover::after{animation:shine .9s}
  @keyframes shine{to{left:150%}}
  button.loading{pointer-events:none;opacity:.85}
  button.loading::before{content:"";display:inline-block;width:16px;height:16px;margin-right:10px;vertical-align:-2px;
    border:3px solid rgba(6,35,31,.3);border-top-color:#06231f;border-radius:50%;animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .result{
    margin-top:26px;padding:24px 18px;border-radius:18px;text-align:center;display:none;
    background:linear-gradient(135deg,rgba(20,184,166,.2),rgba(251,191,36,.18));
    border:1.5px solid rgba(251,191,36,.6);box-shadow:0 0 40px rgba(251,191,36,.3);
  }
  .result.show{display:block;animation:pop .7s cubic-bezier(.2,1.4,.4,1) both}
  @keyframes pop{from{opacity:0;transform:scale(.6) rotate(-3deg)}to{opacity:1;transform:none}}
  .result small{display:block;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
  .price{font-size:44px;font-weight:800;color:var(--gold);text-shadow:0 0 25px rgba(251,191,36,.6);word-break:break-all}
  .result.err{background:rgba(251,113,133,.15);border-color:var(--coral);box-shadow:0 0 30px rgba(251,113,133,.3);font-size:15px;color:#fecdd3}

  .spark{position:fixed;z-index:5;pointer-events:none;font-size:20px;animation:up 1.6s ease-out forwards}
  @keyframes up{from{opacity:1;transform:translateY(0) scale(.6)}to{opacity:0;transform:translateY(-140px) scale(1.4) rotate(40deg)}}

  footer{text-align:center;margin-top:22px;font-size:11px;color:rgba(148,163,184,.65)}
  @media(max-width:520px){.grid{grid-template-columns:1fr}.card{padding:28px 20px}.price{font-size:34px}}
</style>
</head>
<body>
  <div class="orb o1"></div><div class="orb o2"></div>
  <span class="float f1">🏠</span><span class="float f2">🏡</span><span class="float f3">🏢</span><span class="float f4">🔑</span>

  <main class="card">
    <div class="logo">🏡</div>
    <h1>House Price Predictor</h1>
    <p class="sub">Enter the property details to estimate its market value</p>

    <form id="form" autocomplete="off">
      <div class="grid">
        <div class="field">
          <label for="Square_Footage">Square Footage</label>
          <input type="number" id="Square_Footage" name="Square_Footage" min="1" step="any" placeholder="e.g. 2000" required>
        </div>
        <div class="field">
          <label for="Lot_Size">Lot Size</label>
          <input type="number" id="Lot_Size" name="Lot_Size" min="0" step="any" placeholder="e.g. 0.5" required>
        </div>
        <div class="field">
          <label for="Num_Bedrooms">Bedrooms</label>
          <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" min="0" step="1" placeholder="e.g. 3" required>
        </div>
        <div class="field">
          <label for="Num_Bathrooms">Bathrooms</label>
          <input type="number" id="Num_Bathrooms" name="Num_Bathrooms" min="0" step="1" placeholder="e.g. 2" required>
        </div>
        <div class="field">
          <label for="Year_Built">Year Built</label>
          <input type="number" id="Year_Built" name="Year_Built" min="1800" max="2100" step="1" placeholder="e.g. 2005" required>
        </div>
        <div class="field">
          <label for="Garage_Size">Garage Size</label>
          <input type="number" id="Garage_Size" name="Garage_Size" min="0" step="1" placeholder="e.g. 1" required>
        </div>
        <div class="field full">
          <label for="Neighborhood_Quality">Neighborhood Quality <b id="qv">5</b></label>
          <input type="range" id="Neighborhood_Quality" name="Neighborhood_Quality" min="1" max="10" step="1" value="5">
        </div>
      </div>
      <button type="submit" id="btn">✨ Estimate Price</button>
    </form>

    <div class="result" id="result"></div>
    <footer>Linear Regression · Flask · Render</footer>
  </main>

<script>
  const CURRENCY = "{{ currency }}";
  const form = document.getElementById('form');
  const btn = document.getElementById('btn');
  const box = document.getElementById('result');
  const q = document.getElementById('Neighborhood_Quality');
  q.addEventListener('input', () => document.getElementById('qv').textContent = q.value);

  const fmt = n => CURRENCY + Math.round(n).toLocaleString('en-US');

  function countUp(el, target) {            // animated number counter
    const start = performance.now(), dur = 1400;
    (function tick(now) {
      const p = Math.min((now - start) / dur, 1), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * e);
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }
  function sparkles() {
    const r = box.getBoundingClientRect();
    for (let i = 0; i < 16; i++) {
      const s = document.createElement('span');
      s.className = 'spark'; s.textContent = ['✨','⭐','💫','🏠'][i % 4];
      s.style.left = (r.left + Math.random() * r.width) + 'px';
      s.style.top = (r.top + r.height / 2) + 'px';
      s.style.animationDelay = (Math.random() * .4) + 's';
      document.body.appendChild(s); setTimeout(() => s.remove(), 2400);
    }
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    btn.classList.add('loading'); btn.textContent = 'Calculating...';
    box.className = 'result';
    const payload = Object.fromEntries(new FormData(form).entries());
    try {
      const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      const data = await res.json();
      void box.offsetWidth;
      if (data.ok) {
        box.className = 'result show';
        box.innerHTML = '<small>Estimated Price</small><div class="price" id="p">' + CURRENCY + '0</div>';
        countUp(document.getElementById('p'), data.price);
        setTimeout(sparkles, 300);
      } else {
        box.className = 'result show err'; box.textContent = '⚠️ ' + data.error;
      }
    } catch (err) {
      box.className = 'result show err'; box.textContent = '⚠️ Could not reach the server.';
    }
    btn.classList.remove('loading'); btn.textContent = '✨ Estimate Price';
  });
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE, currency=CURRENCY)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form
        values = [float(data[name]) for name in FEATURES]
        price = float(model.predict(np.array([values]))[0])
        return jsonify(ok=True, price=max(price, 0.0))
    except (KeyError, ValueError) as e:
        return jsonify(ok=False, error=f"Invalid input: {e}"), 400
    except Exception as e:  # pragma: no cover
        return jsonify(ok=False, error=str(e)), 500


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
