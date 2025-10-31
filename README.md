# crop-risk-by-region
Crop Risk by Area is a reproducible repo for crop risk–return analytics across regions. Using historical price, yield and weather, it fits classical time-series (SARIMA/ETS) and quantile models, then runs Monte Carlo to estimate E(π), VaR/CVaR and downside odds. Outputs: forecasts, risk reports and cross-area comparisons.

<h2> Project Overview</h2>
<p>
  <strong>Crop Risk by Area</strong> estimates and compares risk–return profiles of field crops
  (e.g., wheat, barley, corn) across geographic units (province/district/plain). Using historical
  <strong>price</strong>, <strong>yield</strong>, and <strong>weather</strong> series, it combines
  <strong>classical time-series models</strong> (seasonal naïve, SARIMA/ETS) and
  <strong>conditional quantile learners</strong> with <strong>Monte Carlo simulation</strong> to produce
  actionable metrics: expected profit E(π), <strong>VaR/CVaR</strong>, <strong>downside probability</strong>,
  and <strong>scenario-based</strong> recommendations.
</p>

<p><strong>Why it matters.</strong> Farmers and policymakers face joint price–yield volatility, input-cost shocks, and climate variability. Sound decisions should weigh upside potential <em>and</em> tail risk, not averages alone.</p>

<h3> Scope (What you get)</h3>
<ul>
  <li><strong>Per-area, per-crop forecasting</strong> with exogenous weather features</li>
  <li><strong>Rolling-origin evaluation</strong> (MAE/RMSE/sMAPE) + quantile coverage diagnostics</li>
  <li><strong>Risk reporting</strong> &amp; cross-area/cross-crop comparisons</li>
  <li><em>(Optional)</em> <strong>Portfolio view</strong>: land allocation under a risk limit</li>
</ul>

<h3> What this repo provides</h3>
<ul>
  <li>Modular <strong>data pipeline</strong> (market, official stats, meteorology), cleaning &amp; feature engineering</li>
  <li><strong>Config-driven</strong> training/evaluation (<code>configs/area.yaml</code>)</li>
  <li>Reproducible <strong>notebooks &amp; scripts</strong> for EDA, modeling, risk reporting</li>
  <li>Clear <strong>artifacts</strong>: figures, tables, forecasts, and risk summaries</li>
</ul>

<h3> Design principles</h3>
<p>Transparent assumptions, interpretable methods (no deep learning required), lean dependencies aligned with <strong>DSA210</strong>.</p>

<h3> Primary users</h3>
<p>Farmers, cooperatives, analysts, and local authorities needing evidence-based crop selection under explicit risk constraints.</p>

<h3> Out of scope (for now)</h3>
<p>High-frequency trading signals, large-scale satellite ingestion, or black-box models without interpretability.</p>

<h2>🎯 Motivation</h2>
<p>
  Farmers and policymakers must decide <em>what to plant, where, and under which risk profile</em> while facing joint
  <strong>price–yield volatility</strong>, rising and volatile input costs, and increasing climate variability. Traditional
  workflows lean on averages or point forecasts (mean yield or price), which can mask tail risks and lead to fragile
  cash flows. Decisions should instead be framed in terms of <strong>risk–return</strong>: expected profit versus the
  probability and severity of adverse outcomes.
</p>

<ul>
  <li><strong>Reality of the problem:</strong> Prices and yields co-vary across areas and seasons; downside episodes
      (droughts, pest outbreaks, market shocks) drive losses, not the average year.</li>
  <li><strong>Gap in practice:</strong> Area- and crop-specific risk is rarely quantified with <em>calibrated</em>
      uncertainty bands and tail metrics (VaR/CVaR), and sensitivity to input-cost shocks is often ignored.</li>
  <li><strong>Why this repo:</strong> Provide a transparent, reproducible pipeline that couples
      <strong>classical time-series</strong> (seasonal naive, SARIMA/ETS) and <strong>conditional quantile</strong> models
      with <strong>Monte Carlo</strong> to measure expected profit E(π), <strong>VaR/CVaR</strong>, and
      <strong>downside probabilities</strong> per crop and area.</li>
  <li><strong>Design principles:</strong> Interpretable methods aligned with DSA210 (no deep learning required), clear
      assumptions, config-driven experiments, and lean dependencies.</li>
</ul>

<p>
  <strong>Impact.</strong> The outputs inform crop choice, land allocation, and risk limits at the
  <em>province/district/plain</em> level, enabling stakeholders—farmers, cooperatives, and local authorities—to move from
  intuition and averages to <strong>evidence-based, risk-aware decisions</strong>.
</p>

<h2>🔎 Research Questions</h2>

<ol>

  <li>
    <strong>RQ1 — Decision:</strong> For each area, which crop maximizes expected profit <em>E(&pi;)</em> under a tail-risk cap <em>CVaR<sub>0.95</sub> ≤ κ</em>?
    <ul>
      <li><em>Metric:</em> E(&pi;), VaR/CVaR at 95%, downside probability P(&pi;&lt;0)</li>
      <li><em>Output:</em> Area→crop ranking + recommended pick at user-set κ</li>
    </ul>
  </li>

  <li>
    <strong>RQ2 — Forecasting value of exogenous data:</strong> Do weather features (temperature/precipitation indices) improve accuracy over univariate baselines?
    <ul>
      <li><em>Metric:</em> sMAPE / RMSE improvement vs. seasonal naïve & SARIMA/ETS</li>
      <li><em>Target:</em> ≥10% relative error reduction (area-wise)</li>
    </ul>
  </li>

  <li>
    <strong>RQ3 — Risk quantification:</strong> What are <em>VaR<sub>0.95</sub></em>, <em>CVaR<sub>0.95</sub></em> and <em>P(&pi;&lt;0)</em> per crop–area for the next season?
    <ul>
      <li><em>Metric:</em> Tail risk levels from Monte Carlo over price &amp; yield distributions</li>
      <li><em>Output:</em> Risk table + scenario bands (P10/P50/P90)</li>
    </ul>
  </li>

  <li>
    <strong>RQ4 — Sensitivity &amp; robustness:</strong> How do ±10% input-cost shocks (e.g., fertilizer, fuel) change E(&pi;), rankings, and risk?
    <ul>
      <li><em>Metric:</em> &Delta;E(&pi;), &Delta;CVaR, rank shifts</li>
      <li><em>Output:</em> Tornado/sensitivity charts and recommendation deltas</li>
    </ul>
  </li>

  <li>
    <strong>RQ5 — Stability over time:</strong> Are area-level recommendations stable across rolling windows?
    <ul>
      <li><em>Metric:</em> Rank-correlation (Kendall &tau;) between consecutive windows</li>
      <li><em>Target:</em> &tau; ≥ 0.6 for “stable” badges</li>
    </ul>
  </li>

  <li>
    <strong>RQ6 — Portfolio (optional):</strong> Given total land and a risk limit, what crop mix maximizes E(&pi;) subject to <em>CVaR<sub>0.95</sub> ≤ κ</em>?
    <ul>
      <li><em>Metric:</em> Portfolio E(&pi;), CVaR, downside coverage</li>
      <li><em>Output:</em> Area→{wheat, barley, corn} allocation vector (%)</li>
    </ul>
  </li>

  <li>
    <strong>RQ7 — Uncertainty calibration:</strong> Are prediction intervals well-calibrated?
    <ul>
      <li><em>Metric:</em> Empirical coverage of 90% bands (target 90% ± tolerance), PIT uniformity check</li>
      <li><em>Output:</em> Calibration plots and coverage tables</li>
    </ul>
  </li>

</ol>

<h2>📚 Data Sources</h2>

<h3>📈 Price — TÜRİB (Primary)</h3>
<ul>
  <li><strong>What:</strong> Daily transaction prices for grains</li>
  <li><strong>Unit:</strong> TRY/ton (stored as <code>real_price_try_per_ton</code>)</li>
  <li><strong>Frequency:</strong> Daily → aggregated to <em>monthly median</em></li>
  <li><strong>Pipeline use:</strong> Deflate with Agri-PPI (Tarım-ÜFE) to real terms; feed TS models &amp; risk module</li>
</ul>

<h3>🌾 Yield &amp; Area — TÜİK (Crop Production)</h3>
<ul>
  <li><strong>What:</strong> Annual yield and planted/harvested area by crop &amp; geography</li>
  <li><strong>Units:</strong> Yield <em>kg/da</em>, Area <em>da</em>, Production <em>ton</em></li>
  <li><strong>Stored as:</strong> Yield in <em>t/ha</em> via <code>t/ha = kg/da × 0.01</code>; keys = <code>(area_id, crop, year)</code></li>
  <li><strong>Pipeline use:</strong> Target for TS &amp; quantile models; combined with price for profit/risk</li>
</ul>

<h3>🌦️ Weather &amp; Climate — ERA5-Land + CHIRPS</h3>
<ul>
  <li><strong>Primary stack:</strong> <u>ERA5-Land</u> (Tmin/Tmax/Tmean, PET) + <u>CHIRPS</u> (precipitation)</li>
  <li><strong>Derived indices:</strong> SPI-3/6 (from CHIRPS), optional SPEI-3/6 (CHIRPS + PET), GDD, extreme-heat days, frost metrics</li>
  <li><strong>Optional:</strong> MGM station data for <em>bias correction only</em>; ERA5 soil moisture (swvl) for anomalies</li>
  <li><strong>Rule:</strong> Use a single precipitation source for features (prefer CHIRPS); do <em>not</em> include ERA5 &amp; CHIRPS precip simultaneously</li>
  <li><strong>Pipeline use:</strong> Exogenous features for SARIMAX/ETSX &amp; quantile models; hazard summaries</li>
</ul>

<h3>💱 Deflators — Tarım-ÜFE (Agri-PPI)</h3>
<ul>
  <li><strong>What:</strong> Monthly agricultural producer price index (base-year normalized, e.g., 2020=100)</li>
  <li><strong>Use:</strong> Deflation to convert nominal TÜRİB prices to <em>real TRY/ton</em></li>
  <li><strong>Formula:</strong> <code>real_price_t = nominal_price_t × (Index_base / Index_t)</code></li>
</ul>

<h3>🧾 Costs — Local/Regional Bulletins + TGFE</h3>
<ul>
  <li><strong>Bulletins:</strong> Harvest tariffs (combine), transport tariffs, drying fees (silos), seed/fertilizer lists</li>
  <li><strong>Normalization:</strong> Convert to <em>TRY/ha</em> (per-ha items) and <em>TRY/t</em> (per-ton items); record date/region/VAT</li>
  <li><strong>Projection:</strong> Update base-year items with <u>TGFE</u> sub-indices (e.g., seed, fertilizer, pesticides, energy, services)</li>
  <li><strong>Pipeline use:</strong> Nominal cost scenarios (Base/P10/P90 YoY); combined with nominal revenue for profit</li>
</ul>
<h2>🧪 Methodology — P-Y-C Bands & Monte Carlo</h2>

<p><strong>P-Y-C</strong> = <u>Price</u> (P), <u>Yield</u> (Y), <u>Cost</u> (C). For each area–crop pair, we build
probabilistic bands for P, Y, and C and use <em>Monte Carlo</em> to simulate the profit distribution.</p>

<h3>1) Definitions (per hectare)</h3>
<ul>
  <li><strong>P (TRY/t):</strong> TÜRİB price model (deflated → real), with quantile bands (e.g., P10/P50/P90 or full 0.05–0.95).</li>
  <li><strong>Y (t/ha):</strong> TurkStat yield modeled via TS/quantile learners to obtain bands.</li>
  <li><strong>C (TRY/ha):</strong> Crop-specific basket:
    <ul>
      <li><em>per-ha</em> items: seed, fertilizer, pesticide, fuel/labor…</li>
      <li><em>per-ton</em> items: harvesting, transport, <u>corn: drying</u>.</li>
      <li>Nominal projection: scale base-year items with TGFE sub-indices to get <code>C_ha</code> and <code>C_ton</code>.</li>
    </ul>
  </li>
</ul>

<h3>2) Band construction</h3>
<ul>
  <li><strong>P-band:</strong> Seasonal naïve / SARIMA/ETS (+ optional exogenous) → forecast + quantiles.</li>
  <li><strong>Y-band:</strong> SARIMA/ETS/Quantile ML → quantiles.</li>
  <li><strong>C-band:</strong> Sample TGFE YoY (Base/P10/P90 or empirical) to form scenarios for <code>C_ha</code> and <code>C_ton</code>.</li>
</ul>

<h3>3) Monte Carlo (profit simulation)</h3>
<ol>
  <li>For each draw, sample <code>P ~ band_P</code>, <code>Y ~ band_Y</code>, and <code>C_ha, C_ton ~ band_C</code>
      <em>(default: independence; option: rank-copula to impose ρ<sub>P,Y</sub>)</em>.</li>
  <li><strong>Per-ha profit:</strong>
    <pre><code>π = P · Y − [ C_ha + (Y · C_ton) ]</code></pre>
  </li>
  <li>Run N=10,000 draws; collect the simulated profit distribution.</li>
</ol>

<h3>4) Metrics</h3>
<ul>
  <li><strong>E[π]</strong> — expected profit (per ha; for total area A: <code>π_total = A · π</code>).</li>
  <li><strong>VaR<sub>0.95</sub>(π)</strong> — 5% worst-case threshold of profit.</li>
  <li><strong>CVaR<sub>0.95</sub>(π)</strong> — average profit within the worst 5% tail.<br>
      <em>Downside convention:</em> <code>CVaR_down = | E[ π | π ≤ VaR_0.95 ] |</code></li>
</ul>

<h3>5) Risk factor (decision score)</h3>
<p>
  <strong>RiskFactor</strong> = <code>E(π) / CVaR_down</code> &nbsp; (larger ⇒ better).<br>
  Alternative/constraint form: <code>maximize E(π) subject to CVaR_down ≤ κ</code>.
</p>

<h3>6) Reporting</h3>
<ul>
  <li>Area → crop ranking (E[π], VaR/CVaR, RiskFactor).</li>
  <li>P-Y-C band visuals and tornado/sensitivity charts (cost shocks).</li>
  <li>Optional: portfolio land allocation under a CVaR cap.</li>
</ul>

<h3>🧱 Units & Numéraire</h3>
<p>All revenue and costs are expressed per hectare (TRY/ha):</p>
<pre><code>Revenue/ha = P[TRY/t] × Y[t/ha]
Cost/ha    = C_ha[TRY/ha] + Y[t/ha] × C_ton[TRY/t]
Profit/ha  = Revenue/ha − Cost/ha</code></pre>
<ul>
  <li>Use a single numéraire (all-real via Agri-PPI, or all-nominal).</li>
  <li>Convert kg/da → t/ha (×0.01), da → ha (÷10) before modeling.</li>
</ul>

<pre><code>Yield (Y): t/ha         ← from kg/da × 0.01
Price (P): real TRY/t    ← deflated by Agri-PPI (base year)
Costs:     C_ha (TRY/ha), C_ton (TRY/t)

Revenue/ha = P × Y
Cost/ha    = C_ha + Y × C_ton
Profit/ha  = (P × Y) − (C_ha + Y × C_ton)
Total      = Area[ha] × Profit/ha
</code></pre>
<p><em>Rule:</em> Use one numéraire (all-real or all-nominal) and the same deflator for consistency.</p>

<h2>🤖 AI Use & Disclosure</h2>
<p>
  I used <strong>ChatGPT and Claude</strong> as an assistive tool. All code, data pulls, and final decisions were implemented and verified by the project owner; AI outputs were treated as suggestions and edited for accuracy and syllabus alignment.
</p>

<h3>1) README HTML conversion & clarifications</h3>
<ul>
  <li>Converted the README structure to <strong>HTML</strong> (headings, lists, tables, code blocks) for a clean preview.</li>
  <li>Helped draft/clarify text in <em>Project Overview</em>, <em>Motivation</em>, <em>Research Questions</em>, and <em>Methodology (P-Y-C bands)</em>.</li>
  <li>Ensured <strong>units/numéraire</strong> consistency (TRY/ha; <code>t/ha = kg/da × 0.01</code>; real vs nominal prices).</li>
</ul>

<h3>2) Risk-factor design & implementation tips</h3>
<ul>
  <li>Primary metric kept in <strong>money terms</strong>: <code>π = P·Y − [C_ha + Y·C_ton]</code> (TRY/ha).</li>
  <li>Monte Carlo with quantile bands for <strong>P</strong> (price), <strong>Y</strong> (yield), <strong>C</strong> (cost) → estimate <strong>E(π)</strong>, <strong>VaR</strong>, <strong>CVaR</strong>.</li>
  <li>Decision score: <strong>RiskFactor = E(π) / CVaR<sub>down</sub></strong>; constraint form: <code>maximize E(π)</code> s.t. <code>CVaR<sub>down</sub> ≤ κ</code>.</li>
  <li>Technical guardrails: use a single numéraire; avoid dual precip sources; rolling-origin validation; optional rank-copula if P–Y dependence is needed.</li>
</ul>

<h3>3) Data-source recommendations (chosen defaults)</h3>
<ul>
  <li><strong>Price:</strong> <u>TÜRİB</u> daily → monthly median → deflate with <u>Agri-PPI (Tarım-ÜFE)</u> to real TRY/t.</li>
  <li><strong>Yield &amp; Area:</strong> <u>TÜİK</u> Crop Production (store yield as t/ha; keys = (area_id, crop, year)).</li>
  <li><strong>Weather:</strong> <u>ERA5-Land</u> (Tmin/Tmax/Tmean, PET) + <u>CHIRPS</u> (precip). Compute <u>SPI-3/6</u> from CHIRPS; optional <u>SPEI-3/6</u> from CHIRPS+PET; MGM only for bias correction.</li>
  <li><strong>Costs:</strong> Local bulletins (harvest, transport, drying, seed/fertilizer) normalized to TRY/ha &amp; TRY/t; project with <u>TGFE</u> sub-indices; scenarios via TGFE YoY (P10/P50/P90).</li>
</ul>

<h3>4) Minor out-of-syllabus assistance (limited, optional)</h3>
<ul>
  <li>Light suggestions on <em>interval calibration</em> (coverage/PIT), <em>bias correction</em> (quantile mapping), and <em>dependence modeling</em> (rank-copula) — clearly marked as optional.</li>
  <li>No deep-learning pipelines were used; core methods remain <strong>SARIMA/ETS</strong> and <strong>quantile tree/GBM</strong> in line with DSA210.</li>
</ul>

<h4>Integrity &amp; limitations</h4>
<ul>
  <li>AI suggestions are <strong>non-authoritative</strong>; all figures and numbers are produced by our code/notebooks with fixed seeds.</li>
  <li>No confidential data were uploaded; sources and assumptions are documented in the README/configs.</li>
</ul>










