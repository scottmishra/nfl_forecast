/* Gameday Edge dashboard: slate -> matchup -> player quantile strips. */

const view = document.getElementById("view");
const tooltip = document.getElementById("tooltip");
const weekPill = document.getElementById("week-pill");

const STAT_LABELS = {
  passing_yards: "Pass Yds", passing_tds: "Pass TD", interceptions: "INT",
  rushing_yards: "Rush Yds", rushing_tds: "Rush TD", receptions: "Rec",
  receiving_yards: "Rec Yds", receiving_tds: "Rec TD", fantasy_points: "Fantasy",
};
const QS = ["p10", "p25", "p50", "p75", "p90"];
const Q_NAMES = { p10: "Floor (p10)", p25: "p25", p50: "Median", p75: "p75", p90: "Ceiling (p90)" };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw Object.assign(new Error(`API ${res.status}`), { status: res.status });
  return res.json();
}

function fmt(x, digits = 1) {
  if (x == null) return "—";
  return Number(x) >= 100 ? Math.round(x).toString() : Number(x).toFixed(digits).replace(/\.0$/, "");
}

function weatherChips(g) {
  const chips = [];
  if (g.roof === "dome" || g.roof === "retractable") {
    chips.push(`<span class="chip">🏟️ ${g.roof === "dome" ? "Dome" : "Roof"}</span>`);
  } else {
    if (g.temp_c != null) {
      const cold = g.temp_c <= 2;
      chips.push(`<span class="chip ${cold ? "cold" : ""}">🌡️ ${fmt(g.temp_c, 0)}°C</span>`);
    }
    if (g.wind_kph != null) {
      const windy = g.wind_kph >= 24;
      chips.push(`<span class="chip ${windy ? "windy" : ""}">💨 ${fmt(g.wind_kph, 0)} km/h</span>`);
    }
  }
  return chips.join("");
}

function teamLogo(team, cls = "") {
  return `<span class="team-logo ${cls}" style="--team-color:${esc(team.primary)}">${esc(team.abbr)}</span>`;
}

/* ---------- slate view ---------- */

async function renderSlate() {
  view.innerHTML = `<div class="loading"><div class="spinner"></div><p>Loading the slate…</p></div>`;
  let data;
  try {
    data = await api("/api/slate");
  } catch (err) {
    return renderError(err);
  }
  const games = data.games || [];
  if (games.length) {
    weekPill.textContent = `Week ${games[0].week} · ${games[0].season}`;
  }
  view.innerHTML = `
    <div class="section-title">This Week's Slate — ${games.length} games</div>
    <div class="slate-grid">
      ${games.map((g, i) => `
        <article class="game-card" data-game="${esc(g.game_id)}"
                 style="--away-color:${esc(g.away.primary)};--home-color:${esc(g.home.primary)};animation-delay:${i * 40}ms">
          <div class="color-rail"></div>
          <div class="matchup-row">
            <div class="team-chip" style="--team-color:${esc(g.away.primary)}">
              ${teamLogo(g.away)}
              <div><div class="team-name">${esc(g.away.name)}</div>
              <div class="team-abbr-sub">AWAY</div></div>
            </div>
            <div class="at-divider">@</div>
            <div class="team-chip" style="--team-color:${esc(g.home.primary)}">
              ${teamLogo(g.home)}
              <div><div class="team-name">${esc(g.home.name)}</div>
              <div class="team-abbr-sub">HOME</div></div>
            </div>
          </div>
          <div class="game-meta">
            <span class="chip">📍 ${esc(g.stadium || g.home.stadium || "")}</span>
            ${weatherChips(g)}
          </div>
          ${g.headliners?.length ? `
            <div class="headliners">
              ${g.headliners.map((h) => `
                <div class="headliner">
                  <b>${esc(h.player_display_name)}</b>
                  <span>${esc(h.position)} · ${esc(h.team)}</span>
                  <span class="hl-pts">${fmt(h.fantasy_points_p50)} proj</span>
                </div>`).join("")}
            </div>` : ""}
        </article>`).join("")}
    </div>`;

  view.querySelectorAll(".game-card").forEach((el) =>
    el.addEventListener("click", () => {
      location.hash = `#game/${el.dataset.game}`;
    }));
}

/* ---------- game detail ---------- */

async function renderGame(gameId) {
  view.innerHTML = `<div class="loading"><div class="spinner"></div><p>Breaking down the matchup…</p></div>`;
  let g;
  try {
    g = await api(`/api/game/${encodeURIComponent(gameId)}`);
  } catch (err) {
    return renderError(err);
  }
  const away = g.away.team, home = g.home.team;
  view.innerHTML = `
    <button class="back-btn" id="back">← Full Slate</button>
    <section class="hero" style="--away-color:${esc(away.primary)};--home-color:${esc(home.primary)}">
      <div class="hero-inner">
        <div class="team-chip">${teamLogo(away)}<div class="team-name">${esc(away.name)}</div></div>
        <div class="hero-center">
          <div class="hero-vs">VS</div>
          <div class="hero-venue">${esc(g.stadium || "")} · ${esc(g.city || "")}<br>${esc(g.gameday || "")} ${esc(g.gametime || "")}</div>
          <div class="hero-weather">${weatherChips(g)}</div>
        </div>
        <div class="team-chip">${teamLogo(home)}<div class="team-name">${esc(home.name)}</div></div>
      </div>
    </section>
    <div class="teams-cols">
      ${teamColumn(g.away)}
      ${teamColumn(g.home)}
    </div>`;

  document.getElementById("back").addEventListener("click", () => { location.hash = ""; });
  bindStripTooltips();
}

function teamColumn(block) {
  const t = block.team;
  return `
    <div>
      <div class="team-col-head" style="--team-color:${esc(t.primary)}">
        ${teamLogo(t)}<h3>${esc(t.name)}</h3>
      </div>
      ${block.players.map(playerCard).join("") || `<p style="color:var(--ink-3)">No forecastable players.</p>`}
    </div>`;
}

function playerCard(p) {
  return `
    <article class="player-card">
      <div class="player-head">
        <span class="pos-badge pos-${esc(p.position)}">${esc(p.position)}</span>
        <span class="player-name">${esc(p.name)}</span>
        <span class="player-sub">${p.is_home ? "vs" : "@"} ${esc(p.opponent)}</span>
      </div>
      ${p.forecasts.map((f) => statRow(p, f)).join("")}
    </article>`;
}

function statRow(p, f) {
  // Scale the strip to the p90 ceiling of this row (own scale per stat).
  const max = Math.max(f.p90 * 1.08, 1e-6);
  const pct = (v) => `${Math.min(100, (v / max) * 100).toFixed(1)}%`;
  const data = QS.map((q) => `${q}:${f[q]}`).join("|");
  return `
    <div class="stat-row">
      <div class="stat-label">${esc(STAT_LABELS[f.stat] || f.stat)}</div>
      <div class="qstrip" data-q="${esc(data)}" data-stat="${esc(STAT_LABELS[f.stat] || f.stat)}"
           data-player="${esc(p.name)}"
           style="--p10:${pct(f.p10)};--p25:${pct(f.p25)};--p50:${pct(f.p50)};--p75:${pct(f.p75)};--p90:${pct(f.p90)}">
        <div class="track"></div>
        <div class="band"></div>
        <div class="median"></div>
      </div>
      <div class="stat-p50">${fmt(f.p50)}</div>
    </div>`;
}

/* ---------- tooltips (hover layer for every strip) ---------- */

function bindStripTooltips() {
  view.querySelectorAll(".qstrip").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      const vals = Object.fromEntries(el.dataset.q.split("|").map((kv) => kv.split(":")));
      tooltip.innerHTML = `
        <div class="tt-title">${esc(el.dataset.player)} — ${esc(el.dataset.stat)}</div>
        <table>${QS.map((q) =>
          `<tr><td>${Q_NAMES[q]}</td><td>${fmt(Number(vals[q]))}</td></tr>`).join("")}
        </table>`;
      tooltip.hidden = false;
      const pad = 14;
      const w = tooltip.offsetWidth, h = tooltip.offsetHeight;
      let x = e.clientX + pad, y = e.clientY + pad;
      if (x + w > innerWidth - 8) x = e.clientX - w - pad;
      if (y + h > innerHeight - 8) y = e.clientY - h - pad;
      tooltip.style.left = `${x}px`;
      tooltip.style.top = `${y}px`;
    });
    el.addEventListener("mouseleave", () => { tooltip.hidden = true; });
  });
}

/* ---------- search ---------- */

const searchInput = document.getElementById("search");
const searchResults = document.getElementById("search-results");
let searchTimer;

searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = searchInput.value.trim();
  if (q.length < 2) { searchResults.hidden = true; return; }
  searchTimer = setTimeout(async () => {
    try {
      const data = await api(`/api/players?q=${encodeURIComponent(q)}`);
      const players = (data.players || []).slice(0, 12);
      searchResults.innerHTML = players.length
        ? players.map((p) => {
            const fp = p.forecasts.find((f) => f.stat === "fantasy_points");
            return `
              <div class="search-row" data-team="${esc(p.team)}" data-opp="${esc(p.opponent)}">
                <span class="pos-badge pos-${esc(p.position)}">${esc(p.position)}</span>
                <span class="sr-name">${esc(p.name)}</span>
                <span class="sr-meta">${esc(p.team)} · ${fp ? fmt(fp.p50) + " proj" : ""}</span>
              </div>`;
          }).join("")
        : `<div class="search-row"><span class="sr-meta">No matches</span></div>`;
      searchResults.hidden = false;
      searchResults.querySelectorAll(".search-row[data-team]").forEach((row) =>
        row.addEventListener("click", async () => {
          searchResults.hidden = true;
          searchInput.value = "";
          // jump to the game containing this player's team
          const slate = await api("/api/slate");
          const g = slate.games.find((x) =>
            x.home.abbr === row.dataset.team || x.away.abbr === row.dataset.team);
          if (g) location.hash = `#game/${g.game_id}`;
        }));
    } catch { searchResults.hidden = true; }
  }, 180);
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) searchResults.hidden = true;
});

/* ---------- errors & routing ---------- */

function renderError(err) {
  const setup = err.status === 503;
  view.innerHTML = `
    <div class="error-box">
      <h2 style="margin-bottom:12px">${setup ? "No forecasts yet" : "Something went sideways"}</h2>
      ${setup
        ? `<p>Generate a slate first:</p>
           <p style="margin-top:10px"><code>gameday demo</code> &nbsp;(offline synthetic league)<br>
           or <code>gameday forecast</code> &nbsp;(real nflverse data)</p>
           <p style="margin-top:10px">then refresh this page.</p>`
        : `<p>${esc(err.message)}</p>`}
    </div>`;
}

function route() {
  const m = location.hash.match(/^#game\/(.+)$/);
  if (m) renderGame(decodeURIComponent(m[1]));
  else renderSlate();
}

document.getElementById("brand-link").addEventListener("click", (e) => {
  e.preventDefault();
  location.hash = "";
});

window.addEventListener("hashchange", route);
route();
