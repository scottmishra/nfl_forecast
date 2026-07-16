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
  let g, sim = null;
  try {
    [g, sim] = await Promise.all([
      api(`/api/game/${encodeURIComponent(gameId)}`),
      api(`/api/game/${encodeURIComponent(gameId)}/sim`).catch(() => null),
    ]);
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
    ${sim ? `
      <nav class="tabs" role="tablist">
        <button class="tab active" data-tab="forecasts">Player Forecasts</button>
        <button class="tab" data-tab="sim">Game Simulation
          <span class="tab-note">${sim.n_sims.toLocaleString()} sims</span></button>
      </nav>` : ""}
    <div id="panel-forecasts" class="tab-panel">
      <div class="teams-cols">
        ${teamColumn(g.away)}
        ${teamColumn(g.home)}
      </div>
    </div>
    ${sim ? `<div id="panel-sim" class="tab-panel" hidden>${simPanel(g, sim)}</div>` : ""}`;

  document.getElementById("back").addEventListener("click", () => { location.hash = ""; });
  view.querySelectorAll(".tab").forEach((btn) =>
    btn.addEventListener("click", () => {
      view.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
      view.querySelectorAll(".tab-panel").forEach((p) => {
        p.hidden = p.id !== `panel-${btn.dataset.tab}`;
      });
    }));
  bindStripTooltips();
}

/* ---------- simulation panel ---------- */

function simPanel(g, sim) {
  const away = g.away.team, home = g.home.team;
  const sa = sim.teams[away.abbr], sh = sim.teams[home.abbr];
  const awayPct = Math.round(sim.away_win_prob * 100);
  const homePct = Math.round(sim.home_win_prob * 100);
  const tiePct = Math.max(0, 100 - awayPct - homePct);
  return `
    <section class="sim-hero">
      <div class="sim-score">
        <span class="sim-score-team" style="color:${esc(away.primary)}">${esc(away.abbr)}</span>
        <span class="sim-score-num">${sa.points.p50}</span>
        <span class="sim-score-dash">–</span>
        <span class="sim-score-num">${sh.points.p50}</span>
        <span class="sim-score-team" style="color:${esc(home.primary)}">${esc(home.abbr)}</span>
      </div>
      <div class="sim-score-sub">projected median score · ${sim.n_sims.toLocaleString()} simulated games</div>
      <div class="winprob" role="img"
           aria-label="Win probability: ${esc(away.abbr)} ${awayPct}%, ${esc(home.abbr)} ${homePct}%">
        <div class="wp-seg" style="width:${awayPct}%;background:${esc(away.primary)}"></div>
        ${tiePct ? `<div class="wp-seg wp-tie" style="width:${tiePct}%"></div>` : ""}
        <div class="wp-seg" style="width:${homePct}%;background:${esc(home.primary)}"></div>
      </div>
      <div class="wp-labels">
        <span><b>${esc(away.abbr)}</b> ${awayPct}% win</span>
        ${tiePct ? `<span class="wp-tie-label">tie ${tiePct}%</span>` : ""}
        <span><b>${esc(home.abbr)}</b> ${homePct}% win</span>
      </div>
      <div class="pts-strips">
        ${[[away, sa], [home, sh]].map(([t, s]) => `
          <div class="pts-row">
            <span class="pts-team">${esc(t.abbr)}</span>
            <div class="qstrip" data-player="${esc(t.name)}" data-stat="Points"
                 data-q="p10:${s.points.p10}|p25:${s.points.p10}|p50:${s.points.p50}|p75:${s.points.p90}|p90:${s.points.p90}"
                 style="--p10:${pctOf(s.points.p10, 55)};--p25:${pctOf(s.points.p10, 55)};--p50:${pctOf(s.points.p50, 55)};--p75:${pctOf(s.points.p90, 55)};--p90:${pctOf(s.points.p90, 55)}">
              <div class="track"></div><div class="band"></div><div class="median"></div>
            </div>
            <span class="pts-range">${s.points.p10}–${s.points.p90}</span>
          </div>`).join("")}
        <div class="pts-scale-note">floor p10 → ceiling p90, shared 0–55 pt scale</div>
      </div>
    </section>
    <div class="teams-cols">
      ${simTeamCol(away, sa)}
      ${simTeamCol(home, sh)}
    </div>`;
}

function pctOf(v, max) { return `${Math.min(100, (v / max) * 100).toFixed(1)}%`; }

function simTeamCol(team, s) {
  const downs = ["1", "2", "3", "4"];
  const outcomeMeta = [
    ["td", "TD"], ["fg", "FG"], ["punt", "Punt"], ["turnover", "TO"], ["downs", "4th ↓"],
  ];
  const passMean = s.pass_plays_mean, runMean = s.run_plays_mean;
  const passPct = (passMean / (passMean + runMean)) * 100;
  return `
    <div>
      <div class="team-col-head" style="--team-color:${esc(team.primary)}">
        ${teamLogo(team)}<h3>${esc(team.name)}</h3>
      </div>

      <div class="sim-card">
        <h4 class="sim-card-title">Play Calling <span class="sim-card-sub">${fmt(s.plays_mean, 0)} plays · ${fmt(s.drives_mean, 0)} drives</span></h4>
        <div class="mix-legend">
          <span><i class="swatch swatch-pass"></i>Pass ${fmt(passMean, 0)}</span>
          <span><i class="swatch swatch-run"></i>Run ${fmt(runMean, 0)}</span>
        </div>
        <div class="mixbar" role="img" aria-label="Pass ${fmt(passMean,0)} plays, run ${fmt(runMean,0)} plays">
          <div class="mix-pass" style="width:${passPct.toFixed(1)}%"></div>
          <div class="mix-run" style="width:${(100 - passPct).toFixed(1)}%"></div>
        </div>
        <div class="downgrid">
          ${downs.map((d) => {
            const r = s.pass_rate_by_down[d];
            return `
              <div class="downcell">
                <div class="down-label">${d}${["st","nd","rd","th"][d-1]} down</div>
                <div class="downbar"><div class="mix-pass" style="width:${r == null ? 0 : (r * 100).toFixed(0)}%"></div></div>
                <div class="down-val">${r == null ? "—" : Math.round(r * 100) + "% pass"}</div>
              </div>`;
          }).join("")}
        </div>
      </div>

      <div class="sim-card">
        <h4 class="sim-card-title">Drive Outcomes</h4>
        <div class="outcome-chips">
          ${outcomeMeta.map(([k, label]) =>
            `<span class="chip">${label} ${Math.round((s.drive_outcomes[k] || 0) * 100)}%</span>`).join("")}
        </div>
      </div>

      <div class="sim-card">
        <h4 class="sim-card-title">Expected Snap Counts</h4>
        ${s.players.map((p) => {
          const maxSnaps = s.plays_mean;
          return `
          <div class="snap-row" data-tip="${esc(p.name)}: ${p.snaps_p10}–${p.snaps_p90} snaps · ${fmt(p.touches_mean)} touches · ${fmt(p.carries_mean)} car · ${fmt(p.targets_mean)} tgt">
            <span class="pos-badge pos-${esc(p.position)}">${esc(p.position)}</span>
            <span class="snap-name">${esc(p.name)}</span>
            <div class="snapbar">
              <div class="snap-fill" style="width:${Math.min(100, (p.snaps_mean / maxSnaps) * 100).toFixed(1)}%"></div>
              <div class="snap-whisker" style="left:${Math.min(100, (p.snaps_p10 / maxSnaps) * 100).toFixed(1)}%;width:${Math.max(0, ((p.snaps_p90 - p.snaps_p10) / maxSnaps) * 100).toFixed(1)}%"></div>
            </div>
            <span class="snap-val">${fmt(p.snaps_mean, 0)} <em>${Math.round(p.snap_share * 100)}%</em></span>
          </div>`;
        }).join("")}
        <div class="pts-scale-note">bar = mean snaps · whisker = p10–p90 · % of team plays</div>
      </div>
    </div>`;
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

  view.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      tooltip.innerHTML = `<div class="tt-title">${esc(el.dataset.tip)}</div>`;
      tooltip.hidden = false;
      const pad = 14;
      let x = e.clientX + pad, y = e.clientY + pad;
      if (x + tooltip.offsetWidth > innerWidth - 8) x = e.clientX - tooltip.offsetWidth - pad;
      if (y + tooltip.offsetHeight > innerHeight - 8) y = e.clientY - tooltip.offsetHeight - pad;
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
