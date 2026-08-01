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

function renderMetaLine(meta) {
  const line = document.getElementById("meta-line");
  if (!line) return;
  const bits = [];
  if (meta?.week != null) bits.push(`Week ${meta.week} ${meta.season}`);
  if (meta?.engine) bits.push(meta.model_version ? `${meta.engine} ${meta.model_version}` : meta.engine);
  if (meta?.generated_at) bits.push(`refreshed ${String(meta.generated_at).replace("T", " ").slice(0, 16)} UTC`);
  line.textContent = bits.join(" · ");
  line.hidden = !bits.length;
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
  renderMetaLine(data.meta);
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
        ${teamColumn(g.away, true)}
        ${teamColumn(g.home, true)}
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
  bindUsageSparks();
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

function teamColumn(block, withUsage = false) {
  const t = block.team;
  return `
    <div>
      <div class="team-col-head" style="--team-color:${esc(t.primary)}">
        ${teamLogo(t)}<h3>${esc(t.name)}</h3>
      </div>
      ${block.players.map((p) => playerCard(p, withUsage)).join("")
        || `<p style="color:var(--ink-3)">No forecastable players.</p>`}
    </div>`;
}

function playerCard(p, withUsage = false) {
  const tag = p.is_rookie ? `<span class="ptag ptag-rookie">Rookie</span>`
    : p.is_new_team ? `<span class="ptag ptag-new">New team</span>` : "";
  return `
    <article class="player-card" data-player-id="${esc(p.player_id)}" data-position="${esc(p.position)}">
      <div class="player-head">
        <span class="pos-badge pos-${esc(p.position)}">${esc(p.position)}</span>
        <span class="player-name">${esc(p.name)}</span>
        ${tag}
        <span class="player-sub">${p.is_home ? "vs" : "@"} ${esc(p.opponent)}</span>
        ${withUsage ? `<button class="usage-btn" title="Usage trend" aria-label="Usage trend for ${esc(p.name)}">📈</button>` : ""}
      </div>
      ${p.forecasts.map((f) => statRow(p, f)).join("")}
      ${withUsage ? `<div class="usage-box" hidden></div>` : ""}
    </article>`;
}

/* ---------- usage sparkline (fetch on expand; hidden quietly on 404) ---------- */

const USAGE_METRIC = { QB: "attempts", RB: "carries", WR: "targets", TE: "targets" };
const USAGE_LABELS = { attempts: "pass att", carries: "carries", targets: "targets" };

function sparklineSVG(values, w = 130, h = 30) {
  if (values.length < 2) return "";
  const max = Math.max(...values), min = Math.min(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = 2 + (i / (values.length - 1)) * (w - 4);
    const y = h - 3 - ((v - min) / span) * (h - 8);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const [lx, ly] = pts[pts.length - 1].split(",");
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">
      <polyline points="${pts.join(" ")}" fill="none" stroke="currentColor"
                stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="${lx}" cy="${ly}" r="2.4" fill="currentColor"/>
    </svg>`;
}

function bindUsageSparks() {
  view.querySelectorAll(".player-card .usage-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const card = btn.closest(".player-card");
      const box = card.querySelector(".usage-box");
      if (!box) return;
      if (!box.hidden) { box.hidden = true; return; }
      box.hidden = false;
      if (box.dataset.loaded) return;
      box.innerHTML = `<span>loading usage…</span>`;
      try {
        const data = await api(`/api/player/${encodeURIComponent(card.dataset.playerId)}/usage`);
        box.dataset.loaded = "1";
        const metric = USAGE_METRIC[card.dataset.position] || "targets";
        const rows = (data.weeks || []).filter((r) => r[metric] != null);
        if (rows.length < 2) {
          box.innerHTML = `<span>not enough usage history</span>`;
          return;
        }
        const vals = rows.map((r) => Number(r[metric]));
        const last = rows[rows.length - 1];
        box.innerHTML = `
          ${sparklineSVG(vals)}
          <span>${esc(USAGE_LABELS[metric] || metric)} · last ${rows.length} wks ·
            latest ${fmt(vals[vals.length - 1], 0)} (wk ${esc(last.week)})</span>`;
      } catch {
        // no artifact / no rows for this player — hide the affordance quietly
        box.hidden = true;
        btn.remove();
      }
    });
  });
}

function statRow(p, f) {
  // Scale the strip to the p90 ceiling of this row (own scale per stat).
  // In replay mode f.actual is present — widen the scale so the marker fits.
  const hasActual = f.actual != null;
  const max = Math.max(f.p90 * 1.08, hasActual ? f.actual * 1.08 : 0, 1e-6);
  const pct = (v) => `${Math.min(100, (v / max) * 100).toFixed(1)}%`;
  const data = QS.map((q) => `${q}:${f[q]}`).join("|");
  const actualMark = hasActual
    ? `<div class="actual" style="left:${pct(f.actual)}"></div>` : "";
  const actualAttr = hasActual ? ` data-actual="${f.actual}"` : "";
  const p50cell = hasActual
    ? `<div class="stat-p50">${fmt(f.p50)}<span class="sp-act">${fmt(f.actual)}</span></div>`
    : `<div class="stat-p50">${fmt(f.p50)}</div>`;
  return `
    <div class="stat-row">
      <div class="stat-label">${esc(STAT_LABELS[f.stat] || f.stat)}</div>
      <div class="qstrip${hasActual ? " has-actual" : ""}" data-q="${esc(data)}"
           data-stat="${esc(STAT_LABELS[f.stat] || f.stat)}" data-player="${esc(p.name)}"${actualAttr}
           style="--p10:${pct(f.p10)};--p25:${pct(f.p25)};--p50:${pct(f.p50)};--p75:${pct(f.p75)};--p90:${pct(f.p90)}">
        <div class="track"></div>
        <div class="band"></div>
        <div class="median"></div>
        ${actualMark}
      </div>
      ${p50cell}
    </div>`;
}

/* ---------- tooltips (hover layer for every strip) ---------- */

function bindStripTooltips() {
  view.querySelectorAll(".qstrip").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      const vals = Object.fromEntries(el.dataset.q.split("|").map((kv) => kv.split(":")));
      const actualRow = el.dataset.actual != null
        ? `<tr class="tt-actual"><td>Actual</td><td>${fmt(Number(el.dataset.actual))}</td></tr>` : "";
      tooltip.innerHTML = `
        <div class="tt-title">${esc(el.dataset.player)} — ${esc(el.dataset.stat)}</div>
        <table>${QS.map((q) =>
          `<tr><td>${Q_NAMES[q]}</td><td>${fmt(Number(vals[q]))}</td></tr>`).join("")}
        ${actualRow}</table>`;
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

/* ---------- historical replay ---------- */

function skillPct(m) {
  if (!m || m.skill_vs_naive == null) return "—";
  return `${m.skill_vs_naive >= 0 ? "+" : ""}${(m.skill_vs_naive * 100).toFixed(1)}%`;
}

function scorecardPanel(sc) {
  const adj = sc.adjusted, base = sc.baseline || null;
  const rows = [];
  const push = (label, a, b) => {
    const delta = (a && b && a.skill_vs_naive != null && b.skill_vs_naive != null)
      ? a.skill_vs_naive - b.skill_vs_naive : null;
    rows.push(`
      <tr>
        <td class="sc-seg">${esc(label)}</td>
        <td class="sc-n">${a ? a.n : "—"}</td>
        <td class="sc-mae">${a ? fmt(a.mae_p50) : "—"}</td>
        <td class="sc-skill ${a && a.skill_vs_naive >= 0 ? "pos" : "neg"}">${skillPct(a)}</td>
        ${base ? `<td class="sc-base">${skillPct(b)}</td>
        <td class="sc-delta ${delta > 0 ? "pos" : delta < 0 ? "neg" : ""}">${
          delta != null ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%` : "—"}</td>` : ""}
      </tr>`);
  };
  push("Overall", adj.overall, base && base.overall);
  for (const [gkey, gname] of [["phase", "Season phase"], ["experience", "Experience"], ["team", "Team"]]) {
    const aseg = adj.segments[gkey] || {}, bseg = base ? (base.segments[gkey] || {}) : {};
    rows.push(`<tr class="sc-grouphead"><td colspan="${base ? 6 : 4}">${gname}</td></tr>`);
    for (const label of Object.keys(aseg)) push(label, aseg[label], bseg[label]);
  }
  return `
    <section class="scorecard">
      <div class="sc-title">Season-to-Season Scorecard
        <span class="sc-sub">fantasy points · skill vs naive trailing-8${base ? " · adjusted vs no-adjustments" : ""}</span>
      </div>
      <div class="sc-table-wrap"><table class="sc-table">
        <thead><tr><th>Segment</th><th>n</th><th>MAE</th><th>Skill</th>${
          base ? "<th>Base</th><th>Δ</th>" : ""}</tr></thead>
        <tbody>${rows.join("")}</tbody>
      </table></div>
    </section>`;
}

function replayGameBlock(g) {
  const away = g.away, home = g.home;
  return `
    <section class="replay-game">
      <div class="replay-game-head">${away ? esc(away.team.abbr) : "?"} <span>@</span> ${home ? esc(home.team.abbr) : "?"}</div>
      <div class="teams-cols">
        ${away ? teamColumn(away) : ""}
        ${home ? teamColumn(home) : ""}
      </div>
    </section>`;
}

function replaySetup() {
  view.innerHTML = `
    <div class="error-box">
      <h2 style="margin-bottom:12px">No replay data yet</h2>
      <p>Generate a historical replay first:</p>
      <p style="margin-top:10px"><code>gameday replay --season 2024 --compare</code></p>
      <p style="margin-top:10px">then refresh this page.</p>
    </div>`;
}

async function renderReplay(season, week) {
  view.innerHTML = `<div class="loading"><div class="spinner"></div><p>Loading the replay…</p></div>`;
  let seasonsData;
  try {
    seasonsData = await api("/api/replay/seasons");
  } catch (err) {
    return renderError(err);
  }
  const seasons = seasonsData.seasons || [];
  if (!seasons.length) return replaySetup();

  season = seasons.some((s) => s.season === Number(season)) ? Number(season) : seasons[0].season;
  const weeks = seasons.find((s) => s.season === season).weeks;
  week = weeks.includes(Number(week)) ? Number(week) : weeks[0];

  let scorecard, weekData;
  try {
    [scorecard, weekData] = await Promise.all([
      api(`/api/replay/${season}/scorecard`).catch(() => null),
      api(`/api/replay/${season}/${week}`),
    ]);
  } catch (err) {
    return renderError(err);
  }
  weekPill.textContent = `Replay · ${season} · Wk ${week}`;

  view.innerHTML = `
    <div class="replay-bar">
      <div class="section-title" style="margin:0">Historical Replay</div>
      <div class="picker-group">
        <label>Season
          <select id="rp-season">${seasons.map((s) =>
            `<option value="${s.season}" ${s.season === season ? "selected" : ""}>${s.season}</option>`).join("")}</select>
        </label>
        <label>Week
          <select id="rp-week">${weeks.map((w) =>
            `<option value="${w}" ${w === week ? "selected" : ""}>Week ${w}</option>`).join("")}</select>
        </label>
      </div>
    </div>
    ${scorecard ? scorecardPanel(scorecard) : ""}
    <div class="section-title">Week ${week} — forecast vs actual · ${weekData.games.length} games</div>
    <div class="replay-games">${weekData.games.map(replayGameBlock).join("")}</div>`;

  document.getElementById("rp-season").addEventListener("change", (e) => {
    location.hash = `#replay/${e.target.value}`;   // reset to the season's first week
  });
  document.getElementById("rp-week").addEventListener("change", (e) => {
    location.hash = `#replay/${season}/${e.target.value}`;
  });
  view.querySelectorAll(".player-card[data-player-id]").forEach((el) => {
    el.classList.add("clickable");
    el.addEventListener("click", () => {
      location.hash = `#replay/${season}/${week}/player/${encodeURIComponent(el.dataset.playerId)}`;
    });
  });
  bindStripTooltips();
}

async function renderReplayPlayer(season, week, playerId) {
  view.innerHTML = `<div class="loading"><div class="spinner"></div><p>Loading player…</p></div>`;
  let p;
  try {
    p = await api(`/api/replay/${season}/${week}/player/${encodeURIComponent(playerId)}`);
  } catch (err) {
    return renderError(err);
  }
  weekPill.textContent = `Replay · ${season} · Wk ${week}`;
  const tags = [
    p.is_rookie ? "Rookie" : p.is_new_team ? "New team" : null,
    p.years_exp != null ? `${Math.round(p.years_exp)} yr exp` : null,
    p.age != null ? `age ${Math.round(p.age)}` : null,
  ].filter(Boolean).join(" · ");
  view.innerHTML = `
    <button class="back-btn" id="back">← Week ${week}</button>
    <section class="player-detail">
      <div class="player-head">
        <span class="pos-badge pos-${esc(p.position)}">${esc(p.position)}</span>
        <span class="player-name" style="font-size:22px">${esc(p.name)}</span>
        <span class="player-sub">${p.is_home ? "vs" : "@"} ${esc(p.opponent)}</span>
      </div>
      <div class="pd-sub">${esc(p.team)} · ${season} Week ${week}${tags ? ` · ${esc(tags)}` : ""}</div>
      <div class="pd-stats">${p.forecasts.map((f) => statRow(p, f)).join("")}</div>
      <div class="pts-scale-note">gold marker = actual result · white tick = forecast median · bar = p10–p90</div>
    </section>`;
  document.getElementById("back").addEventListener("click", () => {
    location.hash = `#replay/${season}/${week}`;
  });
  bindStripTooltips();
}

/* ---------- draft board ---------- */

function wkStrip(weeks, firstWeek, lastWeek, posMax) {
  const cells = [];
  for (let w = firstWeek; w <= lastWeek; w++) {
    const p = weeks[w];
    if (p == null) {
      cells.push(`<span class="wk-cell bye" data-tip="Week ${w}: bye"></span>`);
    } else {
      const alpha = Math.max(0.12, Math.min(1, p / (posMax || 1)));
      cells.push(`<span class="wk-cell" style="opacity:${alpha.toFixed(2)}"
                        data-tip="Week ${w}: ${fmt(p)} proj"></span>`);
    }
  }
  return `<span class="wk-strip">${cells.join("")}</span>`;
}

/* Market columns. `value` is (market rank − our rank) over the players some
   outside source actually ranks, so a positive number means we like a player
   more than the room does. Everything degrades to an em dash: the board has to
   read correctly when latest_market.parquet is missing or a source is stale. */

function adpCell(p) {
  if (p.espn_adp) return `<span data-tip="ESPN average draft position">${p.espn_adp.toFixed(1)}</span>`;
  if (p.sleeper_rank) {
    return `<span class="draft-dim" data-tip="No ESPN ADP — Sleeper search rank ${p.sleeper_rank}
                   (a coarse ordering, not an ADP)">~${p.sleeper_rank}</span>`;
  }
  return `<span class="draft-dim" data-tip="No outside source ranks this player">—</span>`;
}

function valueCell(p) {
  if (p.value == null) return `<span class="draft-dim">—</span>`;
  const tip = `Our board: #${p.our_rank} · market: #${p.market_rank}`;
  const badge = p.value_tier
    ? `<span class="value-badge ${p.value_tier}">${p.value_tier === "sleeper" ? "sleeper" : "reach"}</span>`
    : "";
  return `<span class="draft-value ${p.value > 0 ? "up" : p.value < 0 ? "down" : ""}"
                data-tip="${tip}">${p.value > 0 ? "+" : ""}${p.value}</span>${badge}`;
}

function spreadCell(p) {
  if (p.proj_spread == null) {
    return `<span class="draft-dim" data-tip="Needs at least two projections">—</span>`;
  }
  const parts = [`ours ${fmt(p.total_p50)}`];
  if (p.espn_proj_pts) parts.push(`ESPN ${fmt(p.espn_proj_pts)}`);
  if (p.fft_proj_ppr) parts.push(`FFToday ${fmt(p.fft_proj_ppr)}`);
  const wide = p.proj_spread_pct >= 0.15;
  return `<span class="draft-spread ${wide ? "wide" : ""}"
                data-tip="${parts.join(" · ")} — ${p.proj_sources} of 3 sources"
          >±${fmt(p.proj_spread)}<span class="draft-dim"> ${p.proj_sources}/3</span></span>`;
}

function marketNote(market) {
  if (!market || !market.available) {
    return `No market data — run <code>gameday market</code> to add ESPN / FFToday / Sleeper.`;
  }
  if (market.partial_season) {
    return `Market: ADP and value shown; projection spread hidden mid-season
            (our total covers the remaining weeks, ESPN and FFToday are full-season).`;
  }
  const cov = Object.entries(market.coverage || {})
    .map(([k, v]) => `${k} ${v.top100}/100`).join(" · ");
  const age = market.fetched_at
    ? `fetched ${new Date(market.fetched_at).toLocaleString()}` : "";
  return `Top-100 match rate — ${cov} · ${age}.
          Unmatched players are mostly backup QBs no outside source projects.`;
}

async function renderDraft(position = "ALL", tier = "") {
  view.innerHTML = `<div class="loading"><div class="spinner"></div><p>Building the draft board…</p></div>`;
  let data;
  try {
    data = await api(`/api/draft?position=${encodeURIComponent(position)}`
                     + `&tier=${encodeURIComponent(tier)}`);
  } catch (err) {
    if (err.status === 404) {
      view.innerHTML = `<div class="error-box"><h2>No season projection yet</h2>
        <p>The draft board appears after the next <code>gameday refresh</code>.</p></div>`;
      return;
    }
    return renderError(err);
  }
  const players = data.players || [];
  weekPill.textContent = `${data.season} draft`;
  const positions = ["ALL", "QB", "RB", "WR", "TE"];
  const tiers = [["sleepers", "sleeper"], ["reaches", "reach"]];
  // Week-cell shading is scaled per position (vs the position's best week),
  // so an ALL view still reads sensibly within each row's position.
  const posMax = {};
  players.forEach((p) => {
    const best = Math.max(...Object.values(p.weeks || {}), 0);
    posMax[p.position] = Math.max(posMax[p.position] || 0, best);
  });
  view.innerHTML = `
    <div class="section-title">Draft Board — ${data.season} · weeks ${data.first_week}–${data.last_week}</div>
    <div class="draft-pills">
      ${positions.map((p) => `
        <button class="draft-pill ${!tier && p === position ? "active" : ""}"
                data-hash="${p === "ALL" ? "#draft" : `#draft/${p}`}">${p}</button>`).join("")}
      <span class="draft-pill-sep"></span>
      ${tiers.map(([slug, name]) => `
        <button class="draft-pill tier ${tier === name ? "active" : ""}"
                data-hash="#draft/${slug}">${slug}</button>`).join("")}
    </div>
    ${players.length ? "" : `<div class="draft-note">No players match this filter.</div>`}
    <div class="draft-scroll">
    <table class="draft-table">
      <thead><tr>
        <th></th><th>Player</th><th>Pos</th><th>Team</th><th>Bye</th>
        <th class="num">Total</th><th class="num">Floor–Ceiling</th>
        <th class="num">VORP</th>
        <th class="num">ADP</th><th class="num">Value</th><th class="num">Spread</th>
        <th>Weekly projection</th>
      </tr></thead>
      <tbody>
        ${players.map((p, i) => `
          <tr>
            <td class="draft-rank">${i + 1}</td>
            <td><b>${esc(p.name)}</b></td>
            <td><span class="pos-badge pos-${esc(p.position)}">${esc(p.position)}</span></td>
            <td>${esc(p.team)}</td>
            <td>${p.bye ?? "—"}</td>
            <td class="num"><b>${fmt(p.total_p50)}</b></td>
            <td class="num">${fmt(p.total_floor)}–${fmt(p.total_ceiling)}</td>
            <td class="num draft-vorp ${p.vorp < 0 ? "neg" : ""}">${p.vorp > 0 ? "+" : ""}${fmt(p.vorp)}</td>
            <td class="num">${adpCell(p)}</td>
            <td class="num">${valueCell(p)}</td>
            <td class="num">${spreadCell(p)}</td>
            <td>${wkStrip(p.weeks || {}, data.first_week, data.last_week, posMax[p.position])}</td>
          </tr>`).join("")}
      </tbody>
    </table>
    </div>
    <div class="draft-note">
      Total = sum of weekly median projections (${data.model_version || "current models"}) ·
      VORP = points over the replacement starter
      (${Object.entries(data.replacement || {}).map(([k, v]) => `${k}${REPL_RANKS[k] || ""} ${fmt(v)}`).join(" · ")}) ·
      floor–ceiling sums weekly p25/p75 — an envelope, not a season quantile.
    </div>
    <div class="draft-note">
      Value = market rank − our rank across the ${data.market?.ranked_players || 0} players an
      outside source ranks; positive means we are higher on them than the room
      (±${data.market?.round_size || 12} = one round). Sleeper/reach flags only fire inside the
      draftable top ${data.market?.draft_pool_size || 180}. Spread = the gap between our, ESPN's,
      and FFToday's season projections. ${marketNote(data.market)}
    </div>`;
  view.querySelectorAll(".draft-pill").forEach((btn) =>
    btn.addEventListener("click", () => { location.hash = btn.dataset.hash; }));
  bindTipTargets();
}

const REPL_RANKS = { QB: 12, RB: 30, WR: 36, TE: 12 };

function bindTipTargets() {
  view.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      tooltip.textContent = el.dataset.tip;
      tooltip.hidden = false;
      tooltip.style.left = `${e.clientX + 12}px`;
      tooltip.style.top = `${e.clientY + 12}px`;
    });
    el.addEventListener("mouseleave", () => { tooltip.hidden = true; });
  });
}

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

function updateNav() {
  const isReplay = location.hash.startsWith("#replay");
  const isDraft = location.hash.startsWith("#draft");
  document.getElementById("nav-slate")?.classList.toggle("active", !isReplay && !isDraft);
  document.getElementById("nav-draft")?.classList.toggle("active", isDraft);
  document.getElementById("nav-replay")?.classList.toggle("active", isReplay);
}

function route() {
  const h = location.hash;
  let m;
  if ((m = h.match(/^#replay\/(\d+)\/(\d+)\/player\/(.+)$/))) {
    renderReplayPlayer(Number(m[1]), Number(m[2]), decodeURIComponent(m[3]));
  } else if ((m = h.match(/^#replay(?:\/(\d+))?(?:\/(\d+))?$/))) {
    renderReplay(m[1] ? Number(m[1]) : null, m[2] ? Number(m[2]) : null);
  } else if ((m = h.match(/^#draft\/(sleepers|reaches)$/))) {
    renderDraft("ALL", m[1] === "sleepers" ? "sleeper" : "reach");
  } else if ((m = h.match(/^#draft(?:\/(QB|RB|WR|TE))?$/))) {
    renderDraft(m[1] || "ALL");
  } else if ((m = h.match(/^#game\/(.+)$/))) {
    renderGame(decodeURIComponent(m[1]));
  } else {
    renderSlate();
  }
  updateNav();
}

document.getElementById("brand-link").addEventListener("click", (e) => {
  e.preventDefault();
  location.hash = "";
});

window.addEventListener("hashchange", route);
route();
