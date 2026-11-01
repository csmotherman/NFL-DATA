(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root && root.document) root.NFLMatchup = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const EPA = [
    { id: 'overall', label: 'Overall', off: 'adj_off_epa_per_play', def: 'adj_def_epa_per_play_allowed', unit: 'epa', note: 'EPA per offensive play' },
    { id: 'pass', label: 'Passing', off: 'adj_off_epa_per_pass', def: 'adj_def_epa_per_pass_allowed', unit: 'epa', note: 'EPA per dropback, including sacks' },
    { id: 'rush', label: 'Rushing', off: 'adj_off_epa_per_rush', def: 'adj_def_epa_per_rush_allowed', unit: 'epa', note: 'EPA per carry' }
  ];
  const CONTEXT = [
    { id: 'yd_db', label: 'Yards / dropback', off: 'pre_off_pass_yd_db', def: 'pre_def_pass_yd_db_allowed', unit: 'yards' },
    { id: 'yd_att', label: 'Yards / carry', off: 'pre_off_rush_yd_att', def: 'pre_def_rush_yd_att_allowed', unit: 'yards' },
    { id: 'pass_fd', label: 'Pass first-down rate', off: 'pre_off_pass_fd_rate', def: 'pre_def_pass_fd_rate_allowed', unit: 'percent' },
    { id: 'rush_fd', label: 'Rush first-down rate', off: 'pre_off_rush_fd_rate', def: 'pre_def_rush_fd_rate_allowed', unit: 'percent' },
    { id: 'cpoe', label: 'Completion above expected', off: 'pre_off_cpoe', def: 'pre_def_cpoe_allowed', unit: 'percentPoints' },
    { id: 'completion', label: 'Completion rate', off: 'pre_off_completion_rate', def: 'pre_def_completion_rate_allowed', unit: 'percent' },
    { id: 'sack', label: 'Sack avoidance vs pressure', off: 'pre_off_sack_rate', def: 'pre_def_sack_rate_allowed', unit: 'percent', offSign: -1, defSign: -1 },
    { id: 'turnovers', label: 'Ball security vs takeaways', off: 'pre_off_turnover_rate', def: 'pre_def_turnover_rate_forced', unit: 'percent', offSign: -1, defSign: -1 }
  ];

  function number(value) {
    if (value === null || value === undefined || String(value).trim() === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/&/g, '&amp;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function distribution(rows, column, sign) {
    const values = rows.map((r) => number(r[column])).filter((v) => v !== null)
      .map((v) => v * (sign || 1));
    if (values.length < 16) return null;
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (values.length - 1);
    const sd = Math.sqrt(variance);
    if (!Number.isFinite(sd) || sd < 1e-10) return null;
    return { values: values, mean: mean, sd: sd, n: values.length };
  }

  function percentile(value, dist) {
    if (value === null || !dist) return null;
    const tolerance = Math.max(1e-12, Math.abs(value) * 1e-10);
    const below = dist.values.filter((v) => v < value - tolerance).length;
    const equal = dist.values.filter((v) => Math.abs(v - value) <= tolerance).length;
    return Math.round(1000 * (below + equal / 2) / dist.n) / 10;
  }

  function evaluate(rows, offense, defense, metric) {
    const a = rows.find((r) => r.team === offense);
    const b = rows.find((r) => r.team === defense);
    if (!a || !b || offense === defense) return null;
    const offValue = number(a[metric.off]);
    const defValue = number(b[metric.def]);
    const offSign = metric.offSign || 1;
    const defSign = metric.defSign || 1;
    const offDist = distribution(rows, metric.off, offSign);
    const defDist = distribution(rows, metric.def, defSign);
    if (offValue === null || defValue === null || !offDist || !defDist) return null;
    const offZ = (offSign * offValue - offDist.mean) / offDist.sd;
    const defZ = (defSign * defValue - defDist.mean) / defDist.sd;
    return {
      id: metric.id, label: metric.label, note: metric.note || '',
      unit: metric.unit, offense: offense, defense: defense,
      offenseValue: offValue, defenseValue: defValue,
      offenseZ: offZ, defenseVulnerabilityZ: defZ,
      offensePercentile: percentile(offValue * offSign, offDist),
      defenseVulnerabilityPercentile: percentile(defValue * defSign, defDist),
      index: (offZ + defZ) / 2,
      leagueCount: Math.min(offDist.n, defDist.n)
    };
  }

  function compare(rows, offense, defense, metrics) {
    return metrics.map((metric) => evaluate(rows, offense, defense, metric));
  }

  function assessment(value) {
    if (value === null || !Number.isFinite(value)) return { label: 'Unavailable', tone: 'neutral' };
    if (value >= 0.75) return { label: 'Strong favorable signal', tone: 'positive' };
    if (value >= 0.30) return { label: 'Favorable signal', tone: 'positive' };
    if (value <= -0.75) return { label: 'Strong defensive resistance', tone: 'negative' };
    if (value <= -0.30) return { label: 'Defensive resistance', tone: 'negative' };
    return { label: 'Mixed / neutral', tone: 'neutral' };
  }

  function expectation(metric) {
    if (!metric) return 'Insufficient comparable data.';
    if (metric.offenseZ >= 0.5 && metric.defenseVulnerabilityZ <= -0.5) {
      return 'Strength vs strength: strong production meets a restrictive defense.';
    }
    if (metric.offenseZ <= -0.5 && metric.defenseVulnerabilityZ >= 0.5) {
      return 'Weakness vs weakness: inconsistent production faces a permissive defense.';
    }
    if (metric.index >= 0.75) return 'Both units point toward an unusually favorable efficiency environment.';
    if (metric.index >= 0.30) return 'The league-relative efficiency profile leans toward the offense.';
    if (metric.index <= -0.75) return 'Both units point toward an unusually restrictive efficiency environment.';
    if (metric.index <= -0.30) return 'The league-relative efficiency profile leans toward the defense.';
    return 'Conflicting or average-strength signals; no clear efficiency mismatch.';
  }

  function formatValue(value, unit) {
    if (value === null || !Number.isFinite(value)) return '—';
    if (unit === 'epa') return (value > 0 ? '+' : '') + value.toFixed(3);
    if (unit === 'percent') return (value * 100).toFixed(1) + '%';
    if (unit === 'percentPoints') return (value > 0 ? '+' : '') + value.toFixed(1) + ' pp';
    return value.toFixed(2);
  }

  function signed(value) {
    if (value === null || !Number.isFinite(value)) return '—';
    return (value > 0 ? '+' : '') + value.toFixed(2);
  }

  function metricCard(metric, compact) {
    if (!metric) return '<div class="matchup-metric matchup-no-data">Data unavailable for this split.</div>';
    const score = assessment(metric.index);
    const headline = compact ? '' : '<div class="matchup-metric-expectation">' + escapeHtml(expectation(metric)) + '</div>';
    const note = metric.note && !compact ? '<span class="matchup-detail">' + escapeHtml(metric.note) + '</span>' : '';
    return '<article class="matchup-metric' + (compact ? ' compact' : '') + '">' +
      '<div class="matchup-metric-top"><div><h3>' + escapeHtml(metric.label) + '</h3>' + note + '</div>' +
      '<span class="matchup-signal ' + score.tone + '">' + escapeHtml(score.label) + ' · ' + signed(metric.index) + ' index</span></div>' +
      '<div class="matchup-unit-grid">' +
        '<div class="matchup-unit"><span>' + escapeHtml(metric.offense) + ' offense</span>' +
          '<strong>' + formatValue(metric.offenseValue, metric.unit) + '</strong>' +
          '<small>' + metric.offensePercentile.toFixed(0) + 'th favorability percentile · ' + signed(metric.offenseZ) + ' SD</small></div>' +
        '<div class="matchup-unit"><span>' + escapeHtml(metric.defense) + ' defense</span>' +
          '<strong>' + formatValue(metric.defenseValue, metric.unit) + '</strong>' +
          '<small>' + metric.defenseVulnerabilityPercentile.toFixed(0) + 'th susceptibility percentile · ' +
          signed(metric.defenseVulnerabilityZ) + ' SD</small></div>' +
      '</div>' + headline + '</article>';
  }

  function compareTile(a, b, label) {
    if (!a || !b) return '<article class="matchup-comparison-card"><h3>' + escapeHtml(label) + '</h3><p>Insufficient data</p></article>';
    const gap = a.index - b.index;
    const leader = Math.abs(gap) < 0.3 ? 'No clear relative separation' :
      (gap > 0 ? a.offense : b.offense) + ' has the better efficiency matchup';
    const pct = Math.min(100, Math.abs(gap) / 2.5 * 100);
    return '<article class="matchup-comparison-card"><h3>' + escapeHtml(label) + '</h3>' +
      '<strong>' + escapeHtml(leader) + '</strong>' +
      '<div class="matchup-index-row"><span>' + escapeHtml(a.offense) + ' ' + signed(a.index) + '</span>' +
      '<span>' + escapeHtml(b.offense) + ' ' + signed(b.index) + '</span></div>' +
      '<div class="matchup-gap-rail" aria-hidden="true"><div style="width:' + pct.toFixed(1) +
      '%;margin-left:' + (gap < 0 ? (100 - pct).toFixed(1) : '0') + '%"></div></div>' +
      '<small>Relative index gap: ' + signed(gap) + ' · not a points or betting edge</small></article>';
  }

  function panel(offense, defense, primary, context, games, windowGames) {
    return '<section class="matchup-panel"><div class="matchup-panel-heading">' +
      '<h2>' + escapeHtml(offense) + ' offense <span>vs ' + escapeHtml(defense) + ' defense</span></h2>' +
      '<p>' + escapeHtml(offense) + ': ' + escapeHtml(games == null ? '—' : games) + ' season games · rolling history: ' +
      escapeHtml(windowGames == null ? '—' : windowGames) + ' games</p></div>' +
      '<h3 class="matchup-group-title">Opponent-adjusted season EPA</h3>' +
      primary.map((metric) => metricCard(metric, false)).join('') +
      '<h3 class="matchup-group-title">Recent unadjusted efficiency context</h3>' +
      '<p class="matchup-group-note">Rolling descriptive rates; not opponent-adjusted. Percentiles compare every NFL team on the same measure.</p>' +
      context.map((metric) => metricCard(metric, true)).join('') + '</section>';
  }

  function defaultPair(teams, board) {
    const teamSet = new Set(teams);
    const game = (board || []).find((r) =>
      teamSet.has(r.away_team) && teamSet.has(r.home_team) &&
      (number(r.away_score) === null || number(r.home_score) === null));
    return game ? [game.away_team, game.home_team] : teams.slice(0, 2);
  }

  function init(seasonStats, rollingStats, board, report) {
    const elA = document.getElementById('matchupTeamA');
    const elB = document.getElementById('matchupTeamB');
    const content = document.getElementById('matchupResults');
    if (!elA || !elB || !content) return;
    const all = (seasonStats || []).filter((r) => r.team && number(r.season) !== null);
    const latestSeason = Math.max(...all.map((r) => number(r.season)));
    const stats = all.filter((r) => number(r.season) === latestSeason);
    const recent = (rollingStats || []).filter((r) => r.team);
    const teams = [...new Set(stats.map((r) => r.team))].sort();
    if (teams.length < 16) {
      content.innerHTML = '<div class="notice" role="status">Team statistics are unavailable or incomplete. No matchup edge has been calculated.</div>';
      elA.disabled = true;
      elB.disabled = true;
      document.getElementById('matchupSwap').disabled = true;
      return;
    }
    const options = teams.map((team) => '<option value="' + escapeHtml(team) + '">' + escapeHtml(team) + '</option>').join('');
    elA.innerHTML = options;
    elB.innerHTML = options;
    const pair = defaultPair(teams, board);
    elA.value = pair[0];
    elB.value = pair[1];
    let previous = [elA.value, elB.value];

    function render() {
      const a = elA.value;
      const b = elB.value;
      if (a === b) {
        content.innerHTML = '<div class="notice" role="status">Choose two different teams.</div>';
        return;
      }
      previous = [a, b];
      const first = compare(stats, a, b, EPA);
      const second = compare(stats, b, a, EPA);
      const firstRecent = compare(recent, a, b, CONTEXT);
      const secondRecent = compare(recent, b, a, CONTEXT);
      const aRow = stats.find((r) => r.team === a);
      const bRow = stats.find((r) => r.team === b);
      const aRolling = recent.find((r) => r.team === a);
      const bRolling = recent.find((r) => r.team === b);
      const stamp = report && report.generated_at_utc ? String(report.generated_at_utc).slice(0, 10) : 'not provided';
      document.getElementById('matchupMeta').textContent = 'Season ' + latestSeason +
        ' · ' + teams.length + '-team reference · Data generated ' + stamp + ' UTC';
      const warnings = [];
      if (Math.min(number(aRow.games) || 0, number(bRow.games) || 0) < 4) {
        warnings.push('Small current-season sample: early estimates may be unstable.');
      }
      if (recent.length < 16) warnings.push('Rolling context is unavailable; only season EPA can be compared.');
      document.getElementById('matchupWarning').textContent = warnings.join(' ');
      const summary = '<div class="matchup-comparison-grid">' +
        EPA.map((m, i) => compareTile(first[i], second[i], m.label)).join('') + '</div>';
      content.innerHTML = summary + '<div class="matchup-panel-grid">' +
        panel(a, b, first, firstRecent, number(aRow.games), aRolling && number(aRolling.games_in_window)) +
        panel(b, a, second, secondRecent, number(bRow.games), bRolling && number(bRolling.games_in_window)) +
        '</div>';
    }

    elA.addEventListener('change', function () {
      if (elA.value === elB.value) elB.value = previous[0];
      render();
    });
    elB.addEventListener('change', function () {
      if (elB.value === elA.value) elA.value = previous[1];
      render();
    });
    document.getElementById('matchupSwap').addEventListener('click', function () {
      const from = elA.value;
      elA.value = elB.value;
      elB.value = from;
      render();
    });
    render();
  }

  return { EPA: EPA, CONTEXT: CONTEXT, number: number, distribution: distribution,
    percentile: percentile, evaluate: evaluate, compare: compare, assessment: assessment,
    expectation: expectation, defaultPair: defaultPair, init: init };
});
