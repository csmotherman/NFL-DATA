(function () {
  'use strict';

  var PATHS = {
    board: 'outputs/latest_betting_board.csv',
    props: 'outputs/latest_player_props.csv',
    propEdges: 'outputs/latest_player_prop_edges.csv',
    gameHistory: 'outputs/game_bet_history.csv',
    propHistory: 'outputs/prop_bet_history.csv',
    historySummary: 'outputs/bet_history_summary.json',
    modelReport: 'outputs/model_report.json',
    propReport: 'outputs/player_prop_model_report.json',
    propForward: 'outputs/prop_forward_report.json'
  };

  var state = {
    board: [], props: [], propEdges: [], gameHistory: [], propHistory: [],
    historySummary: {}, modelReport: {}, propReport: {}, propForward: {},
    historyType: 'games'
  };

  function $(id) { return document.getElementById(id); }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }
  function num(value) {
    if (value === null || value === undefined || value === '') return null;
    var n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  function fmt(value, digits) {
    var n = num(value);
    return n === null ? '—' : n.toFixed(digits == null ? 1 : digits);
  }
  function signed(value, digits) {
    var n = num(value);
    if (n === null) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(digits == null ? 1 : digits);
  }
  function pct(value, digits) {
    var n = num(value);
    return n === null ? '—' : (n * 100).toFixed(digits == null ? 1 : digits) + '%';
  }
  function odds(value) {
    var n = num(value);
    if (n === null) return '—';
    return n > 0 ? '+' + Math.round(n) : String(Math.round(n));
  }
  function dateLabel(value) {
    if (!value) return '—';
    var d = new Date(value + 'T12:00:00');
    if (isNaN(d.getTime())) return value;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }
  function marketLabel(value) {
    return String(value || '—').replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }
  function statusTag(value) {
    var s = String(value || 'LEAN').toUpperCase();
    var cls = s.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    if (s.indexOf('VALIDATED') >= 0) cls = 'validated';
    if (s === 'WATCH') cls = 'watch';
    if (s === 'WIN') cls = 'win';
    if (s === 'LOSS') cls = 'loss';
    if (s === 'PUSH') cls = 'push';
    return '<span class="status-tag ' + cls + '">' + esc(s) + '</span>';
  }
  function summaryCard(label, value, meta) {
    return '<div class="summary-card"><div class="summary-label">' + esc(label) + '</div>' +
      '<div class="summary-value">' + esc(value) + '</div><div class="summary-meta">' + esc(meta || '') + '</div></div>';
  }

  function parseCsv(text) {
    text = String(text || '').replace(/^\uFEFF/, '');
    if (!text.trim()) return [];
    var rows = [], row = [], field = '', quoted = false;
    for (var i = 0; i < text.length; i++) {
      var c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (c === '"') quoted = false;
        else field += c;
      } else if (c === '"') quoted = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
      else field += c;
    }
    if (field.length || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
    rows = rows.filter(function (r) { return r.some(function (v) { return String(v).trim() !== ''; }); });
    if (rows.length < 2) return [];
    var headers = rows.shift().map(function (h) { return h.trim(); });
    return rows.map(function (r) {
      var obj = {};
      headers.forEach(function (h, idx) { obj[h] = r[idx] == null ? '' : r[idx]; });
      return obj;
    });
  }

  async function fetchText(path) {
    var res = await fetch(path, { cache: 'no-store' });
    if (!res.ok) throw new Error(path + ' returned ' + res.status);
    return res.text();
  }
  async function fetchCsv(path) {
    try { return parseCsv(await fetchText(path)); }
    catch (e) { console.warn(e.message); return []; }
  }
  async function fetchJson(path) {
    try { return JSON.parse(await fetchText(path)); }
    catch (e) { console.warn(e.message); return {}; }
  }

  function isFinal(row) { return num(row.home_score) !== null && num(row.away_score) !== null; }
  function spreadMarketText(row) {
    var line = num(row.spread_line);
    if (line === null) return '—';
    if (line === 0) return 'PK';
    return (line > 0 ? row.home_team : row.away_team) + ' ' + signed(-Math.abs(line), 1);
  }
  function modelMarginText(row) {
    var m = num(row.model_home_margin);
    if (m === null) return '—';
    return (m >= 0 ? row.home_team : row.away_team) + ' ' + Math.abs(m).toFixed(1);
  }
  function projectedScore(row) {
    var h = num(row.model_home_score), a = num(row.model_away_score);
    if (h === null || a === null) return '—';
    return row.away_team + ' ' + Math.round(a) + ' · ' + row.home_team + ' ' + Math.round(h);
  }
  function finalScore(row) {
    return row.away_team + ' ' + Math.round(num(row.away_score)) + ' · ' + row.home_team + ' ' + Math.round(num(row.home_score));
  }
  function hasWatch(row) {
    return String(row.spread_status).toUpperCase() === 'WATCH' || String(row.total_status).toUpperCase() === 'WATCH' ||
      String(row.spread_status).toUpperCase() === 'VALIDATED' || String(row.total_status).toUpperCase() === 'VALIDATED';
  }

  function renderWeekSummary() {
    var rows = state.board;
    var upcoming = rows.filter(function (r) { return !isFinal(r); });
    var complete = rows.length - upcoming.length;
    var spreadSignals = upcoming.filter(function (r) { return String(r.spread_status).toUpperCase() !== 'LEAN'; }).length;
    var totalSignals = upcoming.filter(function (r) { return String(r.total_status).toUpperCase() !== 'LEAN'; }).length;
    var best = upcoming.reduce(function (max, r) { return Math.max(max, num(r.max_edge) || 0); }, 0);
    $('weekSummary').innerHTML =
      summaryCard('Upcoming games', String(upcoming.length), complete + ' completed in current file') +
      summaryCard('Spread signals', String(spreadSignals), 'WATCH / VALIDATED only') +
      summaryCard('Total signals', String(totalSignals), 'WATCH / VALIDATED only') +
      summaryCard('Largest edge', best ? best.toFixed(1) + ' pts' : '—', 'Largest current spread or total gap');

    if (rows.length) $('weekLabel').textContent = rows[0].season + ' · WEEK ' + Number(rows[0].week);
  }

  function upcomingRows() {
    var mode = $('weekFilter').value || 'all';
    var rows = state.board.filter(function (r) { return !isFinal(r); });
    if (mode === 'watch') rows = rows.filter(hasWatch);
    if (mode === 'spread') rows = rows.filter(function (r) { return Math.abs(num(r.spread_edge) || 0) >= 1.5; });
    if (mode === 'total') rows = rows.filter(function (r) { return Math.abs(num(r.total_edge) || 0) >= 1.5; });
    return rows.sort(function (a, b) { return (num(b.max_edge) || 0) - (num(a.max_edge) || 0); });
  }

  function renderUpcoming() {
    var rows = upcomingRows();
    $('upcomingEmpty').hidden = rows.length > 0;
    $('upcomingBody').innerHTML = rows.map(function (r) {
      return '<tr>' +
        '<td class="sticky-col"><span class="game-main">' + esc(r.away_team + ' @ ' + r.home_team) + '</span><span class="game-sub">' + esc(dateLabel(r.gameday)) + '</span></td>' +
        '<td class="number">' + esc(projectedScore(r)) + '</td>' +
        '<td class="number">' + esc(modelMarginText(r)) + '</td>' +
        '<td class="number">' + esc(spreadMarketText(r)) + '</td>' +
        '<td class="number edge-strong">' + esc((r.spread_pick || '—') + ' ' + signed(r.spread_pick_line, 1)) + '</td>' +
        '<td class="number">' + esc(signed(r.spread_edge, 1)) + '</td>' +
        '<td>' + statusTag(r.spread_status) + '</td>' +
        '<td class="number">' + esc(fmt(r.total_line, 1)) + '</td>' +
        '<td class="number">' + esc(fmt(r.model_total, 1)) + '</td>' +
        '<td class="number">' + esc(signed(r.total_edge, 1)) + '</td>' +
        '<td class="number edge-strong">' + esc((r.total_pick || '—') + ' ' + fmt(r.total_line, 1)) + '</td>' +
        '<td class="number edge-strong">' + esc(r.moneyline_pick || '—') + '</td>' +
        '<td class="number">' + esc(odds(r.moneyline_price)) + '</td>' +
        '</tr>';
    }).join('');
  }

  function renderCompleted() {
    var rows = state.board.filter(isFinal).sort(function (a, b) { return String(b.gameday).localeCompare(String(a.gameday)); });
    $('completedSection').style.display = rows.length ? '' : 'none';
    $('completedBody').innerHTML = rows.map(function (r) {
      return '<tr class="final-row"><td><span class="game-main">' + esc(r.away_team + ' @ ' + r.home_team) + '</span><span class="game-sub">' + esc(dateLabel(r.gameday)) + '</span></td>' +
        '<td class="number">' + esc(finalScore(r)) + '</td>' +
        '<td class="number secondary">' + esc(projectedScore(r)) + '</td>' +
        '<td class="number">' + esc((r.spread_pick || '—') + ' ' + signed(r.spread_pick_line, 1)) + '</td>' +
        '<td class="number">' + esc(signed(r.spread_edge, 1)) + '</td>' +
        '<td class="number">' + esc((r.total_pick || '—') + ' ' + fmt(r.total_line, 1)) + '</td>' +
        '<td class="number">' + esc(signed(r.total_edge, 1)) + '</td>' +
        '<td class="number">' + esc((r.moneyline_pick || '—') + ' ' + odds(r.moneyline_price)) + '</td></tr>';
    }).join('');
  }

  function uniquePropMarkets() {
    var source = state.propEdges.length ? state.propEdges : state.props;
    var set = {};
    source.forEach(function (r) { if (r.market) set[r.market] = true; });
    return Object.keys(set).sort();
  }
  function setupPropMarkets() {
    var markets = uniquePropMarkets();
    $('propMarket').innerHTML = '<option value="all">All markets</option>' + markets.map(function (m) {
      return '<option value="' + esc(m) + '">' + esc(marketLabel(m)) + '</option>';
    }).join('');
  }
  function renderPropCallout() {
    if (state.propEdges.length) {
      $('propCallout').innerHTML = '<strong>Sportsbook lines are loaded.</strong> Model projection, market line, price and estimated value are shown together. Estimated EV is research output, not a guaranteed edge.';
    } else {
      var msg = state.propForward.message || 'No sportsbook prop quotes are currently archived.';
      $('propCallout').innerHTML = '<strong>Projection mode only.</strong> ' + esc(msg) + ' Without an actual sportsbook line and price, these projections are not over/under recommendations.';
    }
  }
  function filteredProps() {
    var rows = (state.propEdges.length ? state.propEdges : state.props).slice();
    var market = $('propMarket').value || 'all';
    var search = ($('propSearch').value || '').trim().toLowerCase();
    if (market !== 'all') rows = rows.filter(function (r) { return r.market === market; });
    if (search) rows = rows.filter(function (r) { return [r.player_name, r.team, r.opponent, r.position].join(' ').toLowerCase().indexOf(search) >= 0; });
    return rows;
  }
  function renderProps() {
    var edges = state.propEdges.length > 0;
    var rows = filteredProps();
    if (edges) rows.sort(function (a, b) { return (num(b.model_ev_per_unit) || -999) - (num(a.model_ev_per_unit) || -999); });
    else rows.sort(function (a, b) { return String(a.market).localeCompare(String(b.market)) || (num(b.projection) || 0) - (num(a.projection) || 0); });
    var shown = rows.slice(0, 250);
    $('propEmpty').hidden = shown.length > 0;
    $('propCount').textContent = 'Showing ' + shown.length + ' of ' + rows.length + ' matching rows';

    if (edges) {
      $('propModeLabel').textContent = 'LIVE MARKET COMPARISON';
      $('propTitle').textContent = 'Prop edge board';
      $('propSubtitle').textContent = 'Sorted by estimated model value against the quoted price.';
      $('propHead').innerHTML = '<tr><th>Player</th><th>Team</th><th>Market</th><th>Line</th><th>Projection</th><th>Edge</th><th>Lean</th><th>Model P</th><th>Price</th><th>Est. EV</th><th>Status</th></tr>';
      $('propBody').innerHTML = shown.map(function (r) {
        var over = String(r.lean || '').toUpperCase() === 'OVER';
        var price = over ? r.price_over : r.price_under;
        return '<tr><td><span class="game-main">' + esc(r.player_name) + '</span><span class="game-sub">' + esc(r.position || '') + '</span></td>' +
          '<td>' + esc((r.team || '—') + (r.opponent ? ' vs ' + r.opponent : '')) + '</td><td>' + esc(marketLabel(r.market)) + '</td>' +
          '<td class="number">' + esc(fmt(r.line, 1)) + '</td><td class="number">' + esc(fmt(r.projection, 1)) + '</td>' +
          '<td class="number">' + esc(signed(r.edge, 1)) + '</td><td class="number edge-strong">' + esc(r.lean || '—') + '</td>' +
          '<td class="number">' + esc(pct(r.model_lean_probability, 1)) + '</td><td class="number">' + esc(odds(price)) + '</td>' +
          '<td class="number">' + esc(pct(r.model_ev_per_unit, 1)) + '</td><td>' + statusTag(r.actionability || 'RESEARCH') + '</td></tr>';
      }).join('');
    } else {
      $('propModeLabel').textContent = 'PROJECTIONS';
      $('propTitle').textContent = 'Player projection board';
      $('propSubtitle').textContent = 'Model projections only until sportsbook lines are loaded.';
      $('propHead').innerHTML = '<tr><th>Player</th><th>Team / Opp</th><th>Market</th><th>Projection</th><th>Recent baseline</th><th>Difference</th><th>Usage</th><th>Games</th><th>Availability</th></tr>';
      $('propBody').innerHTML = shown.map(function (r) {
        var diff = (num(r.projection) !== null && num(r.recent_baseline) !== null) ? num(r.projection) - num(r.recent_baseline) : null;
        return '<tr><td><span class="game-main">' + esc(r.player_name) + '</span><span class="game-sub">' + esc(r.position || '') + '</span></td>' +
          '<td>' + esc((r.team || '—') + (r.opponent ? ' vs ' + r.opponent : '')) + '</td><td>' + esc(marketLabel(r.market)) + '</td>' +
          '<td class="number edge-strong">' + esc(fmt(r.projection, 1)) + '</td><td class="number">' + esc(fmt(r.recent_baseline, 1)) + '</td>' +
          '<td class="number">' + esc(signed(diff, 1)) + '</td><td class="number">' + esc(pct(r.usage_share, 1)) + '</td>' +
          '<td class="number">' + esc(r.games_current_season || '—') + '</td><td>' + statusTag(r.availability_flag || 'NOT LISTED') + '</td></tr>';
      }).join('');
    }
  }

  function record(summary) {
    if (!summary || !num(summary.decisions)) return '0-0';
    return (summary.wins || 0) + '-' + (summary.losses || 0) + ((summary.pushes || 0) ? '-' + summary.pushes : '');
  }
  function propHistoryStats() {
    var settled = state.propHistory.filter(function (r) { return ['WIN','LOSS'].indexOf(String(r.result).toUpperCase()) >= 0; });
    var wins = settled.filter(function (r) { return String(r.result).toUpperCase() === 'WIN'; }).length;
    var profit = settled.reduce(function (s, r) { return s + (num(r.unit_profit) || 0); }, 0);
    return { decisions: settled.length, wins: wins, losses: settled.length - wins, hit_rate: settled.length ? wins / settled.length : null, roi: settled.length ? profit / settled.length : null };
  }
  function renderHistorySummary() {
    var spread = state.historySummary.spread_qualified || {};
    var total = state.historySummary.total_qualified || {};
    var ml = state.historySummary.moneyline_all_model_picks || {};
    var props = propHistoryStats();
    $('historySummary').innerHTML =
      summaryCard('Spread', record(spread), spread.decisions ? pct(spread.hit_rate, 1) + ' hit · ' + pct(spread.roi_per_bet, 1) + ' ROI' : 'No settled forward bets') +
      summaryCard('Total', record(total), total.decisions ? pct(total.hit_rate, 1) + ' hit · ' + pct(total.roi_per_bet, 1) + ' ROI' : 'No settled forward bets') +
      summaryCard('Moneyline', record(ml), ml.decisions ? pct(ml.hit_rate, 1) + ' hit · ' + pct(ml.roi_per_bet, 1) + ' ROI' : 'No settled forward bets') +
      summaryCard('Player props', props.decisions ? props.wins + '-' + props.losses : '0-0', props.decisions ? pct(props.hit_rate, 1) + ' hit · ' + pct(props.roi, 1) + ' ROI' : 'No settled locked props');
  }
  function historyWeeks(rows) {
    var map = {};
    rows.forEach(function (r) { if (r.season && r.week) map[r.season + '-' + r.week] = true; });
    return Object.keys(map).sort(function (a,b) { var x=a.split('-').map(Number), y=b.split('-').map(Number); return y[0]-x[0] || y[1]-x[1]; });
  }
  function populateHistoryWeeks() {
    var rows = state.historyType === 'games' ? state.gameHistory : state.propHistory;
    $('historyWeek').innerHTML = '<option value="all">All weeks</option>' + historyWeeks(rows).map(function (k) {
      var p = k.split('-'); return '<option value="' + esc(k) + '">' + esc(p[0] + ' Week ' + Number(p[1])) + '</option>';
    }).join('');
  }
  function historyFiltered(rows) {
    var v = $('historyWeek').value || 'all';
    if (v === 'all') return rows.slice();
    return rows.filter(function (r) { return r.season + '-' + r.week === v; });
  }
  function emptyHistory(title, body) {
    return '<div class="empty-state"><strong>' + esc(title) + '</strong><p>' + esc(body) + '</p></div>';
  }
  function renderGameHistory(rows) {
    if (!rows.length) return emptyHistory('No forward game bets yet', 'The current archive does not contain a pregame-locked settled game. This prevents hindsight reruns from being presented as a live betting record.');
    return '<div class="table-frame"><table class="simple-table"><thead><tr><th>Week</th><th>Game</th><th>Final</th><th>Spread</th><th>Result</th><th>Total</th><th>Result</th><th>Moneyline</th><th>Result</th></tr></thead><tbody>' + rows.map(function (r) {
      return '<tr><td class="number">' + esc(r.season + ' W' + Number(r.week)) + '</td><td><span class="game-main">' + esc(r.away_team + ' @ ' + r.home_team) + '</span><span class="game-sub">' + esc(dateLabel(r.gameday)) + '</span></td>' +
        '<td class="number">' + esc(r.final_score || '—') + '</td><td class="number">' + esc((r.spread_pick || '—') + ' ' + signed(r.spread_pick_line, 1)) + '</td><td>' + statusTag(r.spread_result) + '</td>' +
        '<td class="number">' + esc((r.total_pick || '—') + ' ' + fmt(r.total_line, 1)) + '</td><td>' + statusTag(r.total_result) + '</td>' +
        '<td class="number">' + esc((r.moneyline_pick || '—') + ' ' + odds(r.moneyline_price)) + '</td><td>' + statusTag(r.moneyline_result) + '</td></tr>';
    }).join('') + '</tbody></table></div>';
  }
  function renderPropHistory(rows) {
    if (!rows.length) return emptyHistory('No settled locked props yet', 'A prop result appears here only after an actual sportsbook line and price were archived before kickoff and later settled.');
    return '<div class="table-frame"><table class="simple-table"><thead><tr><th>Week</th><th>Player</th><th>Market</th><th>Bet</th><th>Projection</th><th>Actual</th><th>Result</th><th>Units</th></tr></thead><tbody>' + rows.map(function (r) {
      return '<tr><td class="number">' + esc(r.season + ' W' + Number(r.week)) + '</td><td><span class="game-main">' + esc(r.player_name) + '</span><span class="game-sub">' + esc((r.team || '') + (r.opponent ? ' vs ' + r.opponent : '')) + '</span></td><td>' + esc(marketLabel(r.market)) + '</td>' +
        '<td class="number">' + esc((r.lean || '—') + ' ' + fmt(r.line, 1) + ' ' + odds(r.lean_price)) + '</td><td class="number">' + esc(fmt(r.projection, 1)) + '</td><td class="number">' + esc(fmt(r.actual, 1)) + '</td><td>' + statusTag(r.result) + '</td><td class="number">' + esc(signed(r.unit_profit, 2)) + '</td></tr>';
    }).join('') + '</tbody></table></div>';
  }
  function renderHistory() {
    var rows = historyFiltered(state.historyType === 'games' ? state.gameHistory : state.propHistory);
    $('historyContent').innerHTML = state.historyType === 'games' ? renderGameHistory(rows) : renderPropHistory(rows);
  }

  function renderModelSummary() {
    var m = state.modelReport.independent_model || {};
    $('modelSummary').innerHTML =
      summaryCard('Holdout season', String(state.modelReport.holdout_season || '—'), 'Latest completed holdout') +
      summaryCard('Winner accuracy', pct(m.winner_accuracy, 1), 'Independent football model') +
      summaryCard('Margin MAE', fmt(m.margin_mae, 2), 'Points per game') +
      summaryCard('Total MAE', fmt(m.total_mae, 2), 'Points per game');
  }
  function validationTable(title, data) {
    data = data || {};
    var thresholds = data.thresholds || {};
    var keys = Object.keys(thresholds).sort(function (a,b) { return Number(a)-Number(b); });
    var badge = data.validated_threshold == null ? statusTag('WATCH') : statusTag('VALIDATED');
    var body = keys.length ? keys.map(function (k) {
      var r = thresholds[k] || {};
      return '<tr><td class="number">' + esc(k + '+ pts') + '</td><td class="number">' + esc(r.bets || 0) + '</td><td class="number">' + esc((r.wins || 0) + '-' + (r.losses || 0)) + '</td><td class="number">' + esc(pct(r.win_rate,1)) + '</td><td class="number">' + esc(pct(r.roi_at_minus_110,1)) + '</td><td>' + statusTag(r.stable ? 'VALIDATED' : 'WATCH') + '</td></tr>';
    }).join('') : '<tr><td colspan="6">No walk-forward thresholds available.</td></tr>';
    return '<div class="model-block"><div class="model-block-title"><h3>' + esc(title) + '</h3>' + badge + '</div><div class="table-frame"><table class="simple-table"><thead><tr><th>Threshold</th><th>Bets</th><th>Record</th><th>Hit rate</th><th>ROI @ -110</th><th>Gate</th></tr></thead><tbody>' + body + '</tbody></table></div></div>';
  }
  function renderValidation() {
    var wf = state.modelReport.walk_forward_market_validation || {};
    $('marketValidation').innerHTML = validationTable('Spread', wf.spread) + validationTable('Total', wf.total);
    var markets = state.propReport.markets || {};
    var keys = Object.keys(markets).sort();
    $('playerValidationBody').innerHTML = keys.map(function (k) {
      var r = markets[k] || {};
      var model = num(r.blended_mae != null ? r.blended_mae : r.model_mae);
      var base = num(r.baseline_mae);
      return '<tr><td>' + esc(marketLabel(k)) + '</td><td class="number">' + esc(fmt(model,2)) + '</td><td class="number">' + esc(fmt(base,2)) + '</td><td class="number">' + esc(model !== null && base !== null ? signed(model-base,2) : '—') + '</td><td class="number">' + esc(r.oof_rows || 0) + '</td></tr>';
    }).join('') || '<tr><td colspan="5">No player validation report available.</td></tr>';
  }

  function setRoute(name) {
    if (['week','props','history','model'].indexOf(name) < 0) name = 'week';
    document.querySelectorAll('.page').forEach(function (el) { el.classList.toggle('is-active', el.id === 'page-' + name); });
    document.querySelectorAll('[data-route]').forEach(function (el) { el.classList.toggle('is-active', el.getAttribute('data-route') === name); });
    if (window.location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
    window.scrollTo(0,0);
  }
  function bindUI() {
    document.querySelectorAll('[data-route]').forEach(function (el) { el.addEventListener('click', function (e) { e.preventDefault(); setRoute(el.getAttribute('data-route')); }); });
    $('weekFilter').addEventListener('change', renderUpcoming);
    $('toggleCompleted').addEventListener('click', function () {
      var wrap = $('completedWrap');
      wrap.hidden = !wrap.hidden;
      $('toggleCompleted').textContent = wrap.hidden ? 'Show completed games' : 'Hide completed games';
    });
    $('propMarket').addEventListener('change', renderProps);
    $('propSearch').addEventListener('input', renderProps);
    $('historyWeek').addEventListener('change', renderHistory);
    document.querySelectorAll('[data-history]').forEach(function (el) {
      el.addEventListener('click', function () {
        state.historyType = el.getAttribute('data-history');
        document.querySelectorAll('[data-history]').forEach(function (x) { x.classList.toggle('is-active', x === el); });
        populateHistoryWeeks(); renderHistory();
      });
    });
  }

  async function load() {
    var results = await Promise.all([
      fetchCsv(PATHS.board), fetchCsv(PATHS.props), fetchCsv(PATHS.propEdges), fetchCsv(PATHS.gameHistory), fetchCsv(PATHS.propHistory),
      fetchJson(PATHS.historySummary), fetchJson(PATHS.modelReport), fetchJson(PATHS.propReport), fetchJson(PATHS.propForward)
    ]);
    state.board = results[0]; state.props = results[1]; state.propEdges = results[2]; state.gameHistory = results[3]; state.propHistory = results[4];
    state.historySummary = results[5]; state.modelReport = results[6]; state.propReport = results[7]; state.propForward = results[8];

    renderWeekSummary(); renderUpcoming(); renderCompleted();
    setupPropMarkets(); renderPropCallout(); renderProps();
    renderHistorySummary(); populateHistoryWeeks(); renderHistory();
    renderModelSummary(); renderValidation();

    var stamp = state.propReport.generated_at_utc || state.modelReport.generated_at_utc || '';
    $('refreshStamp').textContent = stamp ? 'Data refresh: ' + String(stamp).replace('T',' ').replace('+00:00',' UTC').replace('Z',' UTC') : '';
    $('loadState').classList.add('ok');
    $('loadState').querySelector('span:last-child').textContent = 'Data loaded';
  }

  bindUI();
  setRoute((window.location.hash || '#week').slice(1));
  load().catch(function (err) {
    console.error(err);
    $('loadState').classList.add('error');
    $('loadState').querySelector('span:last-child').textContent = 'Data error';
  });
})();
