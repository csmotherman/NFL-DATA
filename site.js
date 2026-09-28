(function () {
  'use strict';

  var state = {
    board: [],
    candidates: [],
    propEdges: [],
    propProjections: [],
    gameHistory: [],
    propHistory: [],
    historySummary: {},
    modelReport: {},
    propModelReport: {},
    propForwardReport: {},
    historyType: 'games'
  };

  var PATHS = {
    board: '/outputs/latest_betting_board.csv',
    fallbackBoard: '/outputs/latest_predictions.csv',
    candidates: '/outputs/latest_candidates.csv',
    propEdges: '/outputs/latest_player_prop_edges.csv',
    propProjections: '/outputs/latest_player_props.csv',
    gameHistory: '/outputs/game_bet_history.csv',
    propHistory: '/outputs/prop_bet_history.csv',
    historySummary: '/outputs/bet_history_summary.json',
    modelReport: '/outputs/model_report.json',
    propModelReport: '/outputs/player_prop_model_report.json',
    propForwardReport: '/outputs/prop_forward_report.json'
  };

  function $(id) {
    return document.getElementById(id);
  }

  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function num(value) {
    var n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function fmt(value, digits) {
    var n = num(value);
    if (n === null) return '—';
    return n.toFixed(digits === undefined ? 1 : digits);
  }

  function signed(value, digits) {
    var n = num(value);
    if (n === null) return '—';
    var d = digits === undefined ? 1 : digits;
    return (n > 0 ? '+' : '') + n.toFixed(d);
  }

  function pct(value, digits) {
    var n = num(value);
    if (n === null) return '—';
    return (n * 100).toFixed(digits === undefined ? 1 : digits) + '%';
  }

  function odds(value) {
    var n = num(value);
    if (n === null) return '—';
    return (n > 0 ? '+' : '') + Math.round(n);
  }

  function money(value) {
    var n = num(value);
    if (n === null) return '—';
    return signed(n, 2) + 'u';
  }

  function marketLabel(value) {
    var labels = {
      pass_attempts: 'Pass Attempts',
      completions: 'Completions',
      passing_yards: 'Pass Yards',
      rush_attempts: 'Rush Attempts',
      rushing_yards: 'Rush Yards',
      receptions: 'Receptions',
      receiving_yards: 'Receiving Yards'
    };
    return labels[value] || String(value || '').replace(/_/g, ' ').replace(/\b\w/g, function (m) { return m.toUpperCase(); });
  }

  function dateLabel(value) {
    if (!value) return '';
    var d = new Date(value + (String(value).length === 10 ? 'T12:00:00' : ''));
    if (Number.isNaN(d.getTime())) return String(value);
    return new Intl.DateTimeFormat('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    }).format(d);
  }

  function parseCsv(text) {
    if (!text || !text.trim()) return [];
    var rows = [];
    var row = [];
    var field = '';
    var quoted = false;

    for (var i = 0; i < text.length; i += 1) {
      var ch = text[i];
      var next = text[i + 1];

      if (ch === '"') {
        if (quoted && next === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = !quoted;
        }
      } else if (ch === ',' && !quoted) {
        row.push(field);
        field = '';
      } else if ((ch === '\n' || ch === '\r') && !quoted) {
        if (ch === '\r' && next === '\n') i += 1;
        row.push(field);
        field = '';
        if (row.some(function (cell) { return cell !== ''; })) rows.push(row);
        row = [];
      } else {
        field += ch;
      }
    }

    if (field !== '' || row.length) {
      row.push(field);
      if (row.some(function (cell) { return cell !== ''; })) rows.push(row);
    }

    if (rows.length < 2) return [];
    var headers = rows[0].map(function (header) { return header.trim(); });

    return rows.slice(1).map(function (values) {
      var obj = {};
      headers.forEach(function (header, idx) {
        obj[header] = values[idx] === undefined ? '' : values[idx];
      });
      return obj;
    });
  }

  async function fetchText(path) {
    try {
      var response = await fetch(path, { cache: 'no-store' });
      if (!response.ok) return '';
      return await response.text();
    } catch (err) {
      return '';
    }
  }

  async function fetchCsv(path) {
    return parseCsv(await fetchText(path));
  }

  async function fetchJson(path) {
    try {
      var response = await fetch(path, { cache: 'no-store' });
      if (!response.ok) return {};
      return await response.json();
    } catch (err) {
      return {};
    }
  }

  function fallbackStatus(edge, kind) {
    var n = Math.abs(num(edge) || 0);
    var validated = num(state.modelReport['validated_' + kind + '_threshold']);
    var watch = num(state.modelReport[kind + '_watch_threshold']);
    if (validated !== null && n >= validated) return 'VALIDATED';
    if (n >= (watch === null ? 3 : watch)) return 'WATCH';
    return 'LEAN';
  }

  function normalizeBoard(rows) {
    return rows.map(function (row) {
      var spreadEdge = num(row.spread_edge);
      var totalEdge = num(row.total_edge);
      var homeMargin = num(row.model_home_margin);
      var total = num(row.model_total);
      var marketSpread = num(row.spread_line);
      var spreadPick = row.spread_pick || (spreadEdge !== null && spreadEdge >= 0 ? row.home_team : row.away_team);
      var teamLine = num(row.spread_pick_line);
      if (teamLine === null && marketSpread !== null) {
        teamLine = spreadPick === row.home_team ? -marketSpread : marketSpread;
      }

      row.spread_pick = spreadPick;
      row.spread_pick_line = teamLine;
      row.spread_status = row.spread_status || fallbackStatus(spreadEdge, 'spread');
      row.total_pick = row.total_pick || (totalEdge !== null && totalEdge >= 0 ? 'OVER' : 'UNDER');
      row.total_status = row.total_status || fallbackStatus(totalEdge, 'total');
      row.moneyline_pick = row.moneyline_pick || row.model_favorite;
      row.model_home_score = num(row.model_home_score);
      row.model_away_score = num(row.model_away_score);

      if (row.model_home_score === null && homeMargin !== null && total !== null) {
        row.model_home_score = (total + homeMargin) / 2;
        row.model_away_score = (total - homeMargin) / 2;
      }

      return row;
    }).sort(function (a, b) {
      return String(a.gameday || '').localeCompare(String(b.gameday || ''));
    });
  }

  function badge(status) {
    var normalized = String(status || 'LEAN').toLowerCase();
    var cls = 'lean';
    if (normalized.indexOf('valid') >= 0) cls = 'validated';
    else if (normalized.indexOf('watch') >= 0) cls = 'watch';
    else if (normalized.indexOf('risk') >= 0 || normalized.indexOf('weak') >= 0 || normalized.indexOf('expensive') >= 0) cls = 'risk';
    else if (normalized.indexOf('research') >= 0) cls = 'research';
    return '<span class="badge ' + cls + '">' + escapeHtml(status || 'LEAN') + '</span>';
  }

  function resultBadge(result) {
    var r = String(result || '').toUpperCase();
    if (r === 'W' || r === 'WIN') return '<span class="badge win">WIN</span>';
    if (r === 'L' || r === 'LOSS') return '<span class="badge loss">LOSS</span>';
    if (r === 'P' || r === 'PUSH') return '<span class="badge push">PUSH</span>';
    return '<span class="badge lean">—</span>';
  }

  function statCard(label, value, meta) {
    return '<div class="stat-card">' +
      '<div class="label">' + escapeHtml(label) + '</div>' +
      '<div class="value">' + escapeHtml(value) + '</div>' +
      '<div class="meta">' + escapeHtml(meta || '') + '</div>' +
      '</div>';
  }

  function renderWeekHeader() {
    if (!state.board.length) {
      $('weekChip').textContent = 'NO BOARD';
      $('weekStats').innerHTML =
        statCard('Games', '0', 'No current model output found') +
        statCard('Spread signals', '—', 'Awaiting data') +
        statCard('Total signals', '—', 'Awaiting data') +
        statCard('Forward record', '—', 'Begins with clean pregame locks');
      return;
    }

    var season = state.board[0].season;
    var week = state.board[0].week;
    var spreadSignals = state.board.filter(function (r) {
      return r.spread_status === 'WATCH' || r.spread_status === 'VALIDATED';
    }).length;
    var totalSignals = state.board.filter(function (r) {
      return r.total_status === 'WATCH' || r.total_status === 'VALIDATED';
    }).length;
    var mlPrices = state.board.filter(function (r) { return num(r.moneyline_price) !== null; }).length;

    $('weekChip').textContent = season + ' • WEEK ' + week;
    $('weekStats').innerHTML =
      statCard('Games', String(state.board.length), 'Full Week ' + week + ' board') +
      statCard('Spread signals', String(spreadSignals), 'WATCH + VALIDATED thresholds') +
      statCard('Total signals', String(totalSignals), 'WATCH + VALIDATED thresholds') +
      statCard('ML prices', String(mlPrices) + '/' + state.board.length, 'Model favorite with available market price');
  }

  function signalRows() {
    var out = [];
    state.board.forEach(function (row) {
      var s = num(row.spread_edge);
      var t = num(row.total_edge);
      if (s !== null) {
        out.push({
          type: 'SPREAD',
          edge: Math.abs(s),
          pick: row.spread_pick + ' ' + signed(row.spread_pick_line, 1),
          game: row.away_team + ' @ ' + row.home_team,
          status: row.spread_status
        });
      }
      if (t !== null) {
        out.push({
          type: 'TOTAL',
          edge: Math.abs(t),
          pick: row.total_pick + ' ' + fmt(row.total_line, 1),
          game: row.away_team + ' @ ' + row.home_team,
          status: row.total_status
        });
      }
    });
    return out.sort(function (a, b) { return b.edge - a.edge; }).slice(0, 3);
  }

  function renderEdgeStrip() {
    var signals = signalRows();
    if (!signals.length) {
      $('edgeStrip').innerHTML = '';
      return;
    }

    $('edgeStrip').innerHTML = signals.map(function (signal, idx) {
      return '<div class="edge-card">' +
        '<div class="edge-rank">' + (idx + 1) + '</div>' +
        '<div><strong>' + escapeHtml(signal.pick) + '</strong>' +
        '<small>' + escapeHtml(signal.game + ' • ' + signal.type) + ' ' + badge(signal.status) + '</small></div>' +
        '<div class="edge-points">' + fmt(signal.edge, 1) + '<span>POINT EDGE</span></div>' +
        '</div>';
    }).join('');
  }

  function candidateMap() {
    var map = {};
    state.candidates.forEach(function (row) {
      map[row.game_id] = row.reason || '';
    });
    return map;
  }

  function spreadCell(row) {
    var edge = num(row.spread_edge);
    var edgeClass = edge !== null && Math.abs(edge) >= 3 ? 'edge-pos' : '';
    return '<div class="pick-line"><span class="metric-main">' +
      escapeHtml(row.spread_pick || '—') + ' ' + escapeHtml(signed(row.spread_pick_line, 1)) +
      '</span>' + badge(row.spread_status) + '</div>' +
      '<div class="metric-sub">Market home margin ' + escapeHtml(signed(row.spread_line, 1)) +
      ' • <span class="' + edgeClass + '">edge ' + escapeHtml(signed(edge, 1)) + '</span>' +
      (num(row.spread_odds) !== null ? ' • ' + escapeHtml(odds(row.spread_odds)) : '') +
      '</div>';
  }

  function totalCell(row) {
    var edge = num(row.total_edge);
    var edgeClass = edge !== null && Math.abs(edge) >= 3 ? 'edge-pos' : '';
    return '<div class="pick-line"><span class="metric-main">' +
      escapeHtml(row.total_pick || '—') + ' ' + escapeHtml(fmt(row.total_line, 1)) +
      '</span>' + badge(row.total_status) + '</div>' +
      '<div class="metric-sub">Model ' + escapeHtml(fmt(row.model_total, 1)) +
      ' • <span class="' + edgeClass + '">edge ' + escapeHtml(signed(edge, 1)) + '</span>' +
      (num(row.total_odds) !== null ? ' • ' + escapeHtml(odds(row.total_odds)) : '') +
      '</div>';
  }

  function renderGameBoard() {
    var tbody = $('gameBoardBody');
    if (!state.board.length) {
      tbody.innerHTML = '<tr><td colspan="6">No current board found.</td></tr>';
      return;
    }

    var notes = candidateMap();
    tbody.innerHTML = state.board.map(function (row) {
      var score = (num(row.model_away_score) !== null && num(row.model_home_score) !== null)
        ? row.away_team + ' ' + fmt(row.model_away_score, 1) + ' – ' + row.home_team + ' ' + fmt(row.model_home_score, 1)
        : 'Margin ' + signed(row.model_home_margin, 1) + ' • Total ' + fmt(row.model_total, 1);
      var mlPrice = num(row.moneyline_price) !== null ? odds(row.moneyline_price) : 'price pending';
      var context = notes[row.game_id] || 'No current WATCH threshold triggered.';

      return '<tr>' +
        '<td><div class="game-name">' + escapeHtml(row.away_team) + ' @ ' + escapeHtml(row.home_team) + '</div>' +
        '<div class="game-date">' + escapeHtml(dateLabel(row.gameday)) + '</div></td>' +
        '<td><div class="metric-main">' + escapeHtml(score) + '</div>' +
        '<div class="metric-sub">Home margin ' + escapeHtml(signed(row.model_home_margin, 1)) + '</div></td>' +
        '<td>' + spreadCell(row) + '</td>' +
        '<td>' + totalCell(row) + '</td>' +
        '<td><div class="metric-main">' + escapeHtml(row.moneyline_pick || row.model_favorite || '—') + '</div>' +
        '<div class="metric-sub">' + escapeHtml(mlPrice) + ' • model winner lean only</div></td>' +
        '<td><div class="context-text">' + escapeHtml(context) + '</div></td>' +
        '</tr>';
    }).join('');
  }

  function uniqueMarkets() {
    var rows = state.propEdges.length ? state.propEdges : state.propProjections;
    var set = {};
    rows.forEach(function (row) {
      if (row.market) set[row.market] = true;
    });
    return Object.keys(set);
  }

  function setupPropFilters() {
    var select = $('propMarketFilter');
    var markets = uniqueMarkets();
    select.innerHTML = '<option value="all">All markets</option>' + markets.map(function (m) {
      return '<option value="' + escapeHtml(m) + '">' + escapeHtml(marketLabel(m)) + '</option>';
    }).join('');
    select.addEventListener('change', renderProps);
    $('propSearch').addEventListener('input', renderProps);
  }

  function renderPropNotice() {
    if (state.propEdges.length) {
      var actionable = state.propEdges.filter(function (row) {
        return String(row.actionability).toUpperCase() === 'RESEARCH';
      }).length;
      $('propNotice').innerHTML = '<div class="notice info"><strong>Live market comparison loaded.</strong> ' +
        escapeHtml(String(state.propEdges.length)) + ' sportsbook quotes are available; ' +
        escapeHtml(String(actionable)) + ' currently clear the pipeline\'s basic research filters. ' +
        'A positive model EV estimate is not the same thing as validated forward ROI.</div>';
    } else {
      var msg = state.propForwardReport.message || 'No live sportsbook prop lines are currently archived.';
      $('propNotice').innerHTML = '<div class="notice"><strong>Projections are not bets yet.</strong> ' +
        escapeHtml(msg) + ' The table below shows football projections only, with no claim that an over or under is mispriced.</div>';
    }
  }

  function renderProps() {
    var usingEdges = state.propEdges.length > 0;
    var rows = usingEdges ? state.propEdges.slice() : state.propProjections.slice();
    var market = $('propMarketFilter').value || 'all';
    var search = ($('propSearch').value || '').trim().toLowerCase();

    if (market !== 'all') {
      rows = rows.filter(function (row) { return row.market === market; });
    }
    if (search) {
      rows = rows.filter(function (row) {
        return [row.player_name, row.team, row.opponent].join(' ').toLowerCase().indexOf(search) >= 0;
      });
    }

    if (usingEdges) {
      rows.sort(function (a, b) {
        return (num(b.model_ev_per_unit) || -999) - (num(a.model_ev_per_unit) || -999);
      });
      $('propTableTitle').textContent = 'Live prop edge board';
      $('propTableSubtitle').textContent = 'One row per sportsbook quote. Sort priority is model estimated EV.';
      $('propTableHead').innerHTML =
        '<tr><th>Player</th><th>Prop</th><th>Line</th><th>Projection</th><th>Lean</th><th>Model P</th><th>Price</th><th>Est. EV</th><th>Status</th></tr>';

      $('propTableBody').innerHTML = rows.slice(0, 200).map(function (row) {
        var price = String(row.lean || '').toUpperCase() === 'OVER' ? row.price_over : row.price_under;
        return '<tr>' +
          '<td><div class="game-name">' + escapeHtml(row.player_name) + '</div><div class="metric-sub">' +
          escapeHtml(row.team + ' vs ' + row.opponent + ' • ' + row.position) + '</div></td>' +
          '<td>' + escapeHtml(marketLabel(row.market)) + '</td>' +
          '<td class="mono">' + escapeHtml(fmt(row.line, 1)) + '</td>' +
          '<td class="mono">' + escapeHtml(fmt(row.projection, 1)) + '<div class="metric-sub">edge ' + escapeHtml(signed(row.edge, 1)) + '</div></td>' +
          '<td class="metric-main">' + escapeHtml(row.lean || '—') + '</td>' +
          '<td class="mono">' + escapeHtml(pct(row.model_lean_probability, 1)) + '</td>' +
          '<td class="mono">' + escapeHtml(odds(price)) + '</td>' +
          '<td class="mono ' + ((num(row.model_ev_per_unit) || 0) > 0 ? 'edge-pos' : 'edge-neg') + '">' + escapeHtml(pct(row.model_ev_per_unit, 1)) + '</td>' +
          '<td>' + badge(row.actionability || 'RESEARCH') + '</td>' +
          '</tr>';
      }).join('');
    } else {
      rows.sort(function (a, b) {
        return (num(b.projection) || 0) - (num(a.projection) || 0);
      });
      $('propTableTitle').textContent = 'Player projections';
      $('propTableSubtitle').textContent = 'No sportsbook line loaded; do not interpret the projection as an over/under recommendation.';
      $('propTableHead').innerHTML =
        '<tr><th>Player</th><th>Prop</th><th>Projection</th><th>Recent baseline</th><th>Usage</th><th>Games</th><th>Availability</th></tr>';

      $('propTableBody').innerHTML = rows.slice(0, 200).map(function (row) {
        var availability = row.availability_flag || 'NOT_LISTED';
        var cls = String(row.availability_ok).toLowerCase() === 'false' ? 'risk' : 'lean';
        return '<tr>' +
          '<td><div class="game-name">' + escapeHtml(row.player_name) + '</div><div class="metric-sub">' +
          escapeHtml(row.team + ' vs ' + row.opponent + ' • ' + row.position) + '</div></td>' +
          '<td>' + escapeHtml(marketLabel(row.market)) + '</td>' +
          '<td class="metric-main mono">' + escapeHtml(fmt(row.projection, 1)) + '</td>' +
          '<td class="mono">' + escapeHtml(fmt(row.recent_baseline, 1)) + '</td>' +
          '<td class="mono">' + escapeHtml(pct(row.usage_share, 1)) + '</td>' +
          '<td class="mono">' + escapeHtml(row.games_current_season || '—') + '</td>' +
          '<td><span class="badge ' + cls + '">' + escapeHtml(availability) + '</span></td>' +
          '</tr>';
      }).join('');
    }

    $('propTableFoot').textContent = 'Showing ' + Math.min(rows.length, 200) + ' of ' + rows.length + ' matching rows.';
  }

  function recordText(summary) {
    if (!summary || !summary.decisions) return '0-0';
    return (summary.wins || 0) + '-' + (summary.losses || 0) + ((summary.pushes || 0) ? '-' + summary.pushes : '');
  }

  function propSummary() {
    var settled = state.propHistory.filter(function (row) {
      var r = String(row.result || '').toUpperCase();
      return r === 'WIN' || r === 'LOSS';
    });
    var wins = settled.filter(function (row) { return String(row.result).toUpperCase() === 'WIN'; }).length;
    var profit = settled.reduce(function (sum, row) {
      return sum + (num(row.unit_profit) || 0);
    }, 0);
    return {
      bets: settled.length,
      wins: wins,
      losses: settled.length - wins,
      hit: settled.length ? wins / settled.length : null,
      roi: settled.length ? profit / settled.length : null
    };
  }

  function renderHistoryStats() {
    var spread = state.historySummary.spread_qualified || {};
    var total = state.historySummary.total_qualified || {};
    var ml = state.historySummary.moneyline_all_model_picks || {};
    var props = propSummary();

    $('historyStats').innerHTML =
      statCard('Qualified spreads', recordText(spread), spread.decisions ? pct(spread.hit_rate, 1) + ' hit • ' + pct(spread.roi_per_bet, 1) + ' ROI' : 'Forward sample not started') +
      statCard('Qualified totals', recordText(total), total.decisions ? pct(total.hit_rate, 1) + ' hit • ' + pct(total.roi_per_bet, 1) + ' ROI' : 'Forward sample not started') +
      statCard('Moneyline picks', recordText(ml), ml.decisions ? pct(ml.hit_rate, 1) + ' hit • ' + pct(ml.roi_per_bet, 1) + ' ROI' : 'Forward sample not started') +
      statCard('Player props', props.bets ? props.wins + '-' + props.losses : '0-0', props.bets ? pct(props.hit, 1) + ' hit • ' + pct(props.roi, 1) + ' ROI' : 'No settled locked prop quotes');
  }

  function historyWeeks(rows) {
    var seen = {};
    rows.forEach(function (row) {
      if (row.season && row.week) seen[row.season + '-' + row.week] = true;
    });
    return Object.keys(seen).sort(function (a, b) {
      var ap = a.split('-').map(Number);
      var bp = b.split('-').map(Number);
      return bp[0] - ap[0] || bp[1] - ap[1];
    });
  }

  function setupHistoryFilter() {
    $('historyWeekFilter').addEventListener('change', renderHistory);
    document.querySelectorAll('[data-history]').forEach(function (button) {
      button.addEventListener('click', function () {
        state.historyType = button.getAttribute('data-history');
        document.querySelectorAll('[data-history]').forEach(function (b) {
          b.classList.toggle('is-active', b === button);
        });
        populateHistoryWeeks();
        renderHistory();
      });
    });
    populateHistoryWeeks();
  }

  function populateHistoryWeeks() {
    var rows = state.historyType === 'games' ? state.gameHistory : state.propHistory;
    var weeks = historyWeeks(rows);
    $('historyWeekFilter').innerHTML = '<option value="all">All weeks</option>' + weeks.map(function (key) {
      var parts = key.split('-');
      return '<option value="' + escapeHtml(key) + '">' + escapeHtml(parts[0] + ' Week ' + parts[1]) + '</option>';
    }).join('');
  }

  function filterHistoryRows(rows) {
    var selected = $('historyWeekFilter').value || 'all';
    if (selected === 'all') return rows.slice();
    return rows.filter(function (row) {
      return row.season + '-' + row.week === selected;
    });
  }

  function gameHistoryTable(rows) {
    if (!rows.length) {
      return '<div class="empty-state"><div class="empty-icon">0–0</div><h3>No forward game record yet</h3>' +
        '<p>The current Week 3 model was refreshed after games had already finished, so it is intentionally excluded. The ledger starts with the first week that can be locked before any result is known.</p></div>';
    }

    return '<div class="table-wrap"><table class="data-table compact">' +
      '<thead><tr><th>Week</th><th>Game</th><th>Final</th><th>Spread</th><th>Total</th><th>Moneyline</th></tr></thead><tbody>' +
      rows.map(function (row) {
        return '<tr>' +
          '<td class="mono">' + escapeHtml(row.season + ' W' + row.week) + '</td>' +
          '<td><strong>' + escapeHtml(row.away_team + ' @ ' + row.home_team) + '</strong><div class="metric-sub">' + escapeHtml(dateLabel(row.gameday)) + '</div></td>' +
          '<td class="mono">' + escapeHtml(row.final_score || '—') + '</td>' +
          '<td><div class="pick-line"><span class="mono">' + escapeHtml((row.spread_pick || '—') + ' ' + signed(row.spread_pick_line, 1)) + '</span>' + resultBadge(row.spread_result) + '</div>' +
          '<div class="metric-sub">' + escapeHtml(row.spread_status || '') + ' • ' + escapeHtml(money(row.spread_profit_units)) + '</div></td>' +
          '<td><div class="pick-line"><span class="mono">' + escapeHtml((row.total_pick || '—') + ' ' + fmt(row.total_line, 1)) + '</span>' + resultBadge(row.total_result) + '</div>' +
          '<div class="metric-sub">' + escapeHtml(row.total_status || '') + ' • ' + escapeHtml(money(row.total_profit_units)) + '</div></td>' +
          '<td><div class="pick-line"><span class="mono">' + escapeHtml((row.moneyline_pick || '—') + ' ' + odds(row.moneyline_price)) + '</span>' + resultBadge(row.moneyline_result) + '</div>' +
          '<div class="metric-sub">' + escapeHtml(money(row.moneyline_profit_units)) + '</div></td>' +
          '</tr>';
      }).join('') +
      '</tbody></table></div>';
  }

  function propHistoryTable(rows) {
    if (!rows.length) {
      return '<div class="empty-state"><div class="empty-icon">PROP</div><h3>No settled locked prop bets yet</h3>' +
        '<p>The model has player projections, but a trustworthy hit rate requires actual sportsbook lines and prices archived before kickoff. The site will populate this table automatically once those quotes exist and settle.</p></div>';
    }

    rows.sort(function (a, b) {
      return Number(b.season) - Number(a.season) || Number(b.week) - Number(a.week);
    });

    return '<div class="table-wrap"><table class="data-table compact">' +
      '<thead><tr><th>Week</th><th>Player</th><th>Market</th><th>Bet</th><th>Projection</th><th>Actual</th><th>Result</th><th>P/L</th></tr></thead><tbody>' +
      rows.map(function (row) {
        return '<tr>' +
          '<td class="mono">' + escapeHtml(row.season + ' W' + row.week) + '</td>' +
          '<td><strong>' + escapeHtml(row.player_name) + '</strong><div class="metric-sub">' + escapeHtml((row.team || '') + (row.opponent ? ' vs ' + row.opponent : '')) + '</div></td>' +
          '<td>' + escapeHtml(marketLabel(row.market)) + '</td>' +
          '<td class="mono">' + escapeHtml((row.lean || '—') + ' ' + fmt(row.line, 1) + ' ' + odds(row.lean_price)) + '</td>' +
          '<td class="mono">' + escapeHtml(fmt(row.projection, 1)) + '</td>' +
          '<td class="mono">' + escapeHtml(fmt(row.actual, 1)) + '</td>' +
          '<td>' + resultBadge(row.result) + '</td>' +
          '<td class="mono">' + escapeHtml(money(row.unit_profit)) + '</td>' +
          '</tr>';
      }).join('') +
      '</tbody></table></div>';
  }

  function renderHistory() {
    var rows = state.historyType === 'games' ? state.gameHistory : state.propHistory;
    rows = filterHistoryRows(rows);
    $('historyContent').innerHTML = state.historyType === 'games' ? gameHistoryTable(rows) : propHistoryTable(rows);
  }

  function renderMethodStats() {
    var independent = state.modelReport.independent_model || {};
    var holdout = state.modelReport.holdout_season || '—';
    var spreadWf = ((state.modelReport.walk_forward_market_validation || {}).spread || {});
    var totalWf = ((state.modelReport.walk_forward_market_validation || {}).total || {});

    $('methodStats').innerHTML =
      statCard('Holdout season', String(holdout), 'Latest completed historical holdout') +
      statCard('Winner accuracy', pct(independent.winner_accuracy, 1), 'Independent football model') +
      statCard('Margin MAE', fmt(independent.margin_mae, 2), 'Independent projected margin error') +
      statCard('Total MAE', fmt(independent.total_mae, 2), 'Independent projected total error');

    var stateEl = $('dataState');
    if (spreadWf.status === 'ok' || totalWf.status === 'ok') stateEl.classList.add('ok');
  }

  function validationBlock(kind, data) {
    var thresholds = data.thresholds || {};
    var keys = Object.keys(thresholds).sort(function (a, b) { return Number(a) - Number(b); });

    if (!keys.length) {
      return '<div class="validation-block"><div class="validation-title"><strong>' + escapeHtml(kind) + '</strong><span class="badge lean">NO TEST</span></div>' +
        '<div class="empty-state"><p>Not enough walk-forward data.</p></div></div>';
    }

    var rows = keys.map(function (key) {
      var r = thresholds[key] || {};
      return '<tr>' +
        '<td class="mono">' + escapeHtml(key) + '+ pts</td>' +
        '<td class="mono">' + escapeHtml(String(r.bets || 0)) + '</td>' +
        '<td class="mono">' + escapeHtml((r.wins || 0) + '-' + (r.losses || 0)) + '</td>' +
        '<td class="mono">' + escapeHtml(pct(r.win_rate, 1)) + '</td>' +
        '<td class="mono">' + escapeHtml(pct(r.roi_at_minus_110, 1)) + '</td>' +
        '<td>' + badge(r.stable ? 'VALIDATED' : 'WATCH') + '</td>' +
        '</tr>';
    }).join('');

    var validated = data.validated_threshold === null || data.validated_threshold === undefined
      ? 'NONE VALIDATED'
      : 'VALIDATED ' + data.validated_threshold + '+';

    return '<div class="validation-block"><div class="validation-title"><strong>' + escapeHtml(kind) + '</strong>' + badge(validated) + '</div>' +
      '<div class="table-wrap"><table class="data-table compact"><thead><tr><th>Edge threshold</th><th>Bets</th><th>Record</th><th>Hit rate</th><th>ROI @ -110</th><th>Gate</th></tr></thead><tbody>' +
      rows + '</tbody></table></div></div>';
  }

  function renderMethod() {
    renderMethodStats();
    var wf = state.modelReport.walk_forward_market_validation || {};
    $('walkForwardTables').innerHTML =
      validationBlock('Spread', wf.spread || {}) +
      validationBlock('Total', wf.total || {});

    var markets = state.propModelReport.markets || {};
    var rows = Object.keys(markets).map(function (key) {
      var r = markets[key] || {};
      return '<tr>' +
        '<td>' + escapeHtml(marketLabel(key)) + '</td>' +
        '<td class="mono">' + escapeHtml(fmt(r.blended_mae, 2)) + '</td>' +
        '<td class="mono">' + escapeHtml(fmt(r.baseline_mae, 2)) + '</td>' +
        '<td class="mono">' + escapeHtml(String(r.oof_rows || 0)) + '</td>' +
        '</tr>';
    }).join('');

    $('playerMethodBody').innerHTML = rows || '<tr><td colspan="4">No player validation report found.</td></tr>';
  }

  function setActiveView(name) {
    var valid = ['this-week', 'player-props', 'history', 'method'];
    if (valid.indexOf(name) < 0) name = 'this-week';

    document.querySelectorAll('.view').forEach(function (view) {
      view.classList.toggle('is-active', view.id === 'view-' + name);
    });
    document.querySelectorAll('.nav-link').forEach(function (button) {
      button.classList.toggle('is-active', button.getAttribute('data-view') === name);
    });

    if (window.location.hash !== '#' + name) {
      history.replaceState(null, '', '#' + name);
    }
  }

  function setupNav() {
    document.querySelectorAll('.nav-link').forEach(function (button) {
      button.addEventListener('click', function () {
        setActiveView(button.getAttribute('data-view'));
      });
    });
    setActiveView((window.location.hash || '#this-week').slice(1));
  }

  async function loadAll() {
    var results = await Promise.all([
      fetchCsv(PATHS.board),
      fetchCsv(PATHS.fallbackBoard),
      fetchCsv(PATHS.candidates),
      fetchCsv(PATHS.propEdges),
      fetchCsv(PATHS.propProjections),
      fetchCsv(PATHS.gameHistory),
      fetchCsv(PATHS.propHistory),
      fetchJson(PATHS.historySummary),
      fetchJson(PATHS.modelReport),
      fetchJson(PATHS.propModelReport),
      fetchJson(PATHS.propForwardReport)
    ]);

    state.modelReport = results[8] || {};
    state.board = normalizeBoard(results[0].length ? results[0] : results[1]);
    state.candidates = results[2] || [];
    state.propEdges = results[3] || [];
    state.propProjections = results[4] || [];
    state.gameHistory = results[5] || [];
    state.propHistory = results[6] || [];
    state.historySummary = results[7] || {};
    state.propModelReport = results[9] || {};
    state.propForwardReport = results[10] || {};

    renderWeekHeader();
    renderEdgeStrip();
    renderGameBoard();
    renderPropNotice();
    setupPropFilters();
    renderProps();
    renderHistoryStats();
    setupHistoryFilter();
    renderHistory();
    renderMethod();

    var generated = state.modelReport.generated_at_utc || state.propModelReport.generated_at_utc || '';
    $('generatedAt').textContent = generated ? 'Model refresh: ' + generated.replace('T', ' ').replace('Z', ' UTC') : '';

    var dataState = $('dataState');
    dataState.classList.add(state.board.length ? 'ok' : 'warn');
    dataState.querySelector('span:last-child').textContent = state.board.length ? 'Model data loaded' : 'Partial data';
  }

  setupNav();
  loadAll().catch(function (error) {
    var dataState = $('dataState');
    dataState.classList.add('warn');
    dataState.querySelector('span:last-child').textContent = 'Data load error';
    console.error(error);
  });
})();