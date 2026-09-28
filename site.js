(function () {
  'use strict';

  const PATHS = {
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

  const state = {
    board: [],
    props: [],
    propEdges: [],
    gameHistory: [],
    propHistory: [],
    historySummary: {},
    modelReport: {},
    propReport: {},
    propForward: {},
    historyType: 'games'
  };

  const $ = (id) => document.getElementById(id);

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function num(value) {
    if (value === null || value === undefined || value === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function fmt(value, digits = 1) {
    const n = num(value);
    return n === null ? '—' : n.toFixed(digits);
  }

  function signed(value, digits = 1) {
    const n = num(value);
    if (n === null) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(digits);
  }

  function pct(value, digits = 1) {
    const n = num(value);
    return n === null ? '—' : (n * 100).toFixed(digits) + '%';
  }

  function odds(value) {
    const n = num(value);
    if (n === null) return '—';
    return n > 0 ? '+' + Math.round(n) : String(Math.round(n));
  }

  function titleCase(value) {
    return String(value || '—')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function dateLabel(value) {
    if (!value) return '—';
    const date = new Date(value + 'T12:00:00');
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  function parseCsv(text) {
    text = String(text || '').replace(/^\uFEFF/, '');
    if (!text.trim()) return [];

    const rows = [];
    let row = [];
    let field = '';
    let quoted = false;

    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];

      if (quoted) {
        if (char === '"' && text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else if (char === '"') {
          quoted = false;
        } else {
          field += char;
        }
      } else if (char === '"') {
        quoted = true;
      } else if (char === ',') {
        row.push(field);
        field = '';
      } else if (char === '\n') {
        row.push(field.replace(/\r$/, ''));
        rows.push(row);
        row = [];
        field = '';
      } else {
        field += char;
      }
    }

    if (field.length || row.length) {
      row.push(field.replace(/\r$/, ''));
      rows.push(row);
    }

    const clean = rows.filter((r) => r.some((cell) => String(cell).trim() !== ''));
    if (clean.length < 2) return [];

    const headers = clean.shift().map((h) => h.trim());

    return clean.map((r) => {
      const obj = {};
      headers.forEach((header, index) => {
        obj[header] = r[index] == null ? '' : r[index];
      });
      return obj;
    });
  }

  async function fetchText(path) {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) throw new Error(path + ' returned ' + response.status);
    return response.text();
  }

  async function fetchCsv(path) {
    try {
      return parseCsv(await fetchText(path));
    } catch (error) {
      console.warn(error.message);
      return [];
    }
  }

  async function fetchJson(path) {
    try {
      return JSON.parse(await fetchText(path));
    } catch (error) {
      console.warn(error.message);
      return {};
    }
  }

  function isFinal(row) {
    return num(row.home_score) !== null && num(row.away_score) !== null;
  }

  function modelMargin(row) {
    const margin = num(row.model_home_margin);
    if (margin === null) return '—';
    const team = margin >= 0 ? row.home_team : row.away_team;
    return team + ' by ' + Math.abs(margin).toFixed(1);
  }

  function marketSpread(row) {
    const line = num(row.spread_line);
    if (line === null) return '—';
    if (line === 0) return 'PK';
    const favorite = line > 0 ? row.home_team : row.away_team;
    return favorite + ' -' + Math.abs(line).toFixed(1);
  }

  function projectedScore(row) {
    const home = num(row.model_home_score);
    const away = num(row.model_away_score);
    if (home === null || away === null) return '—';
    return row.away_team + ' ' + Math.round(away) + ' – ' + row.home_team + ' ' + Math.round(home);
  }

  function finalScore(row) {
    const home = num(row.home_score);
    const away = num(row.away_score);
    if (home === null || away === null) return '—';
    return row.away_team + ' ' + Math.round(away) + ' – ' + row.home_team + ' ' + Math.round(home);
  }

  function statusText(value) {
    const status = String(value || '—').toUpperCase();
    const muted = status === 'LEAN' || status === '—' || status === 'NOT_LISTED' || status === 'NOT LISTED';
    return '<span class="status-text' + (muted ? ' muted' : '') + '">' + esc(status.replace(/_/g, ' ')) + '</span>';
  }

  function summaryCard(label, value, meta) {
    return '<div class="summary-card">' +
      '<div class="summary-label">' + esc(label) + '</div>' +
      '<div class="summary-value">' + esc(value) + '</div>' +
      '<div class="summary-meta">' + esc(meta || '') + '</div>' +
      '</div>';
  }

  function renderWeekSummary() {
    const upcoming = state.board.filter((row) => !isFinal(row));
    const completed = state.board.filter(isFinal);
    const spreadSignals = upcoming.filter((row) => String(row.spread_status).toUpperCase() !== 'LEAN').length;
    const totalSignals = upcoming.filter((row) => String(row.total_status).toUpperCase() !== 'LEAN').length;

    const maxEdge = upcoming.reduce((best, row) => {
      return Math.max(best, Math.abs(num(row.spread_edge) || 0), Math.abs(num(row.total_edge) || 0));
    }, 0);

    $('weekSummary').innerHTML =
      summaryCard('Upcoming', String(upcoming.length), completed.length + ' completed in current file') +
      summaryCard('Spread Signals', String(spreadSignals), 'WATCH / VALIDATED') +
      summaryCard('Total Signals', String(totalSignals), 'WATCH / VALIDATED') +
      summaryCard('Largest Edge', maxEdge ? maxEdge.toFixed(1) + ' pts' : '—', 'Current upcoming games');

    if (state.board.length) {
      $('weekLabel').textContent = state.board[0].season + ' · WEEK ' + Number(state.board[0].week);
    }
  }

  function filteredUpcoming() {
    const filter = $('weekFilter').value || 'all';
    let rows = state.board.filter((row) => !isFinal(row));

    if (filter === 'watch') {
      rows = rows.filter((row) => {
        return ['WATCH', 'VALIDATED'].includes(String(row.spread_status).toUpperCase()) ||
          ['WATCH', 'VALIDATED'].includes(String(row.total_status).toUpperCase());
      });
    } else if (filter === 'spread') {
      rows = rows.filter((row) => Math.abs(num(row.spread_edge) || 0) >= 1.5);
    } else if (filter === 'total') {
      rows = rows.filter((row) => Math.abs(num(row.total_edge) || 0) >= 1.5);
    }

    return rows.sort((a, b) => String(a.gameday).localeCompare(String(b.gameday)));
  }

  function renderUpcoming() {
    const rows = filteredUpcoming();

    $('upcomingEmpty').hidden = rows.length > 0;
    $('upcomingBody').innerHTML = rows.map((row) => {
      return '<tr>' +
        '<td><span class="game-main">' + esc(row.away_team + ' @ ' + row.home_team) + '</span><span class="game-sub">' + esc(dateLabel(row.gameday)) + '</span></td>' +
        '<td class="number">' + esc(projectedScore(row)) + '</td>' +
        '<td class="number">' + esc(modelMargin(row)) + '</td>' +
        '<td class="number">' + esc(marketSpread(row)) + '</td>' +
        '<td class="number strong">' + esc((row.spread_pick || '—') + ' ' + signed(row.spread_pick_line, 1)) + '</td>' +
        '<td class="number">' + esc(signed(row.spread_edge, 1)) + '</td>' +
        '<td>' + statusText(row.spread_status) + '</td>' +
        '<td class="number">' + esc(fmt(row.total_line, 1)) + '</td>' +
        '<td class="number">' + esc(fmt(row.model_total, 1)) + '</td>' +
        '<td class="number strong">' + esc((row.total_pick || '—') + ' ' + fmt(row.total_line, 1)) + '</td>' +
        '<td class="number">' + esc(signed(row.total_edge, 1)) + '</td>' +
        '<td class="number strong">' + esc(row.moneyline_pick || '—') + '</td>' +
        '<td class="number">' + esc(odds(row.moneyline_price)) + '</td>' +
        '</tr>';
    }).join('');
  }

  function spreadResult(row) {
    const pick = String(row.spread_pick || '').toUpperCase();
    const line = num(row.spread_pick_line);
    const home = num(row.home_score);
    const away = num(row.away_score);

    if (!pick || line === null || home === null || away === null) return 'push';

    let pickedScore;
    let opponentScore;

    if (pick === String(row.home_team).toUpperCase()) {
      pickedScore = home;
      opponentScore = away;
    } else if (pick === String(row.away_team).toUpperCase()) {
      pickedScore = away;
      opponentScore = home;
    } else {
      return 'push';
    }

    const coveredBy = pickedScore + line - opponentScore;
    if (coveredBy > 0) return 'win';
    if (coveredBy < 0) return 'loss';
    return 'push';
  }

  function totalResult(row) {
    const pick = String(row.total_pick || '').toUpperCase();
    const line = num(row.total_line);
    const home = num(row.home_score);
    const away = num(row.away_score);

    if (!pick || line === null || home === null || away === null) return 'push';

    const total = home + away;
    if (total === line) return 'push';
    if (pick === 'OVER') return total > line ? 'win' : 'loss';
    if (pick === 'UNDER') return total < line ? 'win' : 'loss';
    return 'push';
  }

  function moneylineResult(row) {
    const pick = String(row.moneyline_pick || '').toUpperCase();
    const home = num(row.home_score);
    const away = num(row.away_score);

    if (!pick || home === null || away === null || home === away) return 'push';

    const winner = home > away ? String(row.home_team).toUpperCase() : String(row.away_team).toUpperCase();
    return pick === winner ? 'win' : 'loss';
  }

  function resultClass(result) {
    if (result === 'win') return ' result-win';
    if (result === 'loss') return ' result-loss';
    return ' result-push';
  }

  function completedRowsHtml(rows) {
    return rows.map((row) => {
      const spreadGrade = spreadResult(row);
      const totalGrade = totalResult(row);
      const moneylineGrade = moneylineResult(row);

      return '<tr>' +
        '<td><span class="game-main">' + esc(row.away_team + ' @ ' + row.home_team) + '</span><span class="game-sub">' + esc(dateLabel(row.gameday)) + '</span></td>' +
        '<td class="number">' + esc(finalScore(row)) + '</td>' +
        '<td class="number">' + esc(projectedScore(row)) + '</td>' +
        '<td class="number' + resultClass(spreadGrade) + '">' + esc((row.spread_pick || '—') + ' ' + signed(row.spread_pick_line, 1)) + '</td>' +
        '<td class="number">' + esc(signed(row.spread_edge, 1)) + '</td>' +
        '<td class="number' + resultClass(totalGrade) + '">' + esc((row.total_pick || '—') + ' ' + fmt(row.total_line, 1)) + '</td>' +
        '<td class="number">' + esc(signed(row.total_edge, 1)) + '</td>' +
        '<td class="number' + resultClass(moneylineGrade) + '">' + esc((row.moneyline_pick || '—') + ' ' + odds(row.moneyline_price)) + '</td>' +
        '</tr>';
    }).join('');
  }

  function renderCompleted() {
    const rows = state.board
      .filter(isFinal)
      .sort((a, b) => String(b.gameday).localeCompare(String(a.gameday)));

    $('completedSection').hidden = rows.length === 0;
    $('completedBody').innerHTML = completedRowsHtml(rows);
  }

  function propMarkets() {
    const source = state.propEdges.length ? state.propEdges : state.props;
    return Array.from(new Set(source.map((row) => row.market).filter(Boolean))).sort();
  }

  function setupPropMarkets() {
    $('propMarket').innerHTML =
      '<option value="all">All markets</option>' +
      propMarkets().map((market) => '<option value="' + esc(market) + '">' + esc(titleCase(market)) + '</option>').join('');
  }

  function renderPropCallout() {
    if (state.propEdges.length) {
      $('propCallout').innerHTML =
        '<strong>Live prop lines loaded.</strong> The table compares the model with sportsbook lines and prices.';
    } else {
      const message = state.propForward.message || 'No sportsbook prop lines are currently archived.';
      $('propCallout').innerHTML =
        '<strong>Projection mode.</strong> ' + esc(message) + ' These rows are projections, not over/under recommendations.';
    }
  }

  function filteredProps() {
    const source = state.propEdges.length ? state.propEdges : state.props;
    const market = $('propMarket').value || 'all';
    const search = ($('propSearch').value || '').trim().toLowerCase();

    return source.filter((row) => {
      const marketMatch = market === 'all' || row.market === market;
      const searchMatch = !search || [row.player_name, row.team, row.opponent, row.position]
        .join(' ')
        .toLowerCase()
        .includes(search);

      return marketMatch && searchMatch;
    });
  }

  function renderProps() {
    const live = state.propEdges.length > 0;
    let rows = filteredProps();

    if (live) {
      rows.sort((a, b) => (num(b.model_ev_per_unit) || -999) - (num(a.model_ev_per_unit) || -999));

      $('propTitle').textContent = 'Prop Market Comparison';
      $('propSubtitle').textContent = 'Model projection compared with the current sportsbook quote.';
      $('propHead').innerHTML =
        '<tr><th>Player</th><th>Team / Opp</th><th>Market</th><th>Line</th><th>Projection</th><th>Edge</th><th>Lean</th><th>Model P</th><th>Price</th><th>Est. EV</th><th>Status</th></tr>';

      $('propBody').innerHTML = rows.slice(0, 250).map((row) => {
        const price = String(row.lean || '').toUpperCase() === 'OVER' ? row.price_over : row.price_under;

        return '<tr>' +
          '<td><span class="game-main">' + esc(row.player_name) + '</span><span class="game-sub">' + esc(row.position || '') + '</span></td>' +
          '<td>' + esc((row.team || '—') + (row.opponent ? ' vs ' + row.opponent : '')) + '</td>' +
          '<td>' + esc(titleCase(row.market)) + '</td>' +
          '<td class="number">' + esc(fmt(row.line, 1)) + '</td>' +
          '<td class="number strong">' + esc(fmt(row.projection, 1)) + '</td>' +
          '<td class="number">' + esc(signed(row.edge, 1)) + '</td>' +
          '<td class="number strong">' + esc(row.lean || '—') + '</td>' +
          '<td class="number">' + esc(pct(row.model_lean_probability, 1)) + '</td>' +
          '<td class="number">' + esc(odds(price)) + '</td>' +
          '<td class="number">' + esc(pct(row.model_ev_per_unit, 1)) + '</td>' +
          '<td>' + statusText(row.actionability || 'RESEARCH') + '</td>' +
          '</tr>';
      }).join('');
    } else {
      rows.sort((a, b) => String(a.market).localeCompare(String(b.market)) || String(a.player_name).localeCompare(String(b.player_name)));

      $('propTitle').textContent = 'Player Projections';
      $('propSubtitle').textContent = 'Model projections by player and market.';
      $('propHead').innerHTML =
        '<tr><th>Player</th><th>Team / Opp</th><th>Market</th><th>Projection</th><th>Recent Baseline</th><th>Difference</th><th>Games</th><th>Availability</th></tr>';

      $('propBody').innerHTML = rows.slice(0, 250).map((row) => {
        const projection = num(row.projection);
        const baseline = num(row.recent_baseline);
        const difference = projection !== null && baseline !== null ? projection - baseline : null;

        return '<tr>' +
          '<td><span class="game-main">' + esc(row.player_name) + '</span><span class="game-sub">' + esc(row.position || '') + '</span></td>' +
          '<td>' + esc((row.team || '—') + (row.opponent ? ' vs ' + row.opponent : '')) + '</td>' +
          '<td>' + esc(titleCase(row.market)) + '</td>' +
          '<td class="number strong">' + esc(fmt(row.projection, 1)) + '</td>' +
          '<td class="number">' + esc(fmt(row.recent_baseline, 1)) + '</td>' +
          '<td class="number">' + esc(signed(difference, 1)) + '</td>' +
          '<td class="number">' + esc(row.games_current_season || '—') + '</td>' +
          '<td>' + statusText(row.availability_flag || 'NOT LISTED') + '</td>' +
          '</tr>';
      }).join('');
    }

    const shown = Math.min(rows.length, 250);
    $('propCount').textContent = 'Showing ' + shown + ' of ' + rows.length + ' matching rows';
    $('propEmpty').hidden = rows.length > 0;
  }

  function recordText(summary) {
    if (!summary || !num(summary.decisions)) return '0-0';
    return String(summary.wins || 0) + '-' + String(summary.losses || 0) +
      ((summary.pushes || 0) ? '-' + String(summary.pushes) : '');
  }

  function propHistoryStats() {
    const settled = state.propHistory.filter((row) => ['WIN', 'LOSS'].includes(String(row.result).toUpperCase()));
    const wins = settled.filter((row) => String(row.result).toUpperCase() === 'WIN').length;
    const profit = settled.reduce((sum, row) => sum + (num(row.unit_profit) || 0), 0);

    return {
      decisions: settled.length,
      wins,
      losses: settled.length - wins,
      hitRate: settled.length ? wins / settled.length : null,
      roi: settled.length ? profit / settled.length : null
    };
  }

  function completedGradeStats(grader) {
    const grades = state.board.filter(isFinal).map(grader);
    const wins = grades.filter((grade) => grade === 'win').length;
    const losses = grades.filter((grade) => grade === 'loss').length;
    const pushes = grades.filter((grade) => grade === 'push').length;
    const decisions = wins + losses;

    return {
      wins,
      losses,
      pushes,
      decisions,
      hitRate: decisions ? wins / decisions : null
    };
  }

  function completedRecordText(stats) {
    if (!stats || (!stats.wins && !stats.losses && !stats.pushes)) return '0-0';
    return stats.wins + '-' + stats.losses + (stats.pushes ? '-' + stats.pushes : '');
  }

  function renderHistorySummary() {
    const spread = completedGradeStats(spreadResult);
    const total = completedGradeStats(totalResult);
    const moneyline = completedGradeStats(moneylineResult);
    const props = propHistoryStats();

    $('historySummary').innerHTML =
      summaryCard('Spread', completedRecordText(spread), spread.decisions ? pct(spread.hitRate, 1) + ' hit · completed model games' : 'No completed model games') +
      summaryCard('Total', completedRecordText(total), total.decisions ? pct(total.hitRate, 1) + ' hit · completed model games' : 'No completed model games') +
      summaryCard('Moneyline', completedRecordText(moneyline), moneyline.decisions ? pct(moneyline.hitRate, 1) + ' hit · completed model games' : 'No completed model games') +
      summaryCard('Player Props', props.decisions ? props.wins + '-' + props.losses : '0-0', props.decisions ? pct(props.hitRate, 1) + ' hit rate' : 'No settled locked props');
  }

  function historyWeeks(rows) {
    const set = new Set();
    rows.forEach((row) => {
      if (row.season && row.week) set.add(row.season + '-' + row.week);
    });

    return Array.from(set).sort((a, b) => {
      const aa = a.split('-').map(Number);
      const bb = b.split('-').map(Number);
      return bb[0] - aa[0] || bb[1] - aa[1];
    });
  }

  function populateHistoryWeeks() {
    const rows = state.historyType === 'games'
      ? state.gameHistory.concat(state.board.filter(isFinal))
      : state.propHistory;

    $('historyWeek').innerHTML =
      '<option value="all">All weeks</option>' +
      historyWeeks(rows).map((key) => {
        const [season, week] = key.split('-');
        return '<option value="' + esc(key) + '">' + esc(season + ' Week ' + Number(week)) + '</option>';
      }).join('');
  }

  function filteredHistoryRows() {
    const rows = state.historyType === 'games' ? state.gameHistory : state.propHistory;
    const selected = $('historyWeek').value || 'all';

    if (selected === 'all') return rows.slice();
    return rows.filter((row) => row.season + '-' + row.week === selected);
  }

  function filteredCompletedHistoryRows() {
    const selected = $('historyWeek').value || 'all';
    let rows = state.board
      .filter(isFinal)
      .sort((a, b) => String(b.gameday).localeCompare(String(a.gameday)));

    if (selected !== 'all') {
      rows = rows.filter((row) => row.season + '-' + row.week === selected);
    }

    return rows;
  }

  function emptyHistory(title, text) {
    return '<div class="empty-state"><strong>' + esc(title) + '</strong><br>' + esc(text) + '</div>';
  }

  function gameHistoryTable(rows) {
    if (!rows.length) {
      return emptyHistory(
        'No forward game bets yet.',
        'A result only appears here if the prediction was locked before kickoff.'
      );
    }

    return '<div class="table-scroll"><table class="data-table">' +
      '<thead><tr><th>Week</th><th>Game</th><th>Final</th><th>Spread Bet</th><th>Spread Result</th><th>Total Bet</th><th>Total Result</th><th>ML Bet</th><th>ML Result</th></tr></thead>' +
      '<tbody>' +
      rows.map((row) => {
        return '<tr>' +
          '<td class="number">' + esc(row.season + ' W' + Number(row.week)) + '</td>' +
          '<td><span class="game-main">' + esc(row.away_team + ' @ ' + row.home_team) + '</span><span class="game-sub">' + esc(dateLabel(row.gameday)) + '</span></td>' +
          '<td class="number">' + esc(row.final_score || '—') + '</td>' +
          '<td class="number">' + esc((row.spread_pick || '—') + ' ' + signed(row.spread_pick_line, 1)) + '</td>' +
          '<td>' + statusText(row.spread_result) + '</td>' +
          '<td class="number">' + esc((row.total_pick || '—') + ' ' + fmt(row.total_line, 1)) + '</td>' +
          '<td>' + statusText(row.total_result) + '</td>' +
          '<td class="number">' + esc((row.moneyline_pick || '—') + ' ' + odds(row.moneyline_price)) + '</td>' +
          '<td>' + statusText(row.moneyline_result) + '</td>' +
          '</tr>';
      }).join('') +
      '</tbody></table></div>';
  }

  function propHistoryTable(rows) {
    if (!rows.length) {
      return emptyHistory(
        'No settled player props yet.',
        'A result only appears here after a sportsbook line was archived before kickoff and later settled.'
      );
    }

    return '<div class="table-scroll"><table class="data-table">' +
      '<thead><tr><th>Week</th><th>Player</th><th>Market</th><th>Bet</th><th>Projection</th><th>Actual</th><th>Result</th><th>Units</th></tr></thead>' +
      '<tbody>' +
      rows.map((row) => {
        return '<tr>' +
          '<td class="number">' + esc(row.season + ' W' + Number(row.week)) + '</td>' +
          '<td><span class="game-main">' + esc(row.player_name) + '</span><span class="game-sub">' + esc((row.team || '') + (row.opponent ? ' vs ' + row.opponent : '')) + '</span></td>' +
          '<td>' + esc(titleCase(row.market)) + '</td>' +
          '<td class="number">' + esc((row.lean || '—') + ' ' + fmt(row.line, 1) + ' ' + odds(row.lean_price)) + '</td>' +
          '<td class="number">' + esc(fmt(row.projection, 1)) + '</td>' +
          '<td class="number">' + esc(fmt(row.actual, 1)) + '</td>' +
          '<td>' + statusText(row.result) + '</td>' +
          '<td class="number">' + esc(signed(row.unit_profit, 2)) + '</td>' +
          '</tr>';
      }).join('') +
      '</tbody></table></div>';
  }

  function completedHistoryTable(rows) {
    if (!rows.length) {
      return emptyHistory(
        'No completed model games for this week.',
        'Completed model results will appear here after games finish.'
      );
    }

    return '<div class="table-count"><strong>Completed Model Games</strong> · Green = correct, red = incorrect, gray = push.</div>' +
      '<div class="table-scroll"><table class="data-table">' +
      '<thead><tr><th>Week</th><th>Game</th><th>Final Score</th><th>Projected Score</th><th>Spread Pick</th><th>Spread Edge</th><th>Total Pick</th><th>Total Edge</th><th>ML Pick</th></tr></thead>' +
      '<tbody>' +
      rows.map((row) => {
        const spreadGrade = spreadResult(row);
        const totalGrade = totalResult(row);
        const moneylineGrade = moneylineResult(row);

        return '<tr>' +
          '<td class="number">' + esc(row.season + ' W' + Number(row.week)) + '</td>' +
          '<td><span class="game-main">' + esc(row.away_team + ' @ ' + row.home_team) + '</span><span class="game-sub">' + esc(dateLabel(row.gameday)) + '</span></td>' +
          '<td class="number">' + esc(finalScore(row)) + '</td>' +
          '<td class="number">' + esc(projectedScore(row)) + '</td>' +
          '<td class="number' + resultClass(spreadGrade) + '">' + esc((row.spread_pick || '—') + ' ' + signed(row.spread_pick_line, 1)) + '</td>' +
          '<td class="number">' + esc(signed(row.spread_edge, 1)) + '</td>' +
          '<td class="number' + resultClass(totalGrade) + '">' + esc((row.total_pick || '—') + ' ' + fmt(row.total_line, 1)) + '</td>' +
          '<td class="number">' + esc(signed(row.total_edge, 1)) + '</td>' +
          '<td class="number' + resultClass(moneylineGrade) + '">' + esc((row.moneyline_pick || '—') + ' ' + odds(row.moneyline_price)) + '</td>' +
          '</tr>';
      }).join('') +
      '</tbody></table></div>';
  }

  function renderHistory() {
    const rows = filteredHistoryRows();

    if (state.historyType === 'games') {
      const completedRows = filteredCompletedHistoryRows();
      let html = completedHistoryTable(completedRows);

      if (rows.length) {
        html += '<div class="table-count" style="margin-top:20px"><strong>Audited Forward Bet Record</strong> · Only pre-kickoff locked predictions.</div>' +
          gameHistoryTable(rows);
      } else {
        html += '<div class="table-count" style="margin-top:14px">Audited forward record: no pre-kickoff locked settled bets yet.</div>';
      }

      $('historyContent').innerHTML = html;
    } else {
      $('historyContent').innerHTML = propHistoryTable(rows);
    }
  }

  function renderModelSummary() {
    const independent = state.modelReport.independent_model || {};

    $('modelSummary').innerHTML =
      summaryCard('Holdout Season', String(state.modelReport.holdout_season || '—'), 'Latest completed holdout') +
      summaryCard('Winner Accuracy', pct(independent.winner_accuracy, 1), 'Independent model') +
      summaryCard('Margin MAE', fmt(independent.margin_mae, 2), 'Points') +
      summaryCard('Total MAE', fmt(independent.total_mae, 2), 'Points');
  }

  function validationTable(data) {
    const thresholds = (data && data.thresholds) || {};
    const keys = Object.keys(thresholds).sort((a, b) => Number(a) - Number(b));

    return '<table class="data-table">' +
      '<thead><tr><th>Edge Threshold</th><th>Bets</th><th>Record</th><th>Hit Rate</th><th>ROI @ -110</th><th>Status</th></tr></thead>' +
      '<tbody>' +
      (keys.length ? keys.map((key) => {
        const row = thresholds[key] || {};
        return '<tr>' +
          '<td class="number">' + esc(key + '+ pts') + '</td>' +
          '<td class="number">' + esc(row.bets || 0) + '</td>' +
          '<td class="number">' + esc((row.wins || 0) + '-' + (row.losses || 0)) + '</td>' +
          '<td class="number">' + esc(pct(row.win_rate, 1)) + '</td>' +
          '<td class="number">' + esc(pct(row.roi_at_minus_110, 1)) + '</td>' +
          '<td>' + statusText(row.stable ? 'VALIDATED' : 'WATCH') + '</td>' +
          '</tr>';
      }).join('') : '<tr><td colspan="6">No walk-forward data available.</td></tr>') +
      '</tbody></table>';
  }

  function renderModelValidation() {
    const walkForward = state.modelReport.walk_forward_market_validation || {};
    $('spreadValidation').innerHTML = validationTable(walkForward.spread || {});
    $('totalValidation').innerHTML = validationTable(walkForward.total || {});

    const markets = state.propReport.markets || {};
    const keys = Object.keys(markets).sort();

    $('playerValidationBody').innerHTML = keys.length ? keys.map((key) => {
      const row = markets[key] || {};
      const modelMae = num(row.blended_mae != null ? row.blended_mae : row.model_mae);
      const baselineMae = num(row.baseline_mae);
      const difference = modelMae !== null && baselineMae !== null ? modelMae - baselineMae : null;

      return '<tr>' +
        '<td>' + esc(titleCase(key)) + '</td>' +
        '<td class="number">' + esc(fmt(modelMae, 2)) + '</td>' +
        '<td class="number">' + esc(fmt(baselineMae, 2)) + '</td>' +
        '<td class="number">' + esc(signed(difference, 2)) + '</td>' +
        '<td class="number">' + esc(row.oof_rows || 0) + '</td>' +
        '</tr>';
    }).join('') : '<tr><td colspan="5">No player validation report available.</td></tr>';
  }

  function setRoute(route) {
    const valid = ['week', 'props', 'history', 'model'];
    if (!valid.includes(route)) route = 'week';

    document.querySelectorAll('.page').forEach((page) => {
      page.classList.toggle('is-active', page.id === 'page-' + route);
    });

    document.querySelectorAll('[data-route]').forEach((button) => {
      button.classList.toggle('is-active', button.getAttribute('data-route') === route);
    });

    if (window.location.hash !== '#' + route) {
      history.replaceState(null, '', '#' + route);
    }

    window.scrollTo(0, 0);
  }

  function bindEvents() {
    document.querySelectorAll('[data-route]').forEach((element) => {
      element.addEventListener('click', (event) => {
        event.preventDefault();
        setRoute(element.getAttribute('data-route'));
      });
    });

    $('weekFilter').addEventListener('change', renderUpcoming);
    $('propMarket').addEventListener('change', renderProps);
    $('propSearch').addEventListener('input', renderProps);
    $('historyWeek').addEventListener('change', renderHistory);

    document.querySelectorAll('[data-history]').forEach((button) => {
      button.addEventListener('click', () => {
        state.historyType = button.getAttribute('data-history');

        document.querySelectorAll('[data-history]').forEach((item) => {
          item.classList.toggle('is-active', item === button);
        });

        populateHistoryWeeks();
        renderHistory();
      });
    });
  }

  async function load() {
    const [
      board,
      props,
      propEdges,
      gameHistory,
      propHistory,
      historySummary,
      modelReport,
      propReport,
      propForward
    ] = await Promise.all([
      fetchCsv(PATHS.board),
      fetchCsv(PATHS.props),
      fetchCsv(PATHS.propEdges),
      fetchCsv(PATHS.gameHistory),
      fetchCsv(PATHS.propHistory),
      fetchJson(PATHS.historySummary),
      fetchJson(PATHS.modelReport),
      fetchJson(PATHS.propReport),
      fetchJson(PATHS.propForward)
    ]);

    state.board = board;
    state.props = props;
    state.propEdges = propEdges;
    state.gameHistory = gameHistory;
    state.propHistory = propHistory;
    state.historySummary = historySummary;
    state.modelReport = modelReport;
    state.propReport = propReport;
    state.propForward = propForward;

    renderWeekSummary();
    renderUpcoming();
    renderCompleted();

    setupPropMarkets();
    renderPropCallout();
    renderProps();

    renderHistorySummary();
    populateHistoryWeeks();
    renderHistory();

    renderModelSummary();
    renderModelValidation();

    const generatedAt = state.propReport.generated_at_utc || state.modelReport.generated_at_utc || '';
    $('refreshStamp').textContent = generatedAt
      ? 'Data refresh: ' + String(generatedAt).replace('T', ' ').replace('+00:00', ' UTC').replace('Z', ' UTC')
      : '';

    $('loadState').classList.add('ok');
    $('loadState').textContent = 'Data loaded';
  }

  bindEvents();
  setRoute((window.location.hash || '#week').slice(1));

  load().catch((error) => {
    console.error(error);
    $('loadState').classList.add('error');
    $('loadState').textContent = 'Data error';
  });
})();
