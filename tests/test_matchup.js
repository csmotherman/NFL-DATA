const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const m = require('../matchup.js');

function fixture() {
  return Array.from({ length: 32 }, (_, i) => ({
    team: 'T' + i,
    season: '2026',
    games: '7',
    adj_off_epa_per_play: String(i / 100),
    adj_def_epa_per_play_allowed: String(i / 100),
    adj_off_epa_per_pass: String(i / 90),
    adj_def_epa_per_pass_allowed: String(i / 90),
    adj_off_epa_per_rush: String(i / 120),
    adj_def_epa_per_rush_allowed: String(i / 120),
    pre_off_sack_rate: String(i / 1000),
    pre_def_sack_rate_allowed: String(i / 1000)
  }));
}

test('uses empirical midrank percentiles and finite sample standard deviation', () => {
  const d = m.distribution(fixture(), 'adj_off_epa_per_play', 1);
  assert.equal(d.n, 32);
  assert.equal(m.percentile(0.31, d), 98.4);
  assert.equal(m.percentile(0, d), 1.6);
  assert.equal(m.number(''), null);
  assert.equal(m.number('NaN'), null);
});

test('strong offense against permissive defense produces positive opportunity index', () => {
  const metric = m.evaluate(fixture(), 'T31', 'T30', m.EPA[0]);
  assert.equal(metric.offensePercentile, 98.4);
  assert.ok(metric.defenseVulnerabilityPercentile > 90);
  assert.ok(metric.index > 1);
  assert.equal(m.assessment(metric.index).tone, 'positive');
});

test('strong defense allowed lowers the offense-friendly index', () => {
  const metric = m.evaluate(fixture(), 'T31', 'T0', m.EPA[0]);
  assert.ok(Math.abs(metric.index) < 0.1);
  assert.ok(metric.defenseVulnerabilityZ < -1);
  assert.equal(m.assessment(metric.index).tone, 'neutral');
});

test('sack rates orient correctly: lower allowed sack rate means easier pressure matchup', () => {
  const metric = m.evaluate(fixture(), 'T0', 'T1', m.CONTEXT.find((x) => x.id === 'sack'));
  assert.ok(metric.index > 1);
  const hard = m.evaluate(fixture(), 'T31', 'T30', m.CONTEXT.find((x) => x.id === 'sack'));
  assert.ok(hard.index < -1);
});

test('missing metrics or degenerate distributions never create artificial edges', () => {
  const rows = fixture();
  rows[0].adj_off_epa_per_play = '';
  assert.equal(m.evaluate(rows, 'T0', 'T1', m.EPA[0]), null);
  rows.forEach((r) => { r.adj_off_epa_per_play = '0.5'; });
  assert.equal(m.evaluate(rows, 'T1', 'T2', m.EPA[0]), null);
  assert.equal(m.evaluate(fixture(), 'T2', 'T2', m.EPA[0]), null);
  assert.equal(m.distribution([], 'adj_off_epa_per_play', 1), null);
});

test('default pair ignores finished games and picks next available matchup', () => {
  const teams = fixture().map((r) => r.team);
  const board = [
    { away_team: 'T0', home_team: 'T1', away_score: '7', home_score: '14' },
    { away_team: 'T4', home_team: 'T7', away_score: '', home_score: '' }
  ];
  assert.deepEqual(m.defaultPair(teams, board), ['T4', 'T7']);
});

test('deployment publishes the matchup source and navigation', () => {
  const root = path.join(__dirname, '..');
  const config = JSON.parse(fs.readFileSync(path.join(root, 'vercel.json')));
  assert.ok(config.buildCommand.includes('matchup.js'));
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  const site = fs.readFileSync(path.join(root, 'site.js'), 'utf8');
  assert.ok(html.includes('id="page-matchup"'));
  assert.ok(html.includes('data-route="matchup"'));
  assert.ok(html.includes('src="matchup.js"'));
  assert.ok(site.includes('window.NFLMatchup.init'));
  assert.ok(site.includes('PATHS.teamState'));
  // A tab in the markup is not navigable unless setRoute accepts its hash.
  assert.match(site, /const valid = \[[^\]]*'matchup'/);
  assert.ok(html.includes('href="#matchup" data-route="matchup"'));
});
