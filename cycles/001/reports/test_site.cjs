// Exercise the real inline script with a minimal DOM adapter, not a browser.
// Node vm: https://nodejs.org/api/vm.html ; assert: https://nodejs.org/api/assert.html
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const project = path.resolve(__dirname, '../../..');
const html = fs.readFileSync(path.join(project, 'site/index.html'), 'utf8');
const data = JSON.parse(fs.readFileSync(path.join(project, 'site/data.json')));
class Element {
  constructor() { this.children = []; this.value = ''; this.textContent = ''; this.events = {}; this.clientWidth = 900; }
  append(node) { this.children.push(node); }
  replaceChildren() { this.children = []; }
  addEventListener(name, fn) { this.events[name] = fn; }
  getContext() { return new Proxy({}, {get: (_, key) => () => {}}); }
}
async function check(ok) {
  const elements = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const document = {getElementById: id => { assert.ok(elements[id], id); return elements[id]; }, createElement: () => new Element()};
  vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], {
    document, window: {devicePixelRatio: 1, addEventListener() {}},
    fetch: async () => ({ok, json: async () => data})
  }, {timeout: 5000});
  await new Promise(resolve => setImmediate(resolve));
  if (!ok) { assert.equal(elements.primary.className, 'error'); return {fetch_error_displayed: true}; }
  assert.equal(elements.title.textContent, data.title);
  const counts = {};
  function verify(expected, name) {
    assert.equal(elements.rows.children.length, expected.length);
    assert.deepEqual(elements.rows.children.map(tr => tr.children[0].textContent), expected.map(r => r.id));
    counts[name] = expected.length;
    const first = expected.find(r => r.y_true?.length);
    assert.equal(elements.examples.children.length, first ? Math.min(8, first.y_true.length) : 0);
    if (first) assert.ok(elements.examples.children[0].textContent.includes(first.id), 'Examples must identify the displayed run');
  }
  verify(data.records, 'all');
  for (const key of ['id', 'method', 'seed', 'split', 'outcome']) {
    const value = String(data.records[0][key] ?? 'not recorded');
    elements[key].value = value;
    elements[key].events.change();
    verify(data.records.filter(r => String(r[key] ?? 'not recorded') === value), key);
    elements[key].value = '';
  }
  elements.id.value = data.records[0].id;
  elements.split.value = data.records[0].split === 'test' ? 'validation' : 'test';
  elements.split.events.change();
  verify([], 'conflicting_filters');
  elements.id.value = ''; elements.split.value = ''; elements.split.events.change();
  verify(data.records, 'reset');
  return {counts, examples_identify_run: true};
}
(async () => {
  const result = {analysis_kind: 'exploratory', environment: 'Node vm with minimal DOM/canvas adapter; no browser rendering', passed: await check(true), error_path: await check(false)};
  const output = path.resolve(__dirname, '../research/review-site-test.json');
  fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
