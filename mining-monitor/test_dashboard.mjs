import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

export async function runDashboardTests() {
  const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
  const app = await readFile(new URL('./app.js', import.meta.url), 'utf8');
  const passed = [];

  assert.match(html, /<main id="dashboard" aria-label="Empty dashboard canvas"><\/main>/);
  passed.push('dashboard canvas is empty');

  assert.doesNotMatch(html, /<script\b/i);
  assert.doesNotMatch(app, /fetch\s*\(/);
  passed.push('dashboard makes no data requests');

  for (const content of ['Champions', 'Tosky versions', 'Race archives', 'score', 'metric', 'table']) {
    assert.ok(!html.includes(content), `unexpected dashboard content: ${content}`);
  }
  passed.push('no dashboard data or placeholder content is rendered');

  return passed;
}
