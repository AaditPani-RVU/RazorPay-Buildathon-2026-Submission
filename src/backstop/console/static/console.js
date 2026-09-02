/* The console's client. It renders what the server says and calls back; it
   holds no rules and computes no verdicts, for the same reason app.py has
   none -- a safety boundary somebody could reach around by editing a page
   would not be one. Every number here was read off a ledger. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  boot: null,
  rows: [],
  rowOffset: 0,
  rowTotal: 0,
  filters: { source: '', surface: '', disposition: '', rule: '', q: '' },
  tab: 'rulings',
  polling: null,
};

const ROW_PAGE = 200;

// ---------------------------------------------------------------- fetching

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* body was not json */ }
    throw new Error(detail);
  }
  return res.json();
}

const post = (path, body) =>
  api(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body || {}),
  });

function banner(text, isError) {
  const el = $('#banner');
  el.textContent = text;
  el.classList.toggle('err', !!isError);
  el.hidden = !text;
}

// ---------------------------------------------------------------- helpers

const fmtTime = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toISOString().slice(0, 16).replace('T', ' ') + 'Z';
};

const fmtDay = (iso) => (iso ? new Date(iso).toISOString().slice(5, 16).replace('T', ' ') : '—');

const chip = (value, extra) =>
  `<span class="chip ${value} ${extra || ''}">${value.replace(/_/g, ' ')}</span>`;

const esc = (s) =>
  String(s == null ? '' : s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------------------------------------------------------------- build

async function build() {
  const seed = Number($('#seed').value) || 1;
  const sample = Number($('#sample').value) || 150;
  $('#build').disabled = true;
  banner('generating a batch, running three scans, and ruling on two plans…');
  try {
    const job = await post('/api/build', { seed, sample });
    await waitFor(job.id);
    $('#empty').hidden = true;
    $('#surfaces').hidden = false;
    $('#tabs').hidden = false;
    $('#clockbox').hidden = false;
    await refreshAll();
    banner('');
  } catch (err) {
    banner(err.message, true);
  } finally {
    $('#build').disabled = false;
  }
}

async function waitFor(jobId) {
  for (;;) {
    const job = await api(`/api/jobs/${jobId}`);
    if (job.state === 'done') return job;
    if (job.state === 'failed') throw new Error(job.error);
    banner(`${job.label}… ${job.seconds}s`);
    await new Promise((r) => setTimeout(r, 400));
  }
}

// ---------------------------------------------------------------- state

async function refreshAll() {
  const s = await api('/api/state');
  state.boot = state.boot || (await api('/api/bootstrap'));
  renderClock(s);
  renderSurfaces(s);
  renderSplits(s);
  fillFilters();
  $('#badge-approvals').textContent = s.pending || '';
  $('#badge-schedule').textContent = s.scheduler.waiting || '';
  await renderTab();
}

function renderClock(s) {
  $('#clock').textContent = fmtTime(s.clock);
}

function renderSurfaces(s) {
  const order = ['payment', 'recurring', 'receivable'];
  $('#surfaces').innerHTML = order
    .map((key) => {
      const card = s.surfaces[key];
      const led = s.ledger_by_surface[key];
      return `<div class="surfacecard">
        <div class="name">${esc(card.label)}</div>
        <div class="sub2">${esc(card.subtitle)}</div>
        <div class="value">${esc(card.at_risk)}</div>
        <div class="unit">at risk · ${card.subjects.toLocaleString()} subjects</div>
        <div class="detail">${esc(card.detail)}</div>
        <div class="detail dim mono">recovered so far ${esc(led.recovered)} ·
          ${led.contacts} contacts · ${led.vetoed} refused</div>
      </div>`;
    })
    .join('');
}

function renderSplits(s) {
  const dispositions = ['allow', 'reschedule', 'require_approval', 'deny'];
  const plans = [
    ['backstop', 'the plan being operated'],
    ['naive', 'ruled beside it, dispatched nowhere'],
  ];
  $('#splits').innerHTML = plans
    .map(([plan, note]) => {
      const counts = dispositions.map((d) => s.counts[`${plan}:${d}`] || 0);
      const total = counts.reduce((a, b) => a + b, 0) || 1;
      const bars = dispositions
        .map((d, i) => `<span class="${d}" style="width:${(counts[i] / total) * 100}%"></span>`)
        .join('');
      const legend = dispositions
        .map((d, i) => `<span class="${d}"><span class="k">${d.replace(/_/g, ' ')}</span>
          <b>${counts[i].toLocaleString()}</b></span>`)
        .join('');
      return `<div class="split">
        <h3>${plan}</h3>
        <div class="total">${total.toLocaleString()} actions <span class="dim">— ${note}</span></div>
        <div class="bar-stack">${bars}</div>
        <div class="legend">${legend}</div>
      </div>`;
    })
    .join('');
}

function fillFilters() {
  if ($('#f-surface').options.length > 1) return;
  for (const surface of state.boot.surfaces) {
    $('#f-surface').add(new Option(surface, surface));
  }
  for (const d of state.boot.dispositions) {
    $('#f-disposition').add(new Option(d.replace(/_/g, ' '), d));
  }
  for (const rule of state.boot.rules) {
    $('#f-rule').add(new Option(rule.id, rule.id));
  }
}

// ---------------------------------------------------------------- tabs

async function renderTab() {
  $$('.panel').forEach((p) => { p.hidden = true; });
  const panel = $(`#panel-${state.tab}`);
  if (panel) panel.hidden = false;
  const render = {
    rulings: loadRows,
    measure: loadMeasurement,
    rules: loadRules,
    approvals: loadApprovals,
    schedule: loadSchedule,
    bench: loadBench,
    detect: loadDetect,
    diagnose: loadDiagnoses,
    live: loadLive,
    log: loadLog,
  }[state.tab];
  if (render) await render();
}

// ---------------------------------------------------------------- rulings

async function loadRows(append) {
  if (!append) state.rowOffset = 0;
  const params = new URLSearchParams({
    ...state.filters,
    limit: ROW_PAGE,
    offset: state.rowOffset,
  });
  const data = await api(`/api/rows?${params}`);
  state.rowTotal = data.total;
  const body = $('#rowtable tbody');
  const html = data.rows.map(rowHtml).join('');
  if (append) body.insertAdjacentHTML('beforeend', html);
  else body.innerHTML = html;
  state.rowOffset += data.rows.length;
  $('#rowcount').textContent =
    `${state.rowOffset.toLocaleString()} of ${data.total.toLocaleString()} shown`;
  $('#more-rows').hidden = state.rowOffset >= data.total;
}

function rowHtml(r) {
  const verdicts = r.verdicts.length
    ? r.verdicts
        .map((v) => `<span class="rulechip ${v.disposition}" title="${esc(v.reason)}"
             data-rule="${v.rule}">${v.rule}</span>`)
        .join('')
    : '<span class="dim">no rule objected</span>';
  const when = r.moved
    ? `<span class="dim">${fmtDay(r.scheduled_at)}</span> → ${fmtDay(r.final_at)}`
    : fmtDay(r.scheduled_at);
  return `<tr>
    <td class="nowrap">${chip(r.disposition)}</td>
    <td class="mono dim">${r.source}</td>
    <td class="mono dim">${r.surface}</td>
    <td class="mono">${r.type}${r.channel ? ` <span class="dim">via ${r.channel}</span>` : ''}</td>
    <td class="mono dim">${esc(r.subject)}</td>
    <td class="mono nowrap">${when}</td>
    <td>${verdicts}</td>
  </tr>`;
}

// ---------------------------------------------------------------- measure

const SURFACE_TITLES = {
  payment: ['Payments', 'one-off checkout failures', 'orders'],
  recurring: ['Recurring', 'mandates that stopped collecting', 'mandates'],
  receivable: ['Receivables', 'invoices that were never paid', 'invoices'],
};

async function loadMeasurement() {
  const data = await api('/api/measurement');
  $('#measurebackend').textContent = data.backend
    ? `reasoning backend: ${data.backend}`
    : 'not run yet — about fifteen seconds offline';
  if (!data.backend) { $('#measureout').innerHTML = ''; return; }
  $('#measureout').innerHTML = Object.entries(SURFACE_TITLES)
    .map(([key, [title, subtitle, subject]]) => {
      const rows = data.surfaces[key] || [];
      return `<h3 class="surfacehead">${title}
          <span class="dim">${subtitle} · ${esc(data.at_risk[key] || '')} at risk</span></h3>
        <div class="tablewrap"><table class="rows"><thead><tr>
          <th>arm</th><th>recovered</th><th>of which illegal</th><th>keepable net</th>
          <th>${subject}</th><th>charges</th><th>contacts</th><th>burst</th><th>violations</th>
        </tr></thead><tbody>${rows
          .map(
            (r) => `<tr class="${r.arm === 'backstop' ? 'winner' : ''}">
            <td class="mono">${esc(r.arm)}</td>
            <td class="mono">${esc(r.recovered)}</td>
            <td class="mono ${r.illegal ? 'bad' : 'dim'}">${esc(r.illegal) || '–'}</td>
            <td class="mono">${esc(r.keepable_net)}</td>
            <td class="mono dim">${r.subjects.toLocaleString()}</td>
            <td class="mono dim">${r.charges.toLocaleString()}</td>
            <td class="mono dim">${r.contacts.toLocaleString()}</td>
            <td class="mono dim">${r.burst}</td>
            <td class="mono ${r.violations ? 'bad' : 'good'}">${r.violations.toLocaleString()}</td>
          </tr>`
          )
          .join('')}</tbody></table></div>`;
    })
    .join('');
}

async function runMeasurement() {
  $('#run-measure').disabled = true;
  banner('replaying the batch through four arms…');
  try {
    const job = await post('/api/measure', { offline: !$('#measure-model').checked });
    await waitFor(job.id);
    banner('');
    await loadMeasurement();
  } catch (err) {
    banner(err.message, true);
  } finally {
    $('#run-measure').disabled = false;
  }
}

// ---------------------------------------------------------------- rules

async function loadRules() {
  const data = await api('/api/rules');
  const why = Object.fromEntries(state.boot.rules.map((r) => [r.id, r.why]));
  const max = Math.max(1, ...data.rules.map((r) => Math.max(r.backstop, r.naive)));
  $('#rulelist').innerHTML = data.rules
    .map((r) => {
      const spoke = r.backstop + r.naive;
      const meter = (plan, n) => `<div class="meterline ${plan}">
        <span class="dim">${plan}</span>
        <span class="track"><span class="fill" style="width:${(n / max) * 100}%"></span></span>
        <span class="n">${n.toLocaleString()}</span>
      </div>`;
      const body = spoke
        ? meter('backstop', r.backstop) + meter('naive', r.naive)
        : `<div class="silent">never spoke on this batch — a guarantee it did not need</div>`;
      return `<div class="rulerow">
        <div>
          <div class="id">${r.rule}</div>
          <div class="why">${esc(why[r.rule] || r.example || '')}</div>
        </div>
        <div class="meter">${body}</div>
      </div>`;
    })
    .join('');
}

// ---------------------------------------------------------------- approvals

async function loadApprovals() {
  const chosen = $('#f-approval-state').value;
  const data = await api(`/api/approvals?state=${encodeURIComponent(chosen)}`);
  $('#approvalcount').textContent = Object.entries(data.counts)
    .map(([k, v]) => `${k} ${v}`)
    .join('  ·  ');
  $('#approvallist').innerHTML = data.requests
    .map(
      (r) => `<div class="card" data-id="${r.id}">
      <div class="head">
        ${chip(r.state)}
        <span class="title">${r.asking_rule}</span>
        <span class="dim mono">${r.surface}</span>
      </div>
      <div class="what">${esc(r.action)}</div>
      <div class="why">${esc(r.reason)}</div>
      <div class="foot">
        <span class="dim mono">expires ${fmtDay(r.expires_at)}
          (${r.hours_left > 0 ? `${r.hours_left}h left` : 'lapsed'})</span>
        ${
          r.state === 'pending'
            ? `<button data-approve="${r.id}">Approve</button>
               <button class="ghost" data-reject="${r.id}">Reject</button>`
            : r.decided_by
              ? `<span class="dim mono">by ${esc(r.decided_by)}</span>`
              : ''
        }
      </div>
    </div>`
    )
    .join('') ||
    `<div class="card"><div class="why">Nothing ${esc(chosen)} at this clock.</div></div>`;
  $('#approvallist').dataset.ids = data.requests
    .filter((r) => r.state === 'pending')
    .map((r) => r.id)
    .join(',');
}

async function decide(ids, approve) {
  if (!ids.length) return;
  const by = $('#reviewer').value.trim();
  if (!by) { banner('an approval must name the person who gave it', true); return; }
  const res = await post('/api/approvals/decide', { ids, approve, by });
  if (res.refused.length) banner(res.refused[0], true);
  else banner('');
  await refreshAll();
}

async function release() {
  const res = await post('/api/approvals/release', {});
  const out = $('#releaseout');
  const refusals = res.released.filter((r) => r.outcome === 'refused');
  out.hidden = false;
  out.innerHTML =
    `<span><b>${res.counts.released}</b> released</span>
     <span><b>${res.counts.rescheduled}</b> rescheduled</span>
     <span class="refusal"><b>${res.counts.refused}</b> refused by a rule after approval</span>` +
    refusals
      .slice(0, 4)
      .map(
        (r) => `<div class="refusal">${esc(r.action)} — <b>${esc(r.rule)}</b>:
           ${esc(r.reason)}</div>`
      )
      .join('');
  await refreshAll();
}

// ---------------------------------------------------------------- schedule

async function loadSchedule() {
  const chosen = $('#f-schedule-state').value;
  const data = await api(`/api/schedule?state=${encodeURIComponent(chosen)}`);
  $('#schedulecount').textContent = Object.entries(data.counts)
    .map(([k, v]) => `${k} ${v}`)
    .join('  ·  ');
  $('#scheduletable tbody').innerHTML = data.entries
    .map(
      (e) => `<tr>
      <td class="nowrap">${chip(e.state)}</td>
      <td class="mono nowrap">${fmtDay(e.due_at)}</td>
      <td class="mono dim">${e.surface}</td>
      <td class="mono">${e.type}</td>
      <td class="mono dim">${esc(e.subject)}</td>
      <td class="mono dim">${e.deferrals || ''}</td>
      <td class="dim">${esc(e.note)}</td>
    </tr>`
    )
    .join('');
}

async function tick(body) {
  const res = await post('/api/clock', body);
  const out = $('#tickout');
  out.hidden = false;
  const parts = [
    ['moved', `${res.moved_hours.toFixed(1)}h`],
    ['fired', res.fired],
    ['refused at fire time', res.refused],
    ['deferred again', res.deferred],
    ['abandoned', res.abandoned],
    ['dropped as stale', res.stale],
    ['approvals expired', res.expired],
  ];
  out.innerHTML = parts
    .map(([k, v]) => `<span><span class="dim">${k}</span> <b>${v}</b></span>`)
    .join('');
  await refreshAll();
}

// ---------------------------------------------------------------- bench

async function loadBench() {
  const data = await api('/api/probes');
  const all = data.matched === data.total;
  $('#benchsummary').innerHTML = `
    <div class="stat"><div class="k">probes</div><div class="v">${data.total}</div></div>
    <div class="stat"><div class="k">ruled as expected</div>
      <div class="v ${all ? 'good' : 'bad'}">${data.matched}/${data.total}</div></div>
    <div class="stat"><div class="k">rules consulted</div>
      <div class="v">${state.boot.rules.length} on every action</div></div>`;
  $('#benchlist').innerHTML = data.probes
    .map(
      (p) => `<div class="card ${p.matches ? 'hit' : 'miss'}">
      <div class="head">
        ${chip(p.got)}
        <span class="title">${esc(p.label)}</span>
      </div>
      <div class="what">${esc(p.proposed)}</div>
      <div class="why">expected <b>${esc(p.expected)}</b>${
        p.moved_to ? ` · moved to ${esc(p.moved_to)}` : ''
      }</div>
      <div>${
        p.verdicts.length
          ? p.verdicts
              .map(
                (v) => `<span class="rulechip ${v.disposition}"
                   title="${esc(v.reason)}">${v.rule}</span>`
              )
              .join('')
          : '<span class="dim">no rule objected</span>'
      }</div>
      <div class="why">${esc(p.verdicts.map((v) => v.reason).join('; '))}</div>
    </div>`
    )
    .join('');
}

// ---------------------------------------------------------------- detect

async function loadDetect() {
  const s = await api('/api/state');
  const d = s.detection;
  $('#detectscore').innerHTML = `
    <div class="stat"><div class="k">incidents in the batch</div><div class="v">${d.incidents}</div></div>
    <div class="stat"><div class="k">found</div><div class="v good">${d.found}</div></div>
    <div class="stat"><div class="k">missed</div>
      <div class="v ${d.missed ? 'bad' : ''}">${d.missed}</div></div>
    <div class="stat"><div class="k">false positives</div>
      <div class="v ${d.false_positives ? 'bad' : ''}">${d.false_positives}</div></div>
    <div class="stat"><div class="k">recall</div><div class="v">${(d.recall * 100).toFixed(0)}%</div></div>
    <div class="stat"><div class="k">precision</div><div class="v">${(d.precision * 100).toFixed(0)}%</div></div>
    <div class="stat"><div class="k">mean latency</div><div class="v">${d.latency_minutes}m</div></div>
    <div class="stat"><div class="k">money found / missed</div>
      <div class="v">${esc(d.money_found)} / ${esc(d.money_missed)}</div></div>`;
  $('#clustertable tbody').innerHTML = d.clusters
    .map(
      (c) => `<tr>
      <td class="mono">${c.id}</td>
      <td class="mono">${esc(c.segment)}</td>
      <td class="mono dim nowrap">${fmtDay(c.starts_at)} · ${c.minutes}m</td>
      <td class="mono">${esc(c.at_risk)}</td>
      <td class="mono dim">${esc(c.decline)}</td>
      <td class="mono dim">${(c.share * 100).toFixed(0)}%</td>
      <td class="mono">${esc(c.truth) || '<span class="dim">no matching incident</span>'}</td>
    </tr>`
    )
    .join('');
}

// ---------------------------------------------------------------- diagnose

async function loadDiagnoses() {
  const data = await api('/api/diagnoses');
  $('#diagnosebackend').textContent = data.backend
    ? `reasoning backend: ${data.backend}`
    : state.boot.settings.groq
      ? 'not run yet'
      : 'GROQ_API_KEY is not set — diagnosis needs a model';
  $('#run-diagnose').disabled = !state.boot.settings.groq;
  $('#diagnoselist').innerHTML = data.diagnoses
    .map(
      (d) => `<div class="card ${d.ok ? (d.correct ? 'hit' : 'miss') : 'miss'}">
      <div class="head">
        ${chip(d.ok ? (d.correct ? 'approved' : 'rejected') : 'rejected',
               '')}
        <span class="title">${esc(d.cluster)}</span>
        <span class="dim mono">${esc(d.segment)}</span>
      </div>
      ${
        d.ok
          ? `<div class="what">predicted <b>${esc(d.root_cause)}</b>
               <span class="dim">confidence ${d.confidence.toFixed(2)}</span></div>
             <div class="why">ground truth <b>${esc(d.truth) || 'no matching incident'}</b></div>
             <ul class="evidence">${d.evidence.map((e) => `<li>${esc(e)}</li>`).join('')}</ul>`
          : `<div class="what">no usable diagnosis</div>
             <div class="why">${esc(d.error)}</div>`
      }
      <div class="why dim">${esc(d.at_risk)} at risk${
        d.repaired ? ' · schema repair was needed' : ''
      }</div>
    </div>`
    )
    .join('') || '<div class="card"><div class="why">Nothing diagnosed yet.</div></div>';
}

async function runDiagnose() {
  $('#run-diagnose').disabled = true;
  banner('asking the model about each cluster…');
  try {
    const job = await post('/api/diagnose', {});
    await waitFor(job.id);
    banner('');
    await loadDiagnoses();
  } catch (err) {
    banner(err.message, true);
  } finally {
    $('#run-diagnose').disabled = false;
  }
}

// ---------------------------------------------------------------- live

async function loadLive() {
  const cfg = state.boot.settings;
  $('#livestatus').textContent = !cfg.razorpay
    ? 'RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set — everything else on this page runs without them'
    : !cfg.razorpay_test_mode
      ? 'the configured key is not a test key; the adapter refuses to construct against one'
      : cfg.webhook_secret
        ? 'test-mode keys and a webhook secret are configured'
        : 'test-mode keys configured; no webhook secret, so the receiver would refuse every delivery';
  $('#go-live').disabled = !cfg.razorpay || !cfg.razorpay_test_mode;

  $('#webhookinfo').innerHTML =
    `<span><span class="dim">endpoint</span> <b>POST ${location.origin}/webhooks/razorpay</b></span>
     <span><span class="dim">header</span> <b>X-Razorpay-Signature</b></span>
     <span class="dim">Razorpay needs a publicly reachable URL, so a laptop wants a tunnel in
       front of this. The verification, matching and crediting are the real ones either way —
       only the postman is local.</span>`;

  const data = await api('/api/dispatches');
  renderDispatchResults(data.results);
  if (!data.live) { $('#livecandidates').innerHTML = ''; return; }
  await renderCandidates();
}

async function renderCandidates() {
  // One dispatchable action per surface, taken off the scheduler. The point is
  // to show a permitted action leaving the building, not to send in volume.
  const data = await api('/api/schedule?state=waiting&limit=400');
  const perSurface = {};
  for (const e of data.entries) {
    if (!perSurface[e.surface]) perSurface[e.surface] = e;
  }
  $('#livecandidates').innerHTML = Object.values(perSurface)
    .map(
      (e) => `<div class="card">
      <div class="head">${chip('waiting')}<span class="title">${e.surface}</span></div>
      <div class="what">${esc(e.action)}</div>
      <div class="why">due ${fmtDay(e.due_at)} · ruled on again at the moment you press this</div>
      <div class="foot"><button data-dispatch="${e.id}">Dispatch for real</button></div>
    </div>`
    )
    .join('') || '<div class="card"><div class="why">Nothing is waiting to dispatch.</div></div>';
}

function renderDispatchResults(results) {
  const out = $('#liveresults');
  out.hidden = !results.length;
  out.innerHTML = results
    .map(
      (r) => `<div>${chip(r.outcome === 'dispatched' ? 'released' : 'refused')}
        ${esc(r.action)} — ${esc(r.detail)}
        ${r.external ? `<a href="${esc(r.external.url)}" target="_blank" rel="noopener">${esc(r.external.id)}</a>` : ''}</div>`
    )
    .join('');
}

async function goLive() {
  $('#go-live').disabled = true;
  try {
    const info = await post('/api/live', {});
    const caps = $('#livecaps');
    caps.hidden = false;
    caps.innerHTML =
      `<div class="stat"><div class="k">key</div><div class="v">${esc(info.key)}</div></div>
       <div class="stat"><div class="k">webhook receiver</div>
         <div class="v ${info.webhook ? 'good' : 'bad'}">${info.webhook ? 'armed' : 'no secret'}</div></div>
       <div class="stat"><div class="k">known dispatches</div><div class="v">${info.known_dispatches}</div></div>
       <div class="stat"><div class="k">already credited</div><div class="v">${info.already_credited}</div></div>` +
      info.capabilities
        .map(
          (c) => `<div class="stat"><div class="k">${esc(c.name)}</div>
            <div class="v ${c.ok ? 'good' : 'bad'}">${c.ok ? 'ok' : 'no'}</div></div>`
        )
        .join('');
    await renderCandidates();
    banner('');
  } catch (err) {
    banner(err.message, true);
  } finally {
    $('#go-live').disabled = false;
  }
}

// ---------------------------------------------------------------- log

async function loadLog() {
  const data = await api('/api/events?limit=400');
  $('#logtable tbody').innerHTML = data.events
    .map(
      (e) => `<tr>
      <td class="mono dim nowrap">${fmtDay(e.at)}</td>
      <td class="nowrap">${chip(e.kind, 'plain')}</td>
      <td class="mono">${
        e.rule ? `<span class="rulechip deny">${esc(e.rule)}</span>` : ''
      }</td>
      <td>${esc(e.text)}</td>
    </tr>`
    )
    .join('');
}

// ---------------------------------------------------------------- wiring

$('#build').addEventListener('click', build);

$('#tabs').addEventListener('click', (ev) => {
  const button = ev.target.closest('button[data-tab]');
  if (!button) return;
  $$('#tabs button').forEach((b) => b.classList.toggle('on', b === button));
  state.tab = button.dataset.tab;
  renderTab().catch((err) => banner(err.message, true));
});

for (const [id, key] of [
  ['#f-source', 'source'],
  ['#f-surface', 'surface'],
  ['#f-disposition', 'disposition'],
  ['#f-rule', 'rule'],
]) {
  $(id).addEventListener('change', () => {
    state.filters[key] = $(id).value;
    loadRows().catch((err) => banner(err.message, true));
  });
}

let searchTimer;
$('#f-q').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.filters.q = $('#f-q').value;
    loadRows().catch((err) => banner(err.message, true));
  }, 250);
});

$('#more-rows').addEventListener('click', () => loadRows(true));

// A rule chip anywhere in the stream filters the stream by that rule. The
// question a reader has on seeing one is always "what else did this rule do".
$('#rowtable').addEventListener('click', (ev) => {
  const chipEl = ev.target.closest('.rulechip[data-rule]');
  if (!chipEl) return;
  state.filters.rule = chipEl.dataset.rule;
  $('#f-rule').value = chipEl.dataset.rule;
  loadRows().catch((err) => banner(err.message, true));
});

$('#approvallist').addEventListener('click', (ev) => {
  const yes = ev.target.closest('[data-approve]');
  const no = ev.target.closest('[data-reject]');
  if (yes) decide([yes.dataset.approve], true).catch((e) => banner(e.message, true));
  if (no) decide([no.dataset.reject], false).catch((e) => banner(e.message, true));
});

const visibleIds = () => ($('#approvallist').dataset.ids || '').split(',').filter(Boolean);
$('#approve-visible').addEventListener('click', () =>
  decide(visibleIds(), true).catch((e) => banner(e.message, true)));
$('#reject-visible').addEventListener('click', () =>
  decide(visibleIds(), false).catch((e) => banner(e.message, true)));
$('#release').addEventListener('click', () => release().catch((e) => banner(e.message, true)));
$('#f-approval-state').addEventListener('change', () =>
  loadApprovals().catch((e) => banner(e.message, true)));

$('#tick-next').addEventListener('click', () => tick({ to_next: true }).catch((e) => banner(e.message, true)));
$('#tick-1').addEventListener('click', () => tick({ hours: 1 }).catch((e) => banner(e.message, true)));
$('#tick-6').addEventListener('click', () => tick({ hours: 6 }).catch((e) => banner(e.message, true)));
$('#tick-24').addEventListener('click', () => tick({ hours: 24 }).catch((e) => banner(e.message, true)));
$('#f-schedule-state').addEventListener('change', () =>
  loadSchedule().catch((e) => banner(e.message, true)));

$('#run-diagnose').addEventListener('click', runDiagnose);
$('#run-measure').addEventListener('click', runMeasurement);

$('#go-live').addEventListener('click', goLive);
$('#livecandidates').addEventListener('click', async (ev) => {
  const button = ev.target.closest('[data-dispatch]');
  if (!button) return;
  button.disabled = true;
  try {
    const res = await post('/api/dispatch', {
      id: button.dataset.dispatch,
      notify: $('#notify').checked,
    });
    if (!res.dispatched) {
      banner(`${res.disposition}: ${res.rule || ''} ${res.reason || ''}`.trim(), true);
    } else {
      banner('');
    }
    await loadLive();
  } catch (err) {
    banner(err.message, true);
  } finally {
    button.disabled = false;
  }
});

// A console started with a batch already in it opens on that batch rather than
// on an invitation to make one -- `--cold` is the flag for the other case.
(async () => {
  state.boot = await api('/api/bootstrap');
  $('#seed').value = state.boot.defaults.seed;
  $('#sample').value = state.boot.defaults.sample;
  $('#reviewer').value = state.boot.defaults.reviewer;
  if (!state.boot.ready) return;
  $('#empty').hidden = true;
  $('#surfaces').hidden = false;
  $('#tabs').hidden = false;
  $('#clockbox').hidden = false;
  await refreshAll();
})().catch((err) => banner(err.message, true));
