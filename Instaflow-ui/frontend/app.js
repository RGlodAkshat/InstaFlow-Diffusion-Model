(() => {
  // Same-origin: FastAPI serves frontend + API
  const API_BASE = '';
  const ROUTES = { compare: `${API_BASE}/compare` };

  const qs = (sel, root = document) => root.querySelector(sel);

  function toast(msg, timeout = 2600) {
    const el = qs('#toast');
    el.textContent = msg;
    el.hidden = false;
    setTimeout(() => { el.hidden = true; }, timeout);
  }

  function setBtnLoading(isLoading) {
    const btn = qs('#btnGenerate');
    if (isLoading) { btn.classList.add('loading'); btn.setAttribute('disabled','true'); }
    else { btn.classList.remove('loading'); btn.removeAttribute('disabled'); }
  }

  function setCardImage(cardId, imageUrl) {
    if (!imageUrl) return;
    const img = document.querySelector(`#${cardId} img`);
    // cache-bust so we always see the newest image
    const abs = imageUrl.startsWith('http') ? imageUrl : `${imageUrl}`;
    img.src = `${abs}${abs.includes('?') ? '&' : '?'}t=${Date.now()}`;
  }

  function setCardMetrics(cardId, latencyMs, clipScore, openUrl) {
    const suffix = cardId.split('-')[1];
    qs(`#latency-${suffix}`).textContent = `${Math.round(latencyMs)} ms`;
    qs(`#clip-${suffix}`).textContent = Number(clipScore).toFixed(4);

    const openA = qs(`#open-${suffix}`);
    const dlA = qs(`#dl-${suffix}`);
    if (openUrl) {
      const abs = openUrl.startsWith('http') ? openUrl : `${openUrl}`;
      openA.href = abs; openA.hidden = false;
      dlA.href  = abs; dlA.hidden  = false;
    }
  }

  async function postJSON(url, payload) {
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); if (j?.error) msg = j.error; } catch {}
      throw new Error(msg);
    }
    return res.json();
  }

  function readForm() {
    const prompt = qs('#prompt').value.trim();
    const seedRaw = qs('#seed').value.trim();
    const seed = seedRaw === '' ? null : Number(seedRaw);
    return { prompt, seed };
  }

  function validate({ prompt }) {
    const errs = [];
    if (prompt.length < 3) errs.push('Prompt must be at least 3 characters.');
    if (prompt.length > 500) errs.push('Prompt must be ≤ 500 characters.');
    return errs;
  }

  function showFormError(errs) {
    const box = qs('#formError');
    if (!errs.length) { box.hidden = true; box.textContent = ''; return; }
    box.hidden = false; box.textContent = errs.join(' ');
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const { prompt, seed } = readForm();
    const errs = validate({ prompt });
    showFormError(errs);
    if (errs.length) return;

    // Reset
    qs('#emptyState').style.display = 'none';
    ['baseline','quantized'].forEach(sfx => {
      qs(`#img-${sfx}`).removeAttribute('src');
      qs(`#latency-${sfx}`).textContent = '—';
      qs(`#clip-${sfx}`).textContent = '—';
      qs(`#open-${sfx}`).hidden = true;
      qs(`#dl-${sfx}`).hidden = true;
    });

    setBtnLoading(true);
    try {
      const out = await postJSON(ROUTES.compare, { prompt, width: 512, height: 512, seed });

      // Baseline
      setCardImage('card-baseline', out?.baseline?.image_url);
      setCardMetrics('card-baseline', out?.baseline?.latency_ms ?? 0, out?.baseline?.clip_score ?? 0, out?.baseline?.image_url);

      // Quantized
      setCardImage('card-quantized', out?.quantized?.image_url);
      setCardMetrics('card-quantized', out?.quantized?.latency_ms ?? 0, out?.quantized?.clip_score ?? 0, out?.quantized?.image_url);

    } catch (err) {
      console.error(err);
      toast(`Generation failed: ${err.message || err}`);
    } finally {
      setBtnLoading(false);
    }
  }

  function bind() {
    qs('#promptForm').addEventListener('submit', handleSubmit);
    qs('#btnClear').addEventListener('click', () => {
      qs('#prompt').value = '';
      qs('#seed').value = '';
      qs('#emptyState').style.display = 'block';
      ['baseline','quantized'].forEach(sfx => {
        qs(`#img-${sfx}`).removeAttribute('src');
        qs(`#latency-${sfx}`).textContent = '—';
        qs(`#clip-${sfx}`).textContent = '—';
        qs(`#open-${sfx}`).hidden = true;
        qs(`#dl-${sfx}`).hidden = true;
      });
    });

    const dlg = qs('#helpDialog');
    qs('#btnHelp').addEventListener('click', () => dlg.showModal());
    dlg.addEventListener('cancel', e => e.preventDefault());

    // Ctrl/Cmd+Enter to submit
    qs('#prompt').addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') qs('#promptForm').requestSubmit();
    });
  }

  document.addEventListener('DOMContentLoaded', bind);
})();
