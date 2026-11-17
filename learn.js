// =========================================================================
// Tracker Eye: learning, recording and reports
//
//  1. Example library   – photos of "ball", "rc-car", "not-target", from a folder,
//                         from the live feed, or from the Inbox
//  2. Verifier          – a second check on top of colour + shape: does this blob
//                         LOOK like the examples? (AI fingerprint, or a basic one)
//  3. Inbox             – labels the app suggests on its own; you tap to accept
//  4. Recorder          – logs every frame, catches failures, grades its own predictions
//  5. Report            – one zip to send for review: scoreboard, filmstrips, log
//  6. Replay + history  – run a saved video again and compare scores between versions
//  7. Saved setup       – rulers, walls, colours and sliders survive a reload
// =========================================================================
const Learn = (() => {
  const CODE_VERSION = '3.0';
  const q = id => document.getElementById(id);
  const LABELS = {ball: 'Ball', 'rc-car': 'RC Car', 'not-target': 'Not target'};
  const RUNS = ['Free run', '1 Ball sitting still', '2 Ball rolling', '3 Ball thrown and bounced', '4 Ball into a wall',
                '5 Person walking through', '6 Ball near skin and wood', '7 RC car driving circles', '8 Lights dimmed'];

  const S = {
    lib: [], libVersion: 0, ai: {status: 'loading', kind: 'basic', model: null},
    inbox: [], rec: null, lastReport: null, replay: false, frozen: false,
    ring: [], pool: [], fails: [], lastBlobs: [], snap: null,
    vAt: -1e9, vFails: 0, lastV: null, cache: new Map(), lastFresh: -1e9,
    suggestAt: {}, frameN: 0, prevLocked: false
  };

  // ---------------------------------------------------------------- helpers
  const cosine = (a, b) => { let s = 0; for(let i = 0; i < a.length; i++) s += a[i]*b[i]; return s; };
  const norm = v => { let s = 0; for(const x of v) s += x*x; s = Math.sqrt(s) || 1; for(let i = 0; i < v.length; i++) v[i] /= s; return v; };
  const median = a => { if(!a.length) return null; const b = [...a].sort((m, n) => m - n); return b[b.length >> 1]; };
  const pct = (a, p) => { if(!a.length) return null; const b = [...a].sort((m, n) => m - n); return b[Math.min(b.length-1, Math.floor(p*b.length))]; };
  const uid = () => Math.random().toString(36).slice(2, 10);
  const loadScript = src => new Promise((ok, bad) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => bad(new Error('could not load ' + src)); document.head.appendChild(s); });
  const toDataURL = c => c.toDataURL('image/jpeg', 0.85);
  const imgFrom = url => new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = url; });
  const download = (blob, name) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 4000); };
  const active = () => q('preset') ? q('preset').value : 'ball';
  const liveStream = () => (typeof stream !== 'undefined' && stream) || null;   // `stream` lives in index.html

  // Square crop around (cx, cy), resized to `out` px
  function crop(src, cx, cy, size, out = 128){
    const c = document.createElement('canvas'); c.width = c.height = out;
    const g = c.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, out, out);
    size = Math.max(16, size);
    g.drawImage(src, cx - size/2, cy - size/2, size, size, 0, 0, out, out);
    return c;
  }

  // ------------------------------------------------------------ fingerprints
  // AI: a small pretrained image network (MobileNet) turns a picture into ~1000 numbers
  // that describe what it looks like. Similar-looking things have similar numbers.
  async function loadAI(){
    try{
      if(!window.tf) await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
      if(!window.mobilenet) await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js');
      let m = null, name = '';
      try { m = await mobilenet.load({version: 2, alpha: 1.0}); name = 'MobileNet v2'; }
      catch(e){ m = await mobilenet.load({version: 1, alpha: 1.0, modelUrl: 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'}); name = 'MobileNet v1'; }
      const warm = document.createElement('canvas'); warm.width = warm.height = 224; aiPrint(m, warm);   // first run compiles shaders
      S.ai = {status: 'ready', kind: 'ai', model: m, name};
      for(const e of S.lib) e.fp = fingerprint(e.canvas);              // re-fingerprint with the AI
      uiLibrary();
    } catch(err){
      S.ai = {status: 'basic', kind: 'basic', model: null, name: 'basic', err: String(err.message || err)};
      uiLibrary();
    }
  }
  function aiPrint(m, canvas){
    return tf.tidy(() => norm(Float32Array.from(m.infer(canvas, true).dataSync())));
  }
  // Basic (no AI): colour histogram + edge-direction histogram. Fast, simple, decent.
  function basicPrint(canvas){
    const c = document.createElement('canvas'); c.width = c.height = 32;
    const g = c.getContext('2d', {willReadFrequently: true}); g.drawImage(canvas, 0, 0, 32, 32);
    const d = g.getImageData(0, 0, 32, 32).data, hue = new Float32Array(18), sv = new Float32Array(6), grey = new Float32Array(1024);
    for(let i = 0, p = 0; i < d.length; i += 4, p++){
      const [h, s, v] = rgbToHsv(d[i], d[i+1], d[i+2]);
      hue[Math.min(17, (h/20)|0)] += s*v; sv[Math.min(2, (s*3)|0)]++; sv[3 + Math.min(2, (v*3)|0)]++;
      grey[p] = 0.3*d[i] + 0.59*d[i+1] + 0.11*d[i+2];
    }
    const grad = new Float32Array(4*4*8);
    for(let y = 1; y < 31; y++) for(let x = 1; x < 31; x++){
      const gx = grey[y*32+x+1] - grey[y*32+x-1], gy = grey[(y+1)*32+x] - grey[(y-1)*32+x];
      const mag = Math.hypot(gx, gy); if(!mag) continue;
      const bin = Math.floor(((Math.atan2(gy, gx) + Math.PI) / (2*Math.PI)) * 8) % 8;
      grad[((y >> 3)*4 + (x >> 3))*8 + bin] += mag;
    }
    norm(hue); norm(sv); norm(grad);
    const out = new Float32Array(hue.length + sv.length + grad.length);
    out.set(hue.map(v => v*1.2)); out.set(sv.map(v => v*0.5), 18); out.set(grad, 24);
    return norm(out);
  }
  const fingerprint = canvas => S.ai.kind === 'ai' ? aiPrint(S.ai.model, canvas) : basicPrint(canvas);

  // --------------------------------------------------------- example library
  const DB = {
    db: null,
    async open(){ if(this.db) return this.db; this.db = await new Promise((ok, bad) => { const r = indexedDB.open('tracker-eye', 1); r.onupgradeneeded = () => r.result.createObjectStore('examples', {keyPath: 'id'}); r.onsuccess = () => ok(r.result); r.onerror = bad; }); return this.db; },
    async all(){ try{ const db = await this.open(); return await new Promise(ok => { const r = db.transaction('examples').objectStore('examples').getAll(); r.onsuccess = () => ok(r.result); r.onerror = () => ok([]); }); } catch(e){ return []; } },
    async put(x){ try{ const db = await this.open(); db.transaction('examples', 'readwrite').objectStore('examples').put(x); } catch(e){} },
    async clear(){ try{ const db = await this.open(); db.transaction('examples', 'readwrite').objectStore('examples').clear(); } catch(e){} },
    async del(id){ try{ const db = await this.open(); db.transaction('examples', 'readwrite').objectStore('examples').delete(id); } catch(e){} }
  };
  function normLabel(name){
    const n = String(name).toLowerCase();
    if(/not|neg|other|wrong|background/.test(n)) return 'not-target';
    if(/car/.test(n)) return 'rc-car';
    if(/ball/.test(n)) return 'ball';
    return n.replace(/[^a-z0-9]+/g, '-');
  }
  // Every 5th example of each label goes to the TEST set, which is never used to decide
  // locks, only to measure how good the verifier is.
  // Each action (a folder, an Inbox batch, one live label) is a "batch", so Undo removes the whole action
  let batchSeq = 0;
  const newBatch = () => `${Date.now()}-${++batchSeq}`;
  function addExample(canvas, label, source, save = true, batch = newBatch()){
    const n = S.lib.filter(e => e.label === label).length;
    const e = {id: uid(), label, source, split: n % 5 === 4 ? 'test' : 'train', canvas, fp: fingerprint(canvas), added: Date.now(), batch};
    S.lib.push(e); S.libVersion++; S.cache.clear();
    if(save) DB.put({id: e.id, label, source, split: e.split, img: toDataURL(canvas), added: e.added, batch});
    return e;
  }
  function removeExamples(ids){
    const set = new Set(ids); S.lib = S.lib.filter(e => !set.has(e.id)); for(const id of ids) DB.del(id);
    S.libVersion++; S.cache.clear(); uiLibrary();
  }
  function undoLast(){
    if(!S.lib.length) return flash('The library is empty.');
    const lastE = S.lib.reduce((m, e) => !m || e.added > m.added || (e.added === m.added && e.batch > m.batch) ? e : m, null);
    const ids = S.lib.filter(e => e.batch === lastE.batch).map(e => e.id);
    removeExamples(ids); if(!q('libModal').hidden) renderLibrary();
    flash(`Undone: removed the last ${ids.length} example${ids.length === 1 ? '' : 's'} added (${lastE.source}).`);
  }
  function relabel(e, label){
    e.label = label; S.libVersion++; S.cache.clear();
    DB.put({id: e.id, label, source: e.source, split: e.split, img: toDataURL(e.canvas), added: e.added, batch: e.batch}); uiLibrary();
  }
  function openLibrary(){ q('libModal').hidden = false; renderLibrary(); }
  function renderLibrary(){
    const box = q('libCards'), want = q('libFilter').value; box.innerHTML = '';
    const list = S.lib.filter(e => want === 'all' || e.label === want).sort((a, b) => b.added - a.added);
    if(!list.length) box.innerHTML = '<p class="hint" style="padding:20px">No examples here yet.</p>';
    for(const e of list.slice(0, 400)){
      const card = document.createElement('div'); card.className = 'icard on';
      const img = document.createElement('img'); img.src = e.canvas.toDataURL('image/jpeg', 0.7); img.alt = '';
      const sel = document.createElement('select');
      for(const [v, t] of Object.entries(LABELS)){ const o = document.createElement('option'); o.value = v; o.textContent = t; if(v === e.label) o.selected = true; sel.appendChild(o); }
      sel.onchange = () => relabel(e, sel.value);
      const del = document.createElement('button'); del.textContent = 'Remove';
      del.onclick = () => { removeExamples([e.id]); card.remove(); q('libSummary').textContent = `${S.lib.length} examples`; };
      const why = document.createElement('div'); why.className = 'why'; why.textContent = `${e.source}${e.split === 'test' ? ' · test set' : ''} · ${new Date(e.added).toLocaleDateString()}`;
      card.append(img, sel, del, why); box.appendChild(card);
    }
    q('libSummary').textContent = `${S.lib.length} examples${list.length > 400 ? ' (newest 400 shown)' : ''}`;
  }
  async function loadLibrary(){
    for(const r of await DB.all()){
      try{ const im = await imgFrom(r.img); const c = document.createElement('canvas'); c.width = c.height = 128; c.getContext('2d').drawImage(im, 0, 0, 128, 128);
        S.lib.push({id: r.id, label: r.label, source: r.source, split: r.split, canvas: c, fp: fingerprint(c), added: r.added, batch: r.batch || r.id}); } catch(e){}
    }
    S.libVersion = S.lib.length; uiLibrary();
  }
  async function importFolder(files){
    let added = 0; const batch = newBatch();
    for(const f of files){
      if(!/^image\//.test(f.type) && !/\.(jpe?g|png|webp|heic|gif|bmp)$/i.test(f.name)) continue;
      const parts = (f.webkitRelativePath || f.name).split('/'); if(parts.length < 2) continue;
      const label = normLabel(parts[parts.length - 2]);
      try{
        const url = URL.createObjectURL(f), im = await imgFrom(url); URL.revokeObjectURL(url);
        const side = Math.min(im.naturalWidth, im.naturalHeight);
        addExample(crop(im, im.naturalWidth/2, im.naturalHeight/2, side), label, 'folder', true, batch); added++;
      } catch(e){}
    }
    uiLibrary(); flash(`Added ${added} photo${added === 1 ? '' : 's'} from the folder.`);
  }
  async function exportLibrary(){
    await ensureZip(); const z = new JSZip();
    for(const e of S.lib) z.file(`examples/${e.label}/${e.split}_${e.id}.jpg`, toDataURL(e.canvas).split(',')[1], {base64: true});
    z.file('examples/library.json', JSON.stringify(S.lib.map(e => ({id: e.id, label: e.label, split: e.split, source: e.source, added: e.added})), null, 1));
    download(await z.generateAsync({type: 'blob'}), `tracker-library_${new Date().toISOString().slice(0, 10)}.zip`);
  }

  // ---------------------------------------------------------------- verifier
  function strictness(){ return (+(q('vstrict')?.value ?? 40)) / 100; }
  function ready(){
    if(!q('verifyOn')?.checked) return false;
    return S.lib.filter(e => e.split === 'train' && e.label === active()).length >= 3;
  }
  // Score a crop against the library: how close to the target examples vs everything else.
  function judge(canvas){
    const fp = fingerprint(canvas), tgt = active(), T = [], N = [];
    for(const e of S.lib){ if(e.split !== 'train') continue; (e.label === tgt ? T : N).push(cosine(fp, e.fp)); }
    const top = (a, k = 3) => a.length ? a.sort((m, n) => n - m).slice(0, k).reduce((s, x) => s + x, 0) / Math.min(k, a.length) : null;
    const sT = top(T), sN = top(N);
    const base = S.ai.kind === 'ai' ? 0.45 : 0.6;               // with no "not" examples, need at least this much likeness
    const k = strictness();
    const ok = sT != null && sT >= base + 0.25*k && (sN == null || sT - sN >= 0.15*k);
    return {ok, sT, sN, score: sT - (sN ?? base), kind: S.ai.kind, fp};
  }
  // Called by the tracker after colour + shape picked candidates.
  //  fresh = picking a new target: test the top 3, keep the best that LOOKS right
  //  locked = re-check the current target 4× a second; two fails in a row drop it
  function check({best, cands, fresh, s, t, rank}){
    if(!ready() || !S.snap) return {best, v: null};
    const box = b => ({cx: (b.x0 + b.x1 + 1)/2*s, cy: (b.y0 + b.y1 + 1)/2*s, size: Math.max(b.x1 - b.x0 + 1, b.y1 - b.y0 + 1)*s*1.3});
    const judgeBox = b => {
      const bx = box(b), key = `${Math.round(bx.cx/32)},${Math.round(bx.cy/32)},${Math.round(bx.size/24)}`;
      const hitc = S.cache.get(key); if(hitc && t - hitc.t < 300) return hitc.v;
      const v = judge(crop(S.snap, bx.cx, bx.cy, bx.size, 224)); S.cache.set(key, {t, v});
      if(S.cache.size > 200) S.cache.clear();
      return v;
    };
    if(fresh){
      if(t - S.lastFresh < 80) return {best: null, v: null, waiting: true};   // don't hammer the AI every frame
      S.lastFresh = t;
      const ranked = [...cands].sort((a, b) => rank(b) - rank(a)).slice(0, 3);
      let pick = null;
      for(const b of ranked){ b.v = judgeBox(b); if(b.v.ok && (!pick || b.v.score > pick.v.score)) pick = b; }
      for(const b of ranked) if(b !== pick && b.v && Math.abs(b.v.score) < 0.04) suggest(S.snap, box(b), active(), 'unsure', 'library check was borderline', false);
      S.vAt = t; S.vFails = 0; S.lastV = pick ? pick.v : null;
      if(pick && rec()) rec().vChecked++, rec().vPassed++;
      else if(ranked.length && rec()) rec().vChecked++;
      return {best: pick, v: pick ? pick.v : (ranked[0] && ranked[0].v) || null, rejectedAll: !pick && ranked.length > 0};
    }
    if(best && t - S.vAt > 250){
      S.vAt = t; best.v = judgeBox(best);
      if(rec()){ rec().vChecked++; if(best.v.ok) rec().vPassed++; }
      if(best.v.ok){ S.vFails = 0; S.lastV = best.v; }
      else if(++S.vFails >= 2){
        S.vFails = 0; S.lastV = null;
        suggest(S.snap, box(best), 'not-target', 'high', 'locked on, but the library says it is not the target', true);
        event(t, 'verify_fail', 'Lock dropped: colour and shape matched, but it did not look like the examples');
        return {best: null, v: best.v, dropped: true};
      }
    }
    return {best, v: S.lastV};
  }
  function testVerifier(){
    const tgt = active(), test = S.lib.filter(e => e.split === 'test');
    if(!test.length) return flash('Add at least 5 examples of a label so some go to the test set.');
    let right = 0;
    for(const e of test){ const v = judge(e.canvas); if(v.ok === (e.label === tgt)) right++; }
    S.testAcc = {right, n: test.length, pct: Math.round(100*right/test.length), libVersion: S.libVersion};
    uiLibrary();
  }
  // Colour profile learned from the examples. Learns the hue from the target examples,
  // then sets the saturation cut-off BETWEEN the target and the not-target examples
  // (e.g. above a wood floor that has almost the same hue as a basketball).
  function useLibraryColours(){
    const ex = S.lib.filter(e => e.label === active()); if(!ex.length) return flash('No examples for this target yet.');
    const pixels = (list, region) => { const out = []; for(const e of list){ const [x, y, w] = region; const d = e.canvas.getContext('2d').getImageData(x, y, w, w).data;
      for(let i = 0; i < d.length; i += 16){ const h = rgbToHsv(d[i], d[i+1], d[i+2]); if(h[2] >= 0.12) out.push(h); } } return out; };
    const hs = pixels(ex, [32, 32, 64]).filter(h => h[1] >= 0.3 && h[2] >= 0.25 && h[2] <= 0.97);
    if(hs.length < 30) return flash('The examples are too dark or grey to learn a colour from.');
    let cx = 0, cy = 0; for(const h of hs){ cx += Math.cos(h[0]*Math.PI/180); cy += Math.sin(h[0]*Math.PI/180); }
    const mh = (Math.atan2(cy, cx)*180/Math.PI + 360) % 360;
    const hueDev = h => { let d = Math.abs(h[0] - mh); return d > 180 ? 360 - d : d; };
    const tol = Math.round(Math.min(40, Math.max(10, pct(hs.map(hueDev), 0.9) * 1.3)));
    const sat = median(hs.map(h => h[1])), val = median(hs.map(h => h[2]));
    // saturation of not-target pixels that share the target's hue
    const ns = pixels(S.lib.filter(e => e.label !== active()), [0, 0, 128]).filter(h => hueDev(h) < tol).map(h => h[1]);
    // Aim to keep 90% of the target's pixels, but rise above DULLER look-alikes (like a wood floor).
    // Same-colour look-alikes can't be fixed by colour: that's the library double-check's job.
    // Never cut away more than 30% of the target's own pixels.
    const ts = hs.map(h => h[1]);
    let cut = pct(ts, 0.1), note = '';
    const duller = ns.filter(x => x < sat);
    if(duller.length > 50) cut = Math.min(Math.max(cut, pct(duller, 0.95) + 0.03), pct(ts, 0.3));
    if(ns.filter(x => x >= sat).length > 50) note = ' Some not-target examples are the same colour as the target, so keep the library double-check on.';
    const rgb = hsvToRgb(mh, sat, val);
    samples = [{hsv: [mh, sat, val], rgb: `rgb(${rgb.join(',')})`}];
    q('tol').value = tol; q('tol').dispatchEvent(new Event('input'));
    q('strict').value = Math.round(Math.max(20, Math.min(95, 100*cut/sat))); q('strict').dispatchEvent(new Event('input'));
    renderSwatches(); trail = []; last = null; track = null;
    flash(`Colour learned from ${ex.length} examples: hue ${mh.toFixed(0)}°, tolerance ${tol}, strictness ${q('strict').value}%.${note}`);
  }
  function hsvToRgb(h, s, v){ const f = n => { const k = (n + h/60) % 6; return Math.round(255*(v - v*s*Math.max(0, Math.min(k, 4 - k, 1)))); }; return [f(5), f(3), f(1)]; }

  // ------------------------------------------------------------------- inbox
  // The app suggests labels while it runs; nothing joins the library until you accept it.
  function suggest(src, bx, label, conf, reason, checked){
    if(S.frozen || S.inbox.length >= 150 || !src) return;
    const key = label + '|' + reason, now = performance.now();
    if(now - (S.suggestAt[key] || 0) < (conf === 'high' ? 600 : 1000)) return;
    S.suggestAt[key] = now;
    S.inbox.push({id: uid(), canvas: crop(src, bx.cx, bx.cy, bx.size), label, conf, reason, checked});
    uiInboxCount();
  }
  function openInbox(){ q('inboxModal').hidden = false; renderInbox(); }
  function renderInbox(){
    const box = q('inboxCards'); box.innerHTML = '';
    if(!S.inbox.length){ box.innerHTML = '<p class="hint" style="padding:20px">Nothing to review yet. Run the tracker (or record a test run) and the app will suggest labels here.</p>'; }
    for(const it of S.inbox){
      const card = document.createElement('div'); card.className = 'icard' + (it.checked ? ' on' : '');
      const img = document.createElement('img'); img.src = it.canvas.toDataURL('image/jpeg', 0.8); img.alt = '';
      const sel = document.createElement('select');
      for(const [v, t] of Object.entries(LABELS).concat([['skip', 'Skip']])){ const o = document.createElement('option'); o.value = v; o.textContent = t; if(v === it.label) o.selected = true; sel.appendChild(o); }
      sel.onchange = () => { it.label = sel.value; };
      const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = it.checked;
      cb.onchange = () => { it.checked = cb.checked; card.classList.toggle('on', cb.checked); };
      const why = document.createElement('div'); why.className = 'why'; why.textContent = (it.conf === 'unsure' ? 'Unsure: ' : '') + it.reason;
      const top = document.createElement('label'); top.className = 'chk'; top.append(cb, document.createTextNode(' add'));
      card.append(img, top, sel, why); box.appendChild(card);
    }
    q('inboxSummary').textContent = `${S.inbox.length} suggestion${S.inbox.length === 1 ? '' : 's'} · ${S.inbox.filter(i => i.checked).length} ticked`;
  }
  function acceptInbox(){
    let n = 0; const batch = newBatch();
    for(const it of S.inbox) if(it.checked && it.label !== 'skip'){ addExample(it.canvas, it.label, 'inbox', true, batch); n++; }
    S.inbox = []; renderInbox(); uiInboxCount(); uiLibrary();
    flash(`Added ${n} example${n === 1 ? '' : 's'} to the library.`);
  }
  // Live labelling: click a blob (in label mode) or press B / C / N / W
  function labelClick(x, y){
    const mode = q('labelMode')?.value; if(!mode || mode === 'off' || !S.snap) return false;
    if(S.frozen){ flash('Demo mode is on: learning is frozen.'); return true; }
    const b = S.lastBlobs.find(b => Math.abs(x - b.cx) < b.size/2 && Math.abs(y - b.cy) < b.size/2) || {cx: x, cy: y, size: 140};
    addExample(crop(S.snap, b.cx, b.cy, b.size), mode, 'live'); uiLibrary();
    flash(`Saved as ${LABELS[mode] || mode}.`);
    return true;
  }
  function labelCurrent(label){
    if(S.frozen) return flash('Demo mode is on: learning is frozen.');
    const b = S.cur; if(!b || !S.snap) return flash('Nothing is locked on right now.');
    addExample(crop(S.snap, b.cx, b.cy, b.size), label, 'live'); uiLibrary();
    flash(`Saved the current lock as ${LABELS[label]}.`);
  }

  // ---------------------------------------------------------------- recorder
  const rec = () => S.rec;
  function event(t, kind, note, extra){
    const r = rec(); if(!r) return;
    const lastSame = r.events.filter(e => e.kind === kind).at(-1);
    if(lastSame && t - lastSame.t < 700) return;                    // one event per burst
    const e = {t, kind, note, ...(extra || {})}; r.events.push(e);
    if(r.failures.length < 40) r.failures.push({...e, due: t + 500, strip: null});
    uiRecCounts();
  }
  function startRecording(){
    if(S.rec) return;
    S.prevLocked = false;
    const r = S.rec = {name: q('runName').value, startPerf: performance.now(), startDate: new Date(), frames: [], events: [], failures: [],
      pending: [], errors: [], errUnit: null, vChecked: 0, vPassed: 0, chunks: [], media: null, mime: '', locked: 0, holding: 0, total: 0, fpsT: [],
      switches: 0, lost: 0, spikes: 0, replay: S.replay, libVersion: S.libVersion, aiKind: S.ai.kind};
    if(q('recVideo').checked && !S.replay && liveStream()){
      const types = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'];
      r.mime = types.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
      try{ r.media = new MediaRecorder(liveStream(), r.mime ? {mimeType: r.mime, videoBitsPerSecond: 2500000} : undefined);
        r.media.ondataavailable = e => e.data.size && r.chunks.push(e.data); r.media.start(1000); } catch(e){ r.media = null; }
    }
    q('recBtn').textContent = '■ Stop recording'; q('recBtn').classList.add('active'); q('reportBtn').disabled = true;
    q('recDot').hidden = false; uiRecCounts();
  }
  async function stopRecording(){
    const r = S.rec; if(!r) return;
    if(r.media && r.media.state !== 'inactive'){ await new Promise(ok => { r.media.onstop = ok; r.media.stop(); }); }
    r.endPerf = performance.now(); S.rec = null; S.lastRun = r;
    await finishStrips(r, true);
    q('recBtn').textContent = '● Record a test run'; q('recBtn').classList.remove('active'); q('reportBtn').disabled = false;
    q('recDot').hidden = true;
    const sum = summarize(r); S.lastSummary = sum; uiScore(sum);
  }
  // Per-frame hook from the tracker
  function onFrame(f){
    S.frameN++;
    if(f.box) S.cur = f.box; else S.cur = null;
    S.lastBlobs = f.blobs || [];
    // status line: does the lock look like the target?
    const v = f.verdict;
    if(f.locked){
      if(v && v.ok){ q('st').textContent = 'CONFIRMED'; q('stsub').textContent = `looks like a ${LABELS[active()]} (${v.sT.toFixed(2)} vs ${v.sN == null ? '—' : v.sN.toFixed(2)})`; }
      else q('stsub').textContent = ready() ? 'checking with the library…' : 'colour + shape only (library not in use)';
    } else q('stsub').textContent = f.holding ? 'holding: ignoring a jump' : (f.waiting ? 'checking candidates…' : ' ');

    autoSuggest(f);
    const r = rec();
    if(!r){ S.prevLocked = f.locked; return; }
    r.total++; if(f.locked) r.locked++; if(f.holding) r.holding++;
    r.fpsT.push(f.t);
    if(f.jumped){ r.switches++; event(f.t, 'jump', 'The lock jumped to a different blob'); }
    if(S.prevLocked && !f.locked && !f.holding){ r.lost++; event(f.t, 'lost', 'Lost the target'); }
    const spikeLimit = f.nowW ? 60 : f.W*4;                         // 60 ft/s ≈ 40 mph: not a thrown basketball indoors
    if(f.locked && f.speed > spikeLimit){ r.spikes++; event(f.t, 'spike', `Impossible speed: ${f.speed.toFixed(0)} ${f.nowW ? 'ft/s' : 'px/s'}`); }
    // Self-grading: compare each prediction with where the target really was `ahead` seconds later
    // (skip a track's first 0.2 s: too few points yet; only compare like with like)
    if(f.locked && f.pr){
      const at = f.ahead*1000;
      if(f.trackAge > 200) r.pending.push({due: f.t + at, track: f.trackId, mode: f.mode, w: f.pr.atW ? f.pr.atW(f.ahead) : null, p: f.pr.at(f.ahead)});
      while(r.pending.length && r.pending[0].due <= f.t + 8){
        const p = r.pending.shift();
        if(p.track !== f.trackId || p.mode !== f.mode || f.t - p.due > 60) continue;
        let err = null, unit;
        if(p.w && f.nowW){ err = Math.hypot(p.w.x - f.nowW.x, p.w.up - f.nowW.up, p.w.f - f.nowW.f); unit = 'ft'; }
        else if(p.p){ err = Math.hypot(p.p[0] - f.u, p.p[1] - f.v); unit = 'px'; }
        if(err == null) continue;
        if(r.errUnit && r.errUnit !== unit){ r.errors = []; }
        r.errUnit = unit; r.errors.push(err);
        if((unit === 'ft' && err > 1) || (unit === 'px' && err > f.W*0.12)) event(f.t, 'predmiss', `Prediction missed by ${err.toFixed(unit === 'ft' ? 2 : 0)} ${unit}`, {err});
      }
    } else r.pending = [];
    if(r.frames.length < 200000) r.frames.push([Math.round(f.t - r.startPerf), f.locked ? 1 : 0, f.holding ? 1 : 0, f.trackId ?? '', f.u?.toFixed(1) ?? '', f.v?.toFixed(1) ?? '', f.d?.toFixed(1) ?? '',
      f.mode || '', f.nowW ? f.nowW.x.toFixed(2) : '', f.nowW ? f.nowW.up.toFixed(2) : '', f.nowW ? f.nowW.f.toFixed(2) : '', f.speed?.toFixed(2) ?? '',
      v ? (v.ok ? 1 : 0) : '', v && v.sT != null ? v.sT.toFixed(3) : '', f.ncands ?? '']);
    S.prevLocked = f.locked;
    if(S.frameN % 15 === 0) uiRecCounts();
  }
  // Suggestions the app makes on its own
  function autoSuggest(f){
    if(S.frozen || !S.snap) return;
    if(f.locked && f.box && f.trackAge > 600 && !f.jumped && (!f.verdict || f.verdict.ok) && f.speed < (f.nowW ? 40 : f.W*2))
      suggest(S.snap, f.box, active(), 'high', 'steady track, believable speed', true);
    if(f.locked && f.box && f.jumped)
      suggest(S.snap, f.box, active(), 'unsure', 'the lock jumped to this', false);
    for(const b of f.rejected || []){
      if(b.reason === 'shape') suggest(S.snap, b, 'not-target', 'high', 'right colour, wrong shape', true);
      else if(f.locked) suggest(S.snap, b, 'not-target', 'unsure', 'right colour and shape, but not the one being tracked', false);
    }
  }
  // Ring buffer of small frames (with overlays) for filmstrips
  function afterDraw(t){
    if(!S.rec && !S.fails.length && !(S.lastRun && S.lastRun.failures.some(x => !x.strip))) { if(S.frameN % 2) return; }
    if(S.frameN % 2) return;
    let c = S.pool.pop(); if(!c){ c = document.createElement('canvas'); c.width = 320; c.height = 180; }
    c.getContext('2d').drawImage(cv, 0, 0, 320, 180); S.ring.push({t, c});
    while(S.ring.length && t - S.ring[0].t > 2500) S.pool.push(S.ring.shift().c);
    if(S.rec) finishStrips(S.rec, false);
  }
  async function finishStrips(r, force){
    const now = S.ring.length ? S.ring.at(-1).t : 0;
    for(const fl of r.failures){
      if(fl.strip || (!force && now < fl.due)) continue;
      const span = S.ring.filter(x => x.t >= fl.t - 1000 && x.t <= fl.t + 500);
      if(!span.length){ fl.strip = 'none'; continue; }
      const pick = []; for(let i = 0; i < 8; i++) pick.push(span[Math.round(i*(span.length - 1)/7)]);
      const W = 8*320, H = 180 + 22, c = document.createElement('canvas'); c.width = W; c.height = H;
      const g = c.getContext('2d'); g.fillStyle = '#0e1214'; g.fillRect(0, 0, W, H);
      pick.forEach((fr, i) => {
        g.drawImage(fr.c, i*320, 22);
        const dt = (fr.t - fl.t)/1000; g.fillStyle = Math.abs(dt) < 0.04 ? '#ff5a4f' : '#dfe7ea';
        g.font = '600 14px monospace'; g.fillText(`${dt >= 0 ? '+' : ''}${dt.toFixed(2)} s`, i*320 + 6, 16);
        if(Math.abs(dt) === Math.min(...pick.map(p => Math.abs((p.t - fl.t)/1000)))){ g.strokeStyle = '#ff5a4f'; g.lineWidth = 4; g.strokeRect(i*320 + 2, 24, 316, 176); }
      });
      fl.strip = await new Promise(ok => c.toBlob(ok, 'image/jpeg', 0.8));
    }
  }

  // ------------------------------------------------------------------ report
  function summarize(r){
    const dur = ((r.endPerf || performance.now()) - r.startPerf)/1000;
    const gaps = []; for(let i = 1; i < r.fpsT.length; i++) gaps.push(r.fpsT[i] - r.fpsT[i-1]);
    const counts = {}; for(const e of r.events) counts[e.kind] = (counts[e.kind] || 0) + 1;
    const libCounts = {}; for(const e of S.lib) libCounts[e.label] = (libCounts[e.label] || 0) + 1;
    return {
      run: r.name, version: `${CODE_VERSION}.L${r.libVersion}`, date: r.startDate.toISOString(), replay: r.replay,
      durationS: +dur.toFixed(1), frames: r.total,
      fpsAvg: gaps.length ? +(1000/(gaps.reduce((s, x) => s + x, 0)/gaps.length)).toFixed(1) : null,
      fpsLow: gaps.length ? +(1000/pct(gaps, 0.95)).toFixed(1) : null,
      lockedPct: r.total ? +(100*r.locked/r.total).toFixed(1) : 0, holdingFrames: r.holding,
      lockSwitches: r.switches, lost: r.lost, speedSpikes: r.spikes,
      predError: {unit: r.errUnit, n: r.errors.length, median: r.errors.length ? +median(r.errors).toFixed(3) : null, p90: r.errors.length ? +pct(r.errors, 0.9).toFixed(3) : null, worst: r.errors.length ? +Math.max(...r.errors).toFixed(3) : null},
      verifier: {kind: r.aiKind, checks: r.vChecked, passed: r.vPassed, library: libCounts, testAccuracy: S.testAcc || null},
      failureCounts: counts, model: q('model').value, calibrated: !!calib(), walls: walls.length
    };
  }
  function settingsSnapshot(){
    const ids = ['tol', 'strict', 'minpx', 'merge', 'model', 'balld', 'ahead', 'win', 'minspd', 'rlen', 'rgap', 'preset', 'vstrict', 'verifyOn', 'mirror'];
    const out = {}; for(const id of ids){ const el = q(id); if(el) out[id] = el.type === 'checkbox' ? el.checked : el.value; }
    const c = calib();
    return {...out, samples: samples.map(s => s.hsv.map(v => +v.toFixed(3))), rulers, walls,
      camera: c ? {heightFt: +c.h.toFixed(3), tiltDeg: +(c.th*180/Math.PI).toFixed(2), focalPx: +c.f.toFixed(1)} : null,
      fingerprint: S.ai.kind === 'ai' ? S.ai.name : 'basic', codeVersion: CODE_VERSION, userAgent: navigator.userAgent};
  }
  const PRIORITY = {user: 0, verify_fail: 1, jump: 2, spike: 3, lost: 4, predmiss: 5};
  async function makeReport(){
    const r = S.lastRun; if(!r) return flash('Record a test run first.');
    q('reportBtn').disabled = true; q('reportBtn').textContent = 'Building report…';
    try{
      await ensureZip(); await finishStrips(r, true);
      const sum = summarize(r), settings = settingsSnapshot(), hist = history();
      const prev = [...hist].reverse().find(h => h.run === sum.run && h.date !== sum.date) || null;
      hist.push(sum); saveHistory(hist);
      const worst = [...r.failures].filter(f => f.strip && f.strip !== 'none')
        .sort((a, b) => (PRIORITY[a.kind] - PRIORITY[b.kind]) || ((b.err || 0) - (a.err || 0))).slice(0, 10);
      const z = new JSZip(), stamp = sum.date.slice(0, 19).replace(/[:T]/g, '-');
      worst.forEach((f, i) => { f.file = `failures/${String(i + 1).padStart(2, '0')}_${f.kind}.jpg`; z.file(f.file, f.strip); });
      const head = ['ms', 'locked', 'holding', 'track', 'u_px', 'v_px', 'diam_px', 'mode', 'x_ft', 'up_ft', 'out_ft', 'speed', 'verified', 'verifier_sim', 'candidates'];
      z.file('log.csv', head.join(',') + '\n' + r.frames.map(row => row.join(',')).join('\n'));
      z.file('summary.json', JSON.stringify({summary: sum, previous: prev, events: r.events.map(e => ({ms: Math.round(e.t - r.startPerf), kind: e.kind, note: e.note}))}, null, 2));
      z.file('settings.json', JSON.stringify(settings, null, 2));
      if(r.chunks.length){ const ext = (r.mime || '').includes('mp4') ? 'mp4' : 'webm'; z.file(`video.${ext}`, new Blob(r.chunks, {type: r.mime || 'video/webm'})); }
      z.file('report.html', reportHTML(sum, prev, worst, settings, r));
      const blob = await z.generateAsync({type: 'blob'});
      download(blob, `tracker-report_${sum.run.replace(/[^A-Za-z0-9]+/g, '-')}_${stamp}.zip`);
      S.lastReport = sum;
      flash('Report saved to your Downloads. Attach the zip in the chat with Claude.');
    } finally { q('reportBtn').disabled = false; q('reportBtn').textContent = 'Make report'; }
  }
  function reportHTML(s, p, worst, settings, r){
    const esc = x => String(x).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
    const better = (k, lowGood) => { if(!p) return ''; const a = get(p, k), b = get(s, k); if(a == null || b == null || a === b) return '<span class="eq">same</span>';
      const good = lowGood ? b < a : b > a; return `<span class="${good ? 'up' : 'down'}">${good ? '▲ better' : '▼ worse'} (was ${a})</span>`; };
    const get = (o, k) => k.split('.').reduce((v, x) => v == null ? v : v[x], o);
    const u = s.predError.unit || '';
    const rows = [
      ['Locked on (frames)', `${s.lockedPct}%`, better('lockedPct', false)],
      ['Lock switches', s.lockSwitches, better('lockSwitches', true)],
      ['Target lost', s.lost, better('lost', true)],
      ['Impossible speeds', s.speedSpikes, better('speedSpikes', true)],
      [`Prediction miss, typical (${u})`, s.predError.median ?? '—', better('predError.median', true)],
      [`Prediction miss, worst 10% (${u})`, s.predError.p90 ?? '—', better('predError.p90', true)],
      ['Frame rate', `${s.fpsAvg ?? '—'} fps (low ${s.fpsLow ?? '—'})`, better('fpsAvg', false)],
      ['Library checks passed', s.verifier.checks ? `${s.verifier.passed}/${s.verifier.checks}` : 'not in use', '']
    ];
    return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tracker report · ${esc(s.run)}</title>
<style>:root{--bg:#0e1214;--p:#161c1f;--l:#273136;--fg:#dfe7ea;--d:#7f8f96;--g:#3cff8f;--r:#ff5a4f;--b:#3aa0ff}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,Menlo,monospace;padding:24px max(16px,4vw)}
h1{font:700 26px system-ui;margin:0}h2{font:600 12px ui-monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--d);margin:28px 0 10px}
.meta{color:var(--d)}table{border-collapse:collapse;width:100%;max-width:820px}td{border-bottom:1px solid var(--l);padding:7px 10px}td:nth-child(2){font-weight:700}
.up{color:var(--g)}.down{color:var(--r)}.eq{color:var(--d)}.f{background:var(--p);border:1px solid var(--l);border-radius:6px;padding:12px;margin:0 0 14px}
.f img{width:100%;border-radius:4px;display:block;margin-top:8px}.k{color:var(--r);font-weight:700}pre{background:var(--p);padding:12px;border-radius:6px;overflow:auto}</style></head><body>
<h1>Tracker report</h1><p class="meta">${esc(s.run)} · ${new Date(s.date).toLocaleString()} · version ${esc(s.version)} · ${s.durationS}s · ${s.frames} frames${s.replay ? ' · replayed video' : ''}</p>
<h2>Scoreboard${p ? ` (compared with version ${esc(p.version)})` : ''}</h2><table>${rows.map(rw => `<tr><td>${rw[0]}</td><td>${esc(rw[1])}</td><td>${rw[2]}</td></tr>`).join('')}</table>
<h2>What failed</h2><p class="meta">${Object.entries(s.failureCounts).map(([k, n]) => `${k}: ${n}`).join(' · ') || 'No failures caught.'}</p>
${worst.map((f, i) => `<div class="f"><span class="k">#${i + 1} ${esc(f.kind)}</span> at ${((f.t - r.startPerf)/1000).toFixed(2)} s · ${esc(f.note)}<img src="${f.file}" alt="Filmstrip of failure ${i + 1}"></div>`).join('')}
<h2>Library</h2><p class="meta">${Object.entries(s.verifier.library).map(([k, n]) => `${k}: ${n}`).join(' · ') || 'empty'} · fingerprint: ${esc(settings.fingerprint)}${s.verifier.testAccuracy ? ` · test accuracy ${s.verifier.testAccuracy.right}/${s.verifier.testAccuracy.n} (${s.verifier.testAccuracy.pct}%)` : ''}</p>
<h2>Settings</h2><pre>${esc(JSON.stringify(settings, null, 2))}</pre></body></html>`;
  }
  const history = () => { try{ return JSON.parse(localStorage.getItem('tracker-history') || '[]'); } catch(e){ return []; } };
  const saveHistory = h => { try{ localStorage.setItem('tracker-history', JSON.stringify(h.slice(-200))); } catch(e){} };
  let zipLoading = null;
  const ensureZip = () => window.JSZip ? Promise.resolve() : (zipLoading ||= loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js'));

  // ------------------------------------------------------------------ replay
  function replayFile(file){
    if(liveStream()) stop();
    S.replay = true; S.replayBase = performance.now();
    vid.srcObject = null; vid.src = URL.createObjectURL(file); vid.loop = false;
    q('msg').hidden = true; q('hud').hidden = false;
    const fail = () => {
      vid.onerror = null; vid.onended = null; S.replay = false; vid.removeAttribute('src');
      q('msg').hidden = false; q('msg').textContent = 'That file could not be played here. Use a video saved by the tracker (.webm), or an .mp4. iPhone .MOV files play in Safari but often not in Chrome.';
      flash('That video could not be played.');
    };
    vid.onerror = fail;
    vid.onended = async () => { vid.onerror = null; if(S.rec) await stopRecording(); S.replay = false; flash('Replay finished. Click "Make report".'); };
    vid.play().then(() => { trail = []; last = null; track = null; startRecording(); nextFrame(); }).catch(fail);
  }

  // ------------------------------------------------------------ saved setup
  const SETUP_IDS = ['tol', 'strict', 'minpx', 'merge', 'model', 'balld', 'ahead', 'win', 'minspd', 'rlen', 'rgap', 'preset', 'vstrict', 'verifyOn', 'mirror', 'trail', 'showmap', 'pred', 'recVideo', 'runName'];
  let lastSaved = '';
  function saveSetup(){
    const o = {samples, rulers, walls}; for(const id of SETUP_IDS){ const el = q(id); if(el) o[id] = el.type === 'checkbox' ? el.checked : el.value; }
    const j = JSON.stringify(o); if(j === lastSaved) return; lastSaved = j;
    try{ localStorage.setItem('tracker-setup', j); } catch(e){}
  }
  function restoreSetup(){
    let o = null; try{ o = JSON.parse(localStorage.getItem('tracker-setup') || 'null'); } catch(e){}
    if(!o) return;
    for(const id of SETUP_IDS){ const el = q(id); if(!el || !(id in o)) continue; if(el.type === 'checkbox') el.checked = o[id]; else el.value = o[id]; el.dispatchEvent(new Event('input')); }
    if(Array.isArray(o.samples)) samples = o.samples;
    if(o.rulers) rulers = o.rulers;
    if(Array.isArray(o.walls)) walls = o.walls;
    renderSwatches(); updateCalStatus();
  }

  // ---------------------------------------------------------------------- UI
  let flashT = 0;
  function flash(msg){ const el = q('learnMsg'); if(!el) return; el.textContent = msg; el.hidden = false; clearTimeout(flashT); flashT = setTimeout(() => el.hidden = true, 4500); }
  function uiLibrary(){
    const c = {}; const t = {}; for(const e of S.lib){ c[e.label] = (c[e.label] || 0) + 1; if(e.split === 'test') t[e.label] = (t[e.label] || 0) + 1; }
    q('libCounts').innerHTML = Object.keys(LABELS).map(k => `<span><b>${c[k] || 0}</b> ${LABELS[k]}${t[k] ? ` <i>(${t[k]} test)</i>` : ''}</span>`).join('');
    const ai = S.ai.status === 'loading' ? 'loading the AI model…' : S.ai.kind === 'ai' ? `AI ready (${S.ai.name})` : 'basic (AI model did not load)';
    q('aiStatus').textContent = `Fingerprint: ${ai}`;
    const need = 3 - S.lib.filter(e => e.split === 'train' && e.label === active()).length;
    q('verifyState').textContent = need > 0 ? `Needs ${need} more ${LABELS[active()]} example${need === 1 ? '' : 's'} before the double-check turns on.` : 'Double-check is active.';
    q('testAcc').textContent = S.testAcc ? `Test set: right ${S.testAcc.right} of ${S.testAcc.n} (${S.testAcc.pct}%)${S.testAcc.libVersion !== S.libVersion ? ' · library changed since' : ''}` : '';
  }
  function uiInboxCount(){ const n = S.inbox.length; q('inboxBtn').textContent = `Review Inbox (${n})`; q('inboxBtn').classList.toggle('active', n > 0); }
  function uiRecCounts(){
    const r = S.rec; if(!r){ return; }
    q('recCounts').textContent = `${((performance.now() - r.startPerf)/1000).toFixed(0)} s · ${r.total} frames · ${r.events.length} problems caught · ${S.inbox.length} suggestions`;
  }
  function uiScore(s){
    const u = s.predError.unit || '';
    q('recCounts').innerHTML = `<b>Last run:</b> locked ${s.lockedPct}% · switches ${s.lockSwitches} · lost ${s.lost} · spikes ${s.speedSpikes} · miss ${s.predError.median == null ? '—' : `${s.predError.median} ${u}`} typical`;
  }
  function setFrozen(on){
    S.frozen = on;
    for(const id of ['labelMode', 'inboxBtn', 'importBtn', 'clearLib', 'acceptInbox', 'undoLib', 'undoLib2']) if(q(id)) q(id).disabled = on;
    flash(on ? 'Demo mode: learning is frozen and locks need the library double-check.' : 'Demo mode off: learning is back on.');
  }
  function applyPreset(){
    const p = active();
    if(p === 'ball'){ q('model').value = 'ball'; q('balld').value = q('balld').value || '9.4'; }
    else { q('model').value = 'turn'; }
    q('model').dispatchEvent(new Event('change')); S.cache.clear(); uiLibrary();
  }

  function init(){
    q('runName').innerHTML = RUNS.map(r => `<option>${r}</option>`).join('');
    q('importBtn').onclick = () => q('folderIn').click();
    q('folderIn').onchange = e => { importFolder([...e.target.files]); e.target.value = ''; };
    q('exportLib').onclick = exportLibrary;
    q('clearLib').onclick = e => {
      const b = e.currentTarget;
      if(b.dataset.armed){ S.lib = []; S.libVersion++; S.cache.clear(); DB.clear(); uiLibrary(); b.textContent = 'Clear library'; delete b.dataset.armed; flash('Library cleared.'); }
      else { b.dataset.armed = '1'; b.textContent = 'Click again to clear'; setTimeout(() => { b.textContent = 'Clear library'; delete b.dataset.armed; }, 3000); }
    };
    q('testVerifier').onclick = testVerifier;
    q('useColours').onclick = useLibraryColours;
    q('preset').onchange = applyPreset;
    q('vstrict').oninput = e => { q('vstrictv').textContent = e.target.value; S.cache.clear(); };
    q('inboxBtn').onclick = openInbox;
    q('viewLib').onclick = openLibrary;
    q('undoLib').onclick = undoLast;
    q('undoLib2').onclick = undoLast;
    q('libFilter').onchange = renderLibrary;
    q('closeLib').onclick = () => q('libModal').hidden = true;
    q('closeInbox').onclick = () => q('inboxModal').hidden = true;
    q('acceptInbox').onclick = acceptInbox;
    q('clearInbox').onclick = () => { S.inbox = []; renderInbox(); uiInboxCount(); };
    q('tickAll').onclick = () => { const on = !S.inbox.every(i => i.checked); S.inbox.forEach(i => i.checked = on); renderInbox(); };
    q('recBtn').onclick = () => S.rec ? stopRecording() : startRecording();
    q('reportBtn').onclick = makeReport;
    q('replayBtn').onclick = () => q('replayIn').click();
    q('replayIn').onchange = e => { if(e.target.files[0]) replayFile(e.target.files[0]); e.target.value = ''; };
    q('flagBtn').onclick = () => flagWrong();
    q('demoMode').onchange = e => setFrozen(e.target.checked);
    document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
      document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
      document.querySelectorAll('[data-pane]').forEach(p => p.hidden = p.dataset.pane !== b.dataset.tab);
    });
    document.addEventListener('keydown', e => {
      if((e.target instanceof Element && e.target.closest('input, select, textarea')) || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if(k === 'b') labelCurrent('ball'); else if(k === 'c') labelCurrent('rc-car'); else if(k === 'n') labelCurrent('not-target');
      else if(k === 'w') flagWrong(); else if(k === 'r') q('recBtn').click(); else if(k === 'i') openInbox();
      else if(k === 'escape'){ q('inboxModal').hidden = true; q('libModal').hidden = true; }
    });
    restoreSetup();
    setInterval(saveSetup, 2000); addEventListener('beforeunload', saveSetup);
    uiInboxCount(); uiLibrary();
    loadLibrary();
    loadAI();
  }
  function flagWrong(){
    const t = S.ring.length ? S.ring.at(-1).t : performance.now();
    event(t, 'user', 'You flagged this moment as wrong');
    if(S.cur && S.snap && !S.frozen){ suggest(S.snap, S.cur, 'not-target', 'high', 'you pressed W while locked on this', true); }
    trail = []; last = null; track = null;
    flash(S.rec ? 'Flagged. It will be in the report.' : 'Flagged. Record a run to keep flags in a report.');
  }

  return {init, check, onFrame, afterDraw, labelClick, isReplay: () => S.replay, replayClock: mt => S.replayBase + mt*1000, setSnap: c => { S.snap = c; },
          // exposed for testing
          _S: S, addExample, undoLast, removeExamples, judge, testVerifier, startRecording, stopRecording, makeReport, summarize, suggest, acceptInbox, ready, fingerprint};
})();
