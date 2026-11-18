/* SoundSense account layer. Load AFTER app.js:  <script src="auth.js"></script>
 *
 * One sign in / sign up for everyone. The server decides the role; developer tabs
 * only appear when /api/me says role === 'developer'. Hiding tabs is cosmetic, the
 * real protection is require_developer on the backend (auth.py).
 *
 * This file takes over the old developer-login code in app.js without editing it:
 * it replaces openAuthModal, handleAccountBtnClick and the login/logout handlers.
 */
(function () {
  'use strict';

  // ───────── config ─────────
  // If the page is opened from Live Server / file://, talk to the Python backend directly.
  var API_BASE = window.SOUNDSENSE_API_BASE ||
    ((location.protocol === 'file:' || /^(5500|5501|5173|3000)$/.test(location.port)) ? 'http://localhost:8000' : '');
  var TOKEN_KEY = 'soundsense_token';

  var currentUser = null;
  var mode = 'login'; // 'login' | 'signup'
  var lastAccountClick = 0;

  var $ = function (id) { return document.getElementById(id); };
  var toast = function (t, m, tone, ttl) { if (typeof showToast === 'function') showToast(t, m, tone || 'info', ttl); };

  function getToken() { try { return localStorage.getItem(TOKEN_KEY); } catch (e) { return null; } }
  function setToken(t) {
    try { if (t) localStorage.setItem(TOKEN_KEY, t); else localStorage.removeItem(TOKEN_KEY); } catch (e) { /* ignore */ }
  }

  // ───────── fetch wrapper: adds the token and the backend address to every /api/ call ─────────
  var rawFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    if (typeof input === 'string' && input.indexOf('/api/') !== -1) {
      if (input.charAt(0) === '/' && API_BASE) input = API_BASE + input;
      var token = getToken();
      if (token) {
        init = Object.assign({}, init);
        var h = new Headers(init.headers || {});
        if (!h.has('Authorization')) h.set('Authorization', 'Bearer ' + token);
        init.headers = h;
      }
    }
    return rawFetch(input, init);
  };

  function api(path, opts) {
    opts = opts || {};
    var headers = { 'Content-Type': 'application/json' };
    var token = getToken();
    if (token) headers.Authorization = 'Bearer ' + token;
    return rawFetch(API_BASE + path, {
      method: opts.method || 'GET',
      headers: headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) {
          var msg = data.detail;
          if (Array.isArray(msg)) msg = msg.map(function (d) { return d.msg; }).join(' ');
          var err = new Error(typeof msg === 'string' && msg ? msg : 'Request failed (' + res.status + ')');
          err.status = res.status;
          throw err;
        }
        return data;
      });
    });
  }

  // ───────── remove the old demo login (prefilled credentials, instant login) ─────────
  try { sessionStorage.removeItem('soundsense_dev_auth'); localStorage.removeItem('soundsense_dev_auth'); } catch (e) { /* ignore */ }
  if (typeof setDeveloperMode === 'function') setDeveloperMode(false); // undo anything app.js restored client-side

  var oldForm = $('auth-login-form');
  var form = oldForm.cloneNode(true); // clone drops the old listeners
  oldForm.parentNode.replaceChild(form, oldForm);

  var demo = form.querySelector('.auth-demo-banner');
  if (demo) demo.remove();

  var emailInput = form.querySelector('#auth-username');
  var passInput = form.querySelector('#auth-password');
  var errBox = form.querySelector('#auth-error-msg');
  var submitBtn = form.querySelector('#btn-login-submit');
  var idleLbl = form.querySelector('#btn-login-idle');
  var busyLbl = form.querySelector('#btn-login-busy');
  emailInput.value = ''; passInput.value = '';
  emailInput.type = 'email';
  emailInput.placeholder = 'you@example.com';
  passInput.placeholder = 'At least 8 characters';
  form.querySelector('label[for="auth-username"]').textContent = 'Email';

  // name field (signup only): copy of the email field so it keeps the same styling
  var emailField = emailInput.closest('.auth-field');
  var nameField = emailField.cloneNode(true);
  nameField.hidden = true;
  var nameInput = nameField.querySelector('input');
  nameInput.id = 'auth-name'; nameInput.type = 'text'; nameInput.value = '';
  nameInput.placeholder = 'Your name'; nameInput.autocomplete = 'name'; nameInput.required = false;
  var nameLabel = nameField.querySelector('label');
  nameLabel.setAttribute('for', 'auth-name'); nameLabel.textContent = 'Name';
  emailField.parentNode.insertBefore(nameField, emailField);

  // "Create an account" / "Sign in" switch
  var switchRow = document.createElement('p');
  switchRow.style.cssText = 'margin:12px 0 0;text-align:center;font-size:0.85rem;opacity:0.9;';
  var switchText = document.createElement('span');
  var switchBtn = document.createElement('button');
  switchBtn.type = 'button';
  switchBtn.style.cssText = 'background:none;border:0;padding:0 0 0 6px;color:var(--cyan,#38bdf8);cursor:pointer;font:inherit;text-decoration:underline;';
  switchRow.appendChild(switchText); switchRow.appendChild(switchBtn);
  form.appendChild(switchRow);

  var titleEl = $('auth-modal-title');
  var eyebrowEl = titleEl.parentNode.querySelector('.modal-eyebrow');
  var descEl = document.querySelector('#auth-modal .auth-modal-desc');

  function showError(msg) {
    errBox.textContent = msg; errBox.hidden = false; errBox.style.display = 'block';
  }
  function clearError() { errBox.hidden = true; errBox.style.display = 'none'; }

  function setMode(m) {
    mode = m;
    var signup = m === 'signup';
    nameField.hidden = !signup;
    nameInput.required = signup;
    passInput.autocomplete = signup ? 'new-password' : 'current-password';
    titleEl.textContent = signup ? 'Create your account' : 'Sign in';
    if (eyebrowEl) eyebrowEl.textContent = 'Account';
    if (descEl) descEl.textContent = signup
      ? 'Create an account to keep a history of your predictions. Everyone can still use the studio without one.'
      : 'Sign in to see your prediction history. Developer accounts also unlock model training and system analytics.';
    var label = idleLbl.querySelector('span');
    if (label) label.textContent = signup ? 'Create account' : 'Sign in';
    switchText.textContent = signup ? 'Already have an account?' : 'New here?';
    switchBtn.textContent = signup ? 'Sign in' : 'Create an account';
    clearError();
  }
  switchBtn.addEventListener('click', function () { setMode(mode === 'login' ? 'signup' : 'login'); });

  function setBusy(on) {
    submitBtn.disabled = on; idleLbl.hidden = on; busyLbl.hidden = !on;
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var email = emailInput.value.trim();
    var password = passInput.value;
    var name = nameInput.value.trim();
    if (!email || !password || (mode === 'signup' && !name)) { showError('Fill in all the fields.'); return; }
    clearError(); setBusy(true);

    var request = mode === 'signup'
      ? api('/api/auth/signup', { method: 'POST', body: { name: name, email: email, password: password } })
      : api('/api/auth/login', { method: 'POST', body: { username: email, password: password } });

    request.then(function (data) {
      var wantedView = (typeof pendingDevView !== 'undefined') ? pendingDevView : null;
      setToken(data.token);
      applyUser(data.user);
      closeAuthModal();
      toast(mode === 'signup' ? 'Account created' : 'Signed in',
        data.user.role === 'developer' ? 'Developer tools are unlocked.' : 'Welcome, ' + data.user.name + '.', 'success', 3500);
      if (wantedView && data.user.role === 'developer') switchView(wantedView);
    }).catch(function (err) {
      showError(err.status ? err.message : 'Cannot reach the server. Is the backend running?');
    }).then(function () { setBusy(false); });
  });

  // ───────── modal open (replaces the version that prefilled developer credentials) ─────────
  window.openAuthModal = function () {
    if (currentUser && currentUser.role !== 'developer' && typeof pendingDevView !== 'undefined' && pendingDevView) {
      pendingDevView = null;
      toast('Developer access required', 'Your account is a standard account.', 'warn', 3500);
      return;
    }
    var modal = $('auth-modal');
    setMode('login');
    emailInput.value = ''; passInput.value = ''; nameInput.value = '';
    modal.hidden = false; modal.removeAttribute('hidden'); modal.style.display = 'grid';
    document.body.style.overflow = 'hidden';
    var pop = $('account-menu-popover');
    if (pop) { pop.hidden = true; pop.style.display = 'none'; }
    setTimeout(function () { emailInput.focus(); }, 60);
  };

  // account button: menu when signed in, login modal otherwise
  window.handleAccountBtnClick = function (e) {
    if (e) {
      if (e.preventDefault) e.preventDefault();
      if (e.stopPropagation) e.stopPropagation();
      if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    }
    var now = Date.now();
    if (now - lastAccountClick < 250) return;
    lastAccountClick = now;
    if (currentUser) toggleAccountMenu(); else window.openAuthModal();
  };

  // ───────── account UI ─────────
  var accountBtn = $('account-btn');
  var popover = $('account-menu-popover');
  var oldLogout = $('btn-logout');
  var logoutBtn = oldLogout.cloneNode(true); // drops the old sign-out handler
  oldLogout.parentNode.replaceChild(logoutBtn, oldLogout);

  logoutBtn.addEventListener('click', function () {
    api('/api/auth/logout', { method: 'POST' }).catch(function () { /* token is stateless */ });
    setToken(null);
    closeAccountMenu();
    applyUser(null);
    toast('Signed out', 'You are back in guest mode.', 'info', 3000);
  });

  // "Prediction history" entry in the account menu
  var historyItem = document.createElement('button');
  historyItem.type = 'button';
  historyItem.className = 'account-menu-item';
  historyItem.id = 'menu-history';
  historyItem.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/></svg><span>Prediction history</span>';
  var links = popover.querySelector('.account-menu-links');
  links.insertBefore(historyItem, links.firstChild);
  historyItem.addEventListener('click', function () { closeAccountMenu(); openHistory(); });

  function paintAccount(u) {
    var initial = (u.name || u.email || '?').charAt(0).toUpperCase();
    var isDev = u.role === 'developer';
    accountBtn.classList.add('logged-in');
    accountBtn.title = 'Signed in as ' + u.email;
    $('account-avatar').innerHTML = '<span class="avatar-letter">' + initial + '</span>';
    $('popover-initial').textContent = initial;
    $('popover-name').textContent = u.name;
    $('popover-email').textContent = u.email;
    var badge = popover.querySelector('.popover-role-badge');
    if (badge) badge.textContent = isDev ? '🛡️ Developer' : 'Standard account';
    var dot = $('dev-badge-dot');
    if (dot) dot.style.display = isDev ? 'block' : 'none';
    ['menu-goto-training', 'menu-goto-analytics'].forEach(function (id) {
      var el = $(id); if (el) el.style.display = isDev ? '' : 'none';
    });
    var label = logoutBtn.querySelector('span');
    if (label) label.textContent = 'Sign out';
  }

  function applyUser(u) {
    currentUser = u;
    if (u && u.role === 'developer') {
      setDeveloperMode(true, { name: u.name, email: u.email, role: 'developer' });
    } else {
      setDeveloperMode(false); // hides dev tabs and leaves any dev view
    }
    if (u) {
      paintAccount(u);
    } else {
      accountBtn.title = 'Sign in';
      ['menu-goto-training', 'menu-goto-analytics'].forEach(function (id) {
        var el = $(id); if (el) el.style.display = '';
      });
      var label = logoutBtn.querySelector('span');
      if (label) label.textContent = 'Sign out';
    }
  }

  // ───────── prediction history window ─────────
  function csvCell(v) {
    var s = v == null ? '' : String(v);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  function pct(c) { return (c <= 1 ? c * 100 : c); }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }

  function closeHistory() {
    var m = $('history-modal');
    if (m) m.remove();
    document.body.style.overflow = '';
  }

  function openHistory() {
    if (!currentUser) { window.openAuthModal(); return; }
    closeHistory();
    var wrap = document.createElement('div');
    wrap.className = 'modal-backdrop';
    wrap.id = 'history-modal';
    wrap.innerHTML =
      '<div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="history-title" style="max-width:860px;width:calc(100% - 32px);">' +
        '<div class="modal-head"><div><span class="modal-eyebrow">' + esc(currentUser.name) + '</span>' +
        '<h3 class="modal-title" id="history-title">Prediction history</h3></div>' +
        '<button class="icon-btn" id="history-close" type="button" aria-label="Close history">&times;</button></div>' +
        '<div id="history-body" style="max-height:60vh;overflow:auto;"><p class="card-desc-compact">Loading…</p></div>' +
        '<div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;">' +
          '<button class="ghost-btn" id="history-csv" type="button" disabled>Download CSV</button>' +
          '<button class="ghost-btn" id="history-clear" type="button" disabled>Clear history</button>' +
        '</div></div>';
    document.body.appendChild(wrap);
    document.body.style.overflow = 'hidden';
    wrap.addEventListener('click', function (e) { if (e.target === wrap) closeHistory(); });
    $('history-close').addEventListener('click', closeHistory);

    var body = $('history-body');
    var csvBtn = $('history-csv');
    var clearBtn = $('history-clear');
    var rows = [];

    api('/api/history?limit=200').then(function (data) {
      rows = data.history || [];
      if (!rows.length) {
        body.innerHTML = '<p class="card-desc-compact">No predictions saved yet. Classify a sound while signed in and it will show up here.</p>';
        return;
      }
      csvBtn.disabled = false; clearBtn.disabled = false;
      body.innerHTML = '<table class="spec-table"><thead><tr><th>When</th><th>File</th><th>Model</th><th>Top prediction</th></tr></thead><tbody>' +
        rows.map(function (r) {
          return '<tr><td>' + esc(new Date(r.created_at).toLocaleString()) + '</td><td>' + esc(r.file_name) + '</td><td>' + esc(r.model) +
            '</td><td>' + esc(String(r.top1_label).replace(/_/g, ' ')) + ' <span style="opacity:.7">' + pct(r.top1_conf).toFixed(1) + '%</span></td></tr>';
        }).join('') + '</tbody></table>';
    }).catch(function (err) {
      body.innerHTML = '<p class="card-desc-compact">' + esc(err.status ? err.message : 'Cannot reach the server.') + '</p>';
    });

    csvBtn.addEventListener('click', function () {
      var head = ['time', 'file_name', 'model'];
      for (var k = 1; k <= 5; k++) head.push('top' + k + '_label', 'top' + k + '_confidence_pct');
      var lines = [head.map(csvCell).join(',')];
      rows.forEach(function (r) {
        var line = [r.created_at, r.file_name, r.model];
        for (var k = 0; k < 5; k++) { var p = r.top5[k]; line.push(p ? p.label : '', p ? pct(p.confidence).toFixed(2) : ''); }
        lines.push(line.map(csvCell).join(','));
      });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob(['\ufeff' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8;' }));
      a.download = 'soundsense-history.csv';
      document.body.appendChild(a); a.click(); a.remove();
    });

    clearBtn.addEventListener('click', function () {
      if (!confirm('Delete all saved predictions for this account?')) return;
      api('/api/history', { method: 'DELETE' }).then(function () {
        rows = []; csvBtn.disabled = true; clearBtn.disabled = true;
        body.innerHTML = '<p class="card-desc-compact">History cleared.</p>';
      }).catch(function (err) { toast('Could not clear history', err.message, 'error'); });
    });
  }

  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeHistory(); });

  // ───────── boot: confirm any saved token with the server ─────────
  setMode('login');
  if (getToken()) {
    api('/api/me').then(applyUser).catch(function (err) {
      if (err.status === 401) setToken(null); // expired or invalid
      applyUser(null);
    });
  }
})();