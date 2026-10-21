// Mitate merchant dashboard. Same origin as the API; the private key lives in sessionStorage only.
(function () {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const main = $("#main");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const KEY = "mitate.sk";
  const PLACEMENTS = {
    similar: "Similar items",
    complements: "Goes well with",
    next: "Shoppers went on to view",
    returning: "For returning shoppers",
  };
  const SOURCES = {
    recpop: "What is popular lately, time-decayed",
    itemknn: "Items the same shoppers engage with",
    rp3beta: "Random walks between shoppers and items",
    ease: "A linear item-to-item model",
    ials: "Matrix factorisation of shopper activity",
    pctm: "What shoppers view next within a visit",
    seqrules: "Sequential rules: what tends to follow what",
    vstan: "Past visits that look like this one",
    sasrec_ce: "A sequence model over each visit",
  };

  // ---------------------------------------------------------------- helpers

  function store(kind) {
    try {
      return kind === "local" ? window.localStorage : window.sessionStorage;
    } catch {
      return null; // storage blocked: the dashboard still works for this page view
    }
  }

  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v === null || v === undefined) continue;
      if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (k === "value") n.value = v;
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat(Infinity)) if (c !== null && c !== undefined && c !== false) n.append(c);
    return n;
  }

  const SVGNS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs, ...kids) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    for (const c of kids.flat()) if (c) n.append(c);
    return n;
  }

  const fmt = new Intl.NumberFormat("en-IN");
  const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
  const inr2 = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 });
  const day = new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", timeZone: "UTC" });
  const stamp = new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short" });
  const isNum = (x) => typeof x === "number" && Number.isFinite(x);
  const pct = (x, d = 1) => (isNum(x) ? `${(100 * x).toFixed(d)}%` : "unbounded");
  const signed = (s) => (/^[-−]/.test(s) ? s.replace("-", "−") : /^[\d₹]/.test(s) ? `+${s}` : s);
  const leaf = (path) => path.split(" > ").pop();
  const parent = (path) => path.split(" > ").slice(0, -1).join(" > ");

  function ago(ts) {
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 90) return "just now";
    if (s < 5400) return `${Math.round(s / 60)} min ago`;
    if (s < 129600) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} days ago`;
  }

  function toast(...content) {
    const t = el("div", { class: "toast" }, ...content);
    $(".toast-region").append(t);
    let timer;
    const arm = () => { timer = setTimeout(() => t.remove(), 4000); };
    t.addEventListener("pointerenter", () => clearTimeout(timer));
    t.addEventListener("pointerleave", arm);
    arm();
  }

  function tiltFor(text) {
    let h = 0;
    for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) | 0;
    return `${(-1.5 + ((Math.abs(h) % 41) / 10 - 2)).toFixed(1)}deg`;
  }

  // ---------------------------------------------------------------- API and session

  let key = store()?.getItem(KEY) || null;

  async function api(path, opts = {}) {
    const headers = { "X-Mitate-Key": key, ...(opts.headers || {}) };
    let body = opts.body;
    if (body !== undefined && !(body instanceof ArrayBuffer) && typeof body !== "string") {
      body = JSON.stringify(body);
      headers["Content-Type"] = "application/json";
    }
    const r = await fetch(path, { method: opts.method || "GET", headers, body });
    if (r.status === 401 || r.status === 403) {
      signOut();
      throw new Error("This key was not accepted.");
    }
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const d = data.detail;
      throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((e) => e.msg).join("; ") : `Request failed (${r.status})`);
    }
    return data;
  }

  function signOut() {
    key = null;
    store()?.removeItem(KEY);
    $("#shell").hidden = true;
    $("#signin").hidden = false;
    $("#key").focus();
  }
  $("#sign-out").addEventListener("click", signOut);

  $("#signin-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#key");
    key = input.value.trim();
    const err = $("#signin-error");
    if (!key.startsWith("sk_")) {
      err.textContent = "That looks like a public key. Paste the private key; it starts with sk_.";
      return;
    }
    try {
      await api("/v1/status");
      store()?.setItem(KEY, key);
      input.value = "";
      err.textContent = "";
      start();
    } catch {
      err.textContent = "That key was not accepted. Check you copied all of it.";
    }
  });

  // ---------------------------------------------------------------- theme

  const themeSelect = $("#theme");
  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t;
    else delete document.documentElement.dataset.theme;
  }
  themeSelect.value = store("local")?.getItem("mitate.theme") || "";
  applyTheme(themeSelect.value);
  themeSelect.addEventListener("change", () => {
    const s = store("local");
    if (s) themeSelect.value ? s.setItem("mitate.theme", themeSelect.value) : s.removeItem("mitate.theme");
    const go = () => applyTheme(themeSelect.value);
    if (document.startViewTransition && !reduceMotion.matches) document.startViewTransition(go);
    else go();
  });

  // ---------------------------------------------------------------- shared pieces

  function pageHead(eyebrow, title, lead) {
    return el("header", { class: "page-head" },
      el("p", { class: "eyebrow" }, eyebrow),
      el("h1", {}, title),
      lead ? el("p", { class: "lead" }, lead) : null);
  }

  function section(title, hint, ...body) {
    return el("section", { class: "section" }, el("h2", {}, title), hint ? el("p", { class: "hint" }, hint) : null, ...body);
  }

  // A product picker that searches by name and shows names, never raw ids.
  function productPicker({ id, label, help, chosen, titles, onChange }) {
    const listId = `${id}-results`;
    const input = el("input", {
      id, type: "search", role: "combobox", autocomplete: "off",
      "aria-autocomplete": "list", "aria-expanded": "false", "aria-controls": listId,
      placeholder: "Search products by name",
    });
    const results = el("div", { class: "results", id: listId, role: "listbox", hidden: true });
    const picked = el("ul", { class: "picked", role: "list" });
    let options = [];
    let active = -1;
    let timer = 0;

    function drawPicked() {
      picked.replaceChildren(...chosen.map((pid) => {
        const name = titles.get(pid) || pid;
        return el("li", {}, el("span", { title: name }, name),
          el("button", {
            type: "button", "aria-label": `Remove ${name}`,
            onclick: () => { chosen.splice(chosen.indexOf(pid), 1); drawPicked(); onChange(); },
          }, "×"));
      }));
    }

    function close() {
      results.hidden = true;
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
      active = -1;
    }

    function choose(item) {
      titles.set(item.id, item.title);
      if (!chosen.includes(item.id)) chosen.push(item.id);
      input.value = "";
      close();
      drawPicked();
      onChange();
    }

    function highlight(i) {
      active = i;
      [...results.children].forEach((o, j) => o.setAttribute("aria-selected", String(j === i)));
      if (i >= 0) {
        input.setAttribute("aria-activedescendant", `${listId}-${i}`);
        results.children[i].scrollIntoView({ block: "nearest" });
      }
    }

    async function search() {
      const q = input.value.trim();
      if (q.length < 2) return close();
      options = (await api(`/v1/catalog/items?q=${encodeURIComponent(q)}&limit=12`)).filter((o) => !chosen.includes(o.id));
      results.replaceChildren(...(options.length
        ? options.map((o, i) => el("div", {
          role: "option", id: `${listId}-${i}`, "aria-selected": "false",
          onpointerdown: (e) => { e.preventDefault(); choose(o); },
        }, o.image ? el("img", { src: o.image, alt: "", width: 32, height: 40 }) : el("span"),
        el("span", {}, o.title), el("small", {}, o.category ? leaf(o.category) : "")))
        : [el("p", { class: "none" }, `No products match “${q}”.`)]));
      results.hidden = false;
      input.setAttribute("aria-expanded", "true");
      active = -1;
    }

    input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(search, 160); });
    input.addEventListener("blur", () => setTimeout(close, 100));
    input.addEventListener("keydown", (e) => {
      if (results.hidden || !options.length) return;
      if (e.key === "ArrowDown") { e.preventDefault(); highlight(Math.min(active + 1, options.length - 1)); }
      else if (e.key === "ArrowUp") { e.preventDefault(); highlight(Math.max(active - 1, 0)); }
      else if (e.key === "Enter" && active >= 0) { e.preventDefault(); choose(options[active]); }
      else if (e.key === "Escape") { close(); }
    });
    drawPicked();
    return el("div", { class: "field" },
      el("label", { for: id }, label),
      help ? el("span", { class: "help" }, help) : null,
      el("div", { class: "picker" }, input, results),
      picked);
  }

  // ---------------------------------------------------------------- home

  function stations(st) {
    const pr = st.progress;
    const maxFrom = Math.max(pr.behavioural_from_events, ...pr.sources.map((s) => s.from_events));
    const allOn = pr.sources.length && pr.sources.every((s) => s.active);
    const list = [{ name: "Catalog only", note: "Product text, categories and the pairings you approve" }];
    // when every source starts with behaviour itself, the last two stations are one
    const staged = maxFrom > pr.behavioural_from_events;
    list.push({ name: "Learning from shoppers", note: `From ${fmt.format(pr.behavioural_from_events)} events${staged ? "" : ", every source"}` });
    if (staged) list.push({ name: "Every source on", note: `From ${fmt.format(maxFrom)} events` });
    const here = pr.stage !== "behavioural" ? 0 : staged && allOn ? 2 : 1;
    return { here, maxFrom, last: list.length - 1, list };
  }

  function stageTrack(st) {
    const s = stations(st);
    const before = Number(store()?.getItem("mitate.stage") ?? s.here);
    store()?.setItem("mitate.stage", String(s.here));
    const moved = before !== s.here && !reduceMotion.matches;
    return el("div", { class: "track", style: `--stations: ${s.list.length}` },
      el("ol", { role: "list", "aria-label": "How far your store has come" }, s.list.map((x, i) =>
        el("li", { class: i < s.here ? "passed" : i === s.here ? "here" : null, "aria-current": i === s.here ? "step" : null },
          i === s.here
            ? el("span", { class: `seal mark${moved ? " move" : ""}`, "aria-hidden": "true" })
            : null,
          el("span", { class: "dot", "aria-hidden": "true" }),
          el("b", {}, x.name), el("small", {}, x.note)))));
  }

  const allSourcesOn = (st) => st.progress.sources.every((x) => x.active);
  const RANKERS = { none: "catalog scores", rrf: "rank fusion", full: "the learned ranker" };
  const rankerName = (r) =>
    RANKERS[r] || (r.startsWith("no_") ? `the learned ranker without ${r.slice(3)}` : r);

  function nextUp(st) {
    const pr = st.progress;
    const s = stations(st);
    if (s.here === s.last) {
      return el("p", { class: "muted" }, allSourcesOn(st)
        ? "Every source is on. Each retrain now adds what your shoppers did since the last one."
        : "Some sources are waiting for the next retrain; the sources table below shows which.");
    }
    const target = s.here === 0 ? pr.behavioural_from_events : s.maxFrom;
    const left = Math.max(0, target - st.events);
    const next = s.list[s.here + 1].name.toLowerCase();
    if (!left) {
      return el("div", { class: "next-up" },
        el("p", {}, `Your store has passed ${fmt.format(target)} events. “${s.list[s.here + 1].name}” switches on at the next retrain.`));
    }
    return el("div", { class: "next-up" },
      el("span", { class: "numeral" }, fmt.format(left)),
      el("p", {}, `more shopper events until ${next}. Views, carts and orders all count.`));
  }

  function multiple(label, rows, keyName, money) {
    const values = rows.map((r) => r[keyName]);
    const total = values.reduce((a, b) => a + b, 0);
    const max = Math.max(1, ...values);
    const w = 280;
    const h = 76;
    const gap = 2;
    const bw = (w - gap * (values.length - 1)) / values.length;
    const peak = values.indexOf(Math.max(...values));
    const chart = svg("svg", {
      viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "none", role: "img",
      "aria-label": `${label} per day over the last ${rows.length} days: ${money ? inr.format(total) : fmt.format(total)} in total, highest on ${day.format(new Date(rows[peak].day * 1000))}.`,
    },
    values.map((v, i) => {
      const bh = v ? Math.max(1.5, (v / max) * (h - 2)) : 0;
      return svg("rect", { class: i === values.length - 1 ? "bar-v last" : "bar-v", x: (i * (bw + gap)).toFixed(2), y: (h - bh).toFixed(2), width: bw.toFixed(2), height: bh.toFixed(2), rx: 1 });
    }),
    svg("line", { class: "base", x1: 0, x2: w, y1: h + 0.5, y2: h + 0.5 }));
    return el("figure", { class: "multiple" },
      el("header", {}, el("span", { class: "eyebrow" }, label),
        el("span", { class: "total" }, money ? inr.format(total) : fmt.format(total))),
      chart,
      el("footer", { "aria-hidden": "true" },
        el("span", {}, day.format(new Date(rows[0].day * 1000))),
        el("span", {}, "today")));
  }

  function changes(models) {
    const sorted = [...models].sort((a, b) => b.version - a.version);
    return sorted.slice(0, 8).map((m, i) => {
      const prev = sorted[i + 1];
      const parts = [];
      if (!prev) parts.push(m.stage === "catalog_only" ? "First model, from your catalog alone." : "First model, already learning from shoppers.");
      else {
        if (prev.stage !== m.stage && m.stage === "behavioural") parts.push("Started learning from your shoppers.");
        const on = m.active_sources.filter((s) => !prev.active_sources.includes(s));
        const off = prev.active_sources.filter((s) => !m.active_sources.includes(s));
        if (on.length) parts.push(`Switched on: ${on.join(", ")}.`);
        if (off.length) parts.push(`Switched off: ${off.join(", ")}.`);
        if (prev.ranker !== m.ranker) parts.push(`Sources now merged by ${rankerName(m.ranker)}.`);
      }
      parts.push(`Trained on ${fmt.format(m.n_events)} events in ${m.fit_seconds.toFixed(0)} s.`);
      return el("li", {},
        el("time", { datetime: new Date(m.created_at * 1000).toISOString() }, ago(m.created_at)),
        el("span", {}, el("b", {}, `Version ${m.version}${m.active ? " (live)" : ""}. `), parts.join(" ")));
    });
  }

  function sourcesTable(st) {
    return el("div", { class: "table-wrap" }, el("table", { class: "sources" },
      el("thead", {}, el("tr", {}, el("th", {}, "Source"), el("th", {}, "What it reads"), el("th", { class: "num" }, "On from"), el("th", {}, "Now"))),
      el("tbody", {}, st.progress.sources.map((s) => el("tr", {},
        el("td", {}, s.name),
        el("td", {}, SOURCES[s.name] || ""),
        el("td", { class: "num" }, `${fmt.format(s.from_events)} events`),
        el("td", { class: s.active ? "state-on" : "state-off" }, s.active ? "On" : "Waiting"))))));
  }

  async function home() {
    const [st, models, daily] = await Promise.all([api("/v1/status"), api("/v1/models"), api("/v1/stats/daily?days=28")]);
    const pr = st.progress;
    const on = pr.sources.filter((s) => s.active).length;
    let title;
    let lead;
    if (pr.stage === "untrained") {
      title = "Mitate has not trained on this store yet.";
      lead = "Import your catalog to begin. Recommendations work from the first day, from product text and categories.";
    } else if (pr.stage === "catalog_only") {
      title = "Recommending from your catalog.";
      lead = `${fmt.format(st.active_items)} products. Shopper behaviour starts counting at ${fmt.format(pr.behavioural_from_events)} events.`;
    } else {
      title = "Learning from your shoppers.";
      lead = `The live model learned from ${fmt.format(pr.trained_on_events)} events. ${on} of ${pr.sources.length} sources are on.`;
    }
    const rt = pr.retrain;
    const retrain = el("button", { class: "btn small", type: "button" }, "Retrain now");
    retrain.addEventListener("click", async () => {
      retrain.disabled = true;
      const r = await api("/v1/retrain", { method: "POST" });
      toast(r.scheduled ? "Retraining. The new model goes live when it finishes." : "A retrain is already running.");
    });
    return [
      pageHead("Home", title, lead),
      stageTrack(st),
      nextUp(st),
      section("Last 28 days", "Counts of what shoppers did on your store, per day. Today is the highlighted bar.",
        el("div", { class: "multiples" },
          multiple("Views", daily, "views"),
          multiple("Added to cart", daily, "carts"),
          multiple("Orders", daily, "orders"),
          multiple("Revenue", daily, "revenue", true))),
      section("What changed", null,
        models.length ? el("ol", { class: "feed", role: "list" }, changes(models)) : el("p", { class: "empty-note" }, "No model yet.")),
      section("Next retrain",
        rt.age_s === null ? "No model has been trained yet."
          : `${fmt.format(rt.new_events)} new events since the last training ${ago(Date.now() / 1000 - rt.age_s)}. ` +
            `Mitate retrains after ${fmt.format(rt.new_events_needed)} new events, or after ${Math.round(rt.daily_after_s / 3600)} hours, whichever comes first.`,
        el("div", { class: "row" }, retrain, st.training ? el("span", { class: "chip" }, "Training now") : null)),
      el("details", { class: "section" },
        el("summary", {}, "Sources and thresholds"),
        el("p", { class: "hint" }, "Each source switches on once the store has enough events for it to help. The thresholds come straight from the engine's configuration."),
        sourcesTable(st)),
    ];
  }

  // ---------------------------------------------------------------- rules

  async function rules() {
    const [saved, cats] = await Promise.all([api("/v1/rules"), api("/v1/catalog/categories")]);
    const state = structuredClone(saved);
    const baseline = () => JSON.stringify(saved);
    const referenced = [...state.blocked_items, ...Object.values(state.pins).flat()];
    const known = referenced.length ? await api(`/v1/catalog/items?ids=${referenced.map(encodeURIComponent).join(",")}`) : [];
    const titles = new Map(known.map((i) => [i.id, i.title]));
    const itemCategory = new Map(known.map((i) => [i.id, i.category]));
    const counts = new Map(cats.map((c) => [c.path, c.products]));
    for (const p of ["similar", "complements", "next", "returning"]) state.pins[p] = state.pins[p] || [];

    const preview = el("p", { class: "preview", role: "status" });
    const savebar = el("div", { class: "savebar", hidden: true, role: "region", "aria-label": "Unsaved changes" });

    function covered(path) {
      return state.blocked_categories.some((c) => path === c || path.startsWith(c + " > "));
    }
    function refresh() {
      // the outermost blocked categories already include anything beneath them
      const roots = state.blocked_categories.filter((c) => !state.blocked_categories.some((o) => o !== c && c.startsWith(o + " > ")));
      const inCats = roots.reduce((a, c) => a + (counts.get(c) || 0), 0);
      const extra = state.blocked_items.filter((id) => !covered(itemCategory.get(id) || "")).length;
      const n = inCats + extra;
      preview.replaceChildren(n
        ? el("span", {}, "Shoppers will no longer see ", el("b", {}, fmt.format(n)), ` ${n === 1 ? "product" : "products"} in any rail.`)
        : "Every product can appear in rails.");
      const dirty = JSON.stringify(clean()) !== baseline();
      savebar.hidden = !dirty;
    }
    function clean() {
      const pins = {};
      for (const [p, ids] of Object.entries(state.pins)) if (ids.length) pins[p] = ids;
      return { ...state, pins };
    }
    const onChange = refresh;

    // category picker: a select of every category path, plus the chosen list
    const catSelect = el("select", { id: "block-cat" },
      el("option", { value: "" }, "Choose a category to block"),
      cats.map((c) => el("option", { value: c.path }, `${c.path} (${fmt.format(c.products)})`)));
    const catList = el("ul", { class: "picked", role: "list" });
    function drawCats() {
      catList.replaceChildren(...state.blocked_categories.map((c) => el("li", {},
        el("span", {}, c),
        el("button", { type: "button", "aria-label": `Unblock ${c}`, onclick: () => { state.blocked_categories.splice(state.blocked_categories.indexOf(c), 1); drawCats(); refresh(); } }, "×"))));
    }
    catSelect.addEventListener("change", () => {
      const v = catSelect.value;
      if (v && !state.blocked_categories.includes(v)) state.blocked_categories.push(v);
      catSelect.value = "";
      drawCats();
      refresh();
    });
    drawCats();

    function number(field, label, help, { min, step = "any", int = false, placeholder = "Off" }) {
      const input = el("input", { id: `r-${field}`, type: "number", min, step, inputmode: int ? "numeric" : "decimal", value: state[field] ?? "", placeholder });
      input.addEventListener("input", () => {
        state[field] = input.value === "" ? null : int ? parseInt(input.value, 10) : Number(input.value);
        refresh();
      });
      return el("div", { class: "field" }, el("label", { for: `r-${field}` }, label), el("span", { class: "help" }, help), input);
    }
    function check(field, label, help) {
      const input = el("input", { type: "checkbox", id: `r-${field}` });
      input.checked = !!state[field];
      input.addEventListener("change", () => { state[field] = input.checked; refresh(); });
      return el("label", { class: "check", for: `r-${field}` }, input, el("span", {}, label), el("span", { class: "help" }, help));
    }
    function group(title, about, ...fields) {
      return el("section", { class: "group" }, el("header", {}, el("h2", {}, title), el("p", {}, about)), el("div", { class: "fields" }, ...fields));
    }

    const discard = el("button", { class: "btn small", type: "button" }, "Discard");
    const save = el("button", { class: "btn small primary", type: "button" }, "Save changes");
    savebar.append(el("p", {}, "You have unsaved changes."), discard, save);
    discard.addEventListener("click", () => route());
    save.addEventListener("click", async () => {
      if (state.retention_days === null || !(state.retention_days >= 1)) {
        toast("Keep shopper data for at least 1 day.");
        $("#r-retention_days")?.focus();
        return;
      }
      save.disabled = true;
      try {
        const out = await api("/v1/rules", { method: "PUT", body: clean() });
        Object.assign(saved, out);
        refresh();
        toast(el("span", { class: "seal mark", "aria-hidden": "true" }), "Saved. Your store uses these rules now.");
      } catch (e) {
        toast(e.message);
      } finally {
        save.disabled = false;
      }
    });

    refresh();
    return [
      pageHead("Rules", "What your shoppers can be shown.", "Rules apply on the next request, with no retraining. Recommendations that break a rule are never shown."),
      el("div", { class: "rules" },
        group("Keep off every rail", "Products and categories shoppers should never be recommended.",
          productPicker({ id: "r-blocked", label: "Products", chosen: state.blocked_items, titles, onChange }),
          el("div", { class: "field" }, el("label", { for: "block-cat" }, "Categories"),
            el("span", { class: "help" }, "Blocking a category also blocks everything beneath it."), catSelect, catList),
          preview),
        group("Featured products", "Pinned products show first and carry the label “Featured by the store”.",
          Object.entries(PLACEMENTS).map(([p, label]) =>
            productPicker({ id: `r-pin-${p}`, label, chosen: state.pins[p], titles, onChange }))),
        group("Price guardrails", "Keep suggestions in a sensible price range for the product on screen.",
          number("complement_max_price_ratio", "“Goes well with” costs at most", "A multiple of the product's price. 3 means a ₹500 pan suggests add-ons up to ₹1,500.", { min: 0.1, step: 0.1 }),
          number("similar_price_ratio", "“Similar items” within", "A multiple either way. 2 keeps a ₹1,000 product's matches between ₹500 and ₹2,000.", { min: 1, step: 0.1 }),
          number("max_per_category", "At most this many from one category per rail", "Leave empty for no limit.", { min: 1, step: 1, int: true })),
        group("Stock", "Out-of-stock products are never recommended.",
          check("hide_untracked_stock", "Also hide products with no stock count", "Use this if your catalog leaves stock empty for discontinued items.")),
        group("Younger shoppers", "When a shopper's age is unknown.",
          check("minor_safe_for_unknown_age", "Treat unknown ages as minors", "No behavioural personalisation for them. Recommended for stores that sell mostly to families or children.")),
        group("Shopper data", "How long Mitate keeps behavioural events.",
          number("retention_days", "Keep events for (days)", "Older events are deleted and stop influencing recommendations at the next retrain.", { min: 1, step: 1, int: true, placeholder: "180" })),
        group("Storefront x-ray", "Let storefront pages ask what the engine did: sources, rules applied and the counted evidence. No personal data is included.",
          check("allow_xray", "Allow x-ray requests", "Useful while you evaluate Mitate. Turn it off when you are done."))),
      savebar,
    ];
  }

  // ---------------------------------------------------------------- pairings

  async function pairings() {
    const [edges, cats] = await Promise.all([api("/v1/graph"), api("/v1/catalog/categories")]);
    const proposed = edges.filter((e) => e.status === "proposed");
    const approved = edges.filter((e) => e.status === "approved").sort((a, b) => a.src.localeCompare(b.src));
    const rejected = edges.filter((e) => e.status === "rejected");
    setPending(proposed.length);
    let at = 0;
    let freshSeal = null;
    const reviewBox = el("div");

    async function decide(e, status, viaKeyboard) {
      await api("/v1/graph", { method: "PUT", body: { edges: [{ ...e, status }] } });
      e.status = status;
      proposed.splice(proposed.indexOf(e), 1);
      if (status === "approved") {
        approved.push(e);
        approved.sort((a, b) => a.src.localeCompare(b.src));
        freshSeal = viaKeyboard || reduceMotion.matches ? null : `${e.src}|${e.dst}`;
      } else if (status === "rejected") {
        rejected.push(e);
      }
      setPending(proposed.length);
      if (at >= proposed.length) at = 0;
      drawReview();
      drawLedger();
      reviewBox.querySelector("button")?.focus();
    }

    async function unpair(e) {
      await api("/v1/graph", { method: "PUT", body: { edges: [{ ...e, status: "rejected" }] } });
      approved.splice(approved.indexOf(e), 1);
      rejected.push({ ...e, status: "rejected" });
      drawLedger();
      toast(`${leaf(e.src)} + ${leaf(e.dst)} removed.`);
    }

    function pairSide(path) {
      return el("div", {}, el("span", { class: "dept" }, parent(path) || "Top level"), el("span", { class: "leaf" }, leaf(path)));
    }

    function drawReview() {
      if (!proposed.length) {
        const ask = el("button", { class: "btn", type: "button" }, "Ask the local model for suggestions");
        const note = el("p", { class: "hint", role: "status" });
        ask.addEventListener("click", async () => {
          ask.disabled = true;
          note.textContent = "Asking the model on this server. This can take a minute.";
          try {
            const r = await api("/v1/graph/propose", { method: "POST" });
            note.textContent = r.proposed ? `${r.proposed} new suggestions to review.` : "No new suggestions.";
            if (r.proposed) route();
          } catch (err) {
            note.textContent = `${err.message}. You can still add pairings by hand below.`;
          } finally {
            ask.disabled = false;
          }
        });
        reviewBox.replaceChildren(el("div", { class: "empty-note" },
          el("p", {}, "No pairings waiting for review."),
          el("p", { class: "hint" }, "Suggestions come from a language model that runs on this server and reads only your category names. None goes live until you approve it."),
          ask, note));
        return;
      }
      const e = proposed[at];
      const btn = (label, k, status, cls) => el("button", {
        class: `btn ${cls || ""}`, type: "button", "data-key": k,
        onclick: () => (status ? decide(e, status, false) : skip()),
      }, label, el("kbd", { "aria-hidden": "true" }, k.toUpperCase()));
      reviewBox.replaceChildren(el("article", { class: "review", "aria-label": `Proposed pairing: ${leaf(e.src)} with ${leaf(e.dst)}` },
        el("p", { class: "progress" }, `${at + 1} of ${proposed.length} to review`),
        el("div", { class: "pair" }, pairSide(e.src), el("span", { class: "with", "aria-hidden": "true" }, "+"), pairSide(e.dst)),
        el("p", { class: "origin" }, e.origin === "llm"
          ? "Suggested by the language model on this server, from category names alone."
          : `Proposed by ${e.origin}.`),
        el("p", {}, `If you approve, shoppers looking at ${leaf(e.src).toLowerCase()} can be shown ${leaf(e.dst).toLowerCase()} under “Goes well with”.`),
        el("div", { class: "decide" }, btn("Approve", "a", "approved", "primary"), btn("Reject", "r", "rejected"), btn("Skip", "s", null, "ghost"))));
    }
    function skip() {
      at = (at + 1) % proposed.length;
      drawReview();
    }

    const ledgerBox = el("div");
    function drawLedger() {
      ledgerBox.replaceChildren(approved.length
        ? el("div", { class: "table-wrap" }, el("table", { class: "edges" },
          el("thead", {}, el("tr", {}, el("th", {}, "Shoppers looking at"), el("th", {}, "can be shown"), el("th", {}, "Added by"), el("th", {}, el("span", { class: "visually-hidden" }, "Actions")))),
          el("tbody", {}, approved.map((e) => {
            const id = `${e.src}|${e.dst}`;
            return el("tr", {},
              el("td", {}, el("span", { class: `seal mark${freshSeal === id ? " stamp" : ""}`, style: `--seal-tilt: ${tiltFor(id)}`, "aria-hidden": "true" }), e.src),
              el("td", {}, e.dst),
              el("td", {}, e.origin === "llm" ? "model, approved by you" : e.origin),
              el("td", {}, el("button", { class: "link", type: "button", onclick: () => unpair(e) }, "Remove")));
          }))))
        : el("p", { class: "empty-note" }, "No approved pairings yet. Until you add some, “Goes well with” uses what shoppers actually bought together, then related items from other categories."));
      freshSeal = null;
    }

    const src = el("select", { id: "p-src", required: true }, el("option", { value: "" }, "Choose a category"), cats.map((c) => el("option", { value: c.path }, c.path)));
    const dst = el("select", { id: "p-dst", required: true }, el("option", { value: "" }, "Choose a category"), cats.map((c) => el("option", { value: c.path }, c.path)));
    const add = el("form", { class: "form-grid" },
      el("div", { class: "field" }, el("label", { for: "p-src" }, "Shoppers looking at"), src),
      el("div", { class: "field" }, el("label", { for: "p-dst" }, "can be shown"), dst),
      el("button", { class: "btn", type: "submit" }, "Add pairing"));
    add.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (!src.value || !dst.value || src.value === dst.value) { toast("Choose two different categories."); return; }
      const e = { src: src.value, dst: dst.value, status: "approved", origin: "merchant" };
      await api("/v1/graph", { method: "PUT", body: { edges: [e] } });
      approved.push(e);
      approved.sort((a, b) => a.src.localeCompare(b.src));
      freshSeal = reduceMotion.matches ? null : `${e.src}|${e.dst}`;
      add.reset();
      drawLedger();
    });

    keyHandler = (ev) => {
      if (!proposed.length || ev.metaKey || ev.ctrlKey || ev.altKey) return;
      if (ev.target.closest("input, select, textarea")) return;
      const e = proposed[at];
      const k = ev.key.toLowerCase();
      if (k === "a") decide(e, "approved", true);
      else if (k === "r") decide(e, "rejected", true);
      else if (k === "s") skip();
    };

    drawReview();
    drawLedger();
    return [
      pageHead("Pairings", "Which categories go together.", "“Goes well with” suggests products from categories you have paired, alongside what your shoppers bought together."),
      section("Waiting for you", null, reviewBox),
      section("Approved", null, ledgerBox),
      rejected.length ? el("details", { class: "section" }, el("summary", {}, `Rejected (${rejected.length})`),
        el("ul", { class: "report-list" }, rejected.map((e) => el("li", {}, `${e.src} + ${e.dst}`)))) : null,
      section("Add a pairing", "Pairings apply in one direction. Add the reverse too if it makes sense for your store.", add),
    ];
  }

  // ---------------------------------------------------------------- tests

  function interval(lo, est, hi, range, fmtv) {
    const span = range || Math.max(Math.abs(lo), Math.abs(hi), 1e-9) * 1.25;
    const x = (v) => `${(50 + (50 * Math.max(-span, Math.min(span, v))) / span).toFixed(2)}%`;
    return el("div", {},
      el("div", { class: "interval", role: "img", "aria-label": `Estimate ${fmtv(est)}; 95% interval ${fmtv(lo)} to ${fmtv(hi)}.` },
        el("div", { class: "axis" }),
        el("div", { class: "zero", style: `inset-inline-start: ${x(0)}` }),
        el("div", { class: "ci", style: `inset-inline-start: ${x(lo)}; inline-size: calc(${x(hi)} - ${x(lo)})` }),
        el("div", { class: "est", style: `inset-inline-start: ${x(est)}` })),
      el("div", { class: "interval-labels", "aria-hidden": "true" }, el("span", {}, fmtv(-span)), el("span", {}, fmtv(span))));
  }

  function needFor(skew) {
    return Math.ceil(355 * skew * skew);
  }

  function holdoutView(rep) {
    if (!rep.revenue_per_visitor) {
      return [el("p", { class: "muted" }, rep.status || "Waiting for visitors in both groups.")];
    }
    const r = rep.revenue_per_visitor;
    const c = rep.conversion;
    const money = (v) => signed(inr2.format(v));
    let verdict;
    if (!r.clt_ok) {
      verdict = el("p", { class: "verdict" }, el("b", {}, "Too early to read. "),
        `A few large orders dominate revenue here, so the interval is not reliable until each group has about ${fmt.format(needFor(r.skewness))} visitors.`);
    } else if (r.ci_low > 0) {
      verdict = el("p", { class: "verdict" }, el("b", { class: "ok-text" }, "Mitate is ahead. "),
        `About ${money(r.abs_effect)} per visitor; the true effect is likely between ${money(r.ci_low)} and ${money(r.ci_high)}.`);
    } else if (r.ci_high < 0) {
      verdict = el("p", { class: "verdict" }, el("b", { class: "err" }, "Mitate is behind. "),
        `About ${money(r.abs_effect)} per visitor; likely between ${money(r.ci_low)} and ${money(r.ci_high)}.`);
    } else {
      verdict = el("p", { class: "verdict" }, el("b", {}, "No clear difference yet. "),
        `The effect is somewhere between ${money(r.ci_low)} and ${money(r.ci_high)} per visitor.`);
    }
    const row = (k, v) => el("tr", {}, el("th", { scope: "row" }, k), el("td", { class: "num" }, v));
    return [
      el("p", { class: "muted" }, `${fmt.format(rep.units.mitate)} visitors with Mitate, ${fmt.format(rep.units.holdout)} in the holdout. ${rep.population}`),
      verdict,
      interval(r.ci_low, r.abs_effect, r.ci_high, null, (v) => inr2.format(v)),
      el("details", {}, el("summary", {}, "The numbers"),
        el("div", { class: "table-wrap" }, el("table", {}, el("tbody", {},
          row("Revenue per visitor, Mitate", inr2.format(r.treatment_mean)),
          row("Revenue per visitor, holdout", inr2.format(r.control_mean)),
          row("Difference (95% interval)", `${money(r.abs_effect)} (${money(r.ci_low)} to ${money(r.ci_high)})`),
          row("Relative (95% interval, Fieller)", `${signed(pct(r.rel_effect))} (${pct(r.rel_ci_low)} to ${pct(r.rel_ci_high)})`),
          row("p-value (Welch)", r.p_value.toFixed(3)),
          row("Variance removed by pre-period adjustment", r.cuped ? pct(r.variance_reduction, 0) : "not used"),
          row("Skewness / visitors needed per group", `${r.skewness.toFixed(1)} / ${fmt.format(needFor(r.skewness))}`),
          row("Conversion, Mitate vs holdout", `${pct(c.treatment_mean, 2)} vs ${pct(c.control_mean, 2)}`),
          row("Conversion difference (95% interval)", `${signed((100 * c.abs_effect).toFixed(2))} pts (${(100 * c.ci_low).toFixed(2)} to ${(100 * c.ci_high).toFixed(2)})`)))),
        el("p", { class: "hint" }, rep.note)),
    ];
  }

  function interleaveView(rep) {
    const il = rep.interleaving;
    if (!il) return [el("p", { class: "muted" }, "No clicks on the shared lists yet.")];
    const share = (v) => signed(v.toFixed(3));
    let verdict;
    if (!il.clt_ok) verdict = el("p", { class: "verdict" }, el("b", {}, "Too early to read. "), `${fmt.format(il.n_visitors)} shoppers have clicked so far; the interval needs more.`);
    else if (il.ci_low > 0) verdict = el("p", { class: "verdict" }, el("b", { class: "ok-text" }, "Shoppers preferred Mitate's picks. "), `Clicks went to Mitate's items more often than to ${rep.comparison.split(" vs ")[1]}.`);
    else if (il.ci_high < 0) verdict = el("p", { class: "verdict" }, el("b", { class: "err" }, "Shoppers preferred the alternative. "), "Clicks went to the comparison's items more often.");
    else verdict = el("p", { class: "verdict" }, el("b", {}, "No clear preference yet. "), "Clicks are split about evenly so far.");
    return [
      el("p", { class: "muted" }, `${fmt.format(il.n_engaged)} lists with a click, from ${fmt.format(il.n_visitors)} shoppers. Mitate won ${fmt.format(il.wins_a)}, the alternative ${fmt.format(il.wins_b)}, ties ${fmt.format(il.ties)}.`),
      verdict,
      interval(il.ci_low, il.delta, il.ci_high, 0.5, share),
      el("p", { class: "hint" }, rep.note),
    ];
  }

  async function tests() {
    const list = await api("/v1/experiments");
    const items = list.map((x) => {
      const body = el("div", {}, el("p", { class: "muted" }, "Reading results…"));
      api(`/v1/experiments/${x.id}/report`)
        .then((rep) => body.replaceChildren(...(x.kind === "holdout" ? holdoutView(rep) : interleaveView(rep))))
        .catch((e) => body.replaceChildren(el("p", { class: "err" }, e.message)));
      const stop = x.status === "running"
        ? el("button", { class: "btn small", type: "button", onclick: async () => { await api(`/v1/experiments/${x.id}/stop`, { method: "POST" }); route(); } }, "Stop test")
        : null;
      return el("article", { class: "experiment" },
        el("header", {},
          el("h3", {}, x.name),
          el("span", { class: x.status === "running" ? "chip ok" : "chip" }, x.status === "running" ? "Running" : "Stopped"),
          el("span", { class: "faint small" }, `${x.kind === "holdout" ? `Holdout, ${pct(x.holdout_share, 0)} of visitors` : `Preference test against ${x.comparison.replace("_", " ")}`} · ${x.placement ? PLACEMENTS[x.placement] : "every rail"} · since ${day.format(new Date(x.started_at * 1000))}`)),
        body, stop);
    });

    const kind = el("select", { id: "x-kind" }, el("option", { value: "holdout" }, "Revenue test (holdout)"), el("option", { value: "interleave" }, "Preference test (interleaving)"));
    const shareField = el("div", { class: "field" }, el("label", { for: "x-share" }, "Visitors held out"),
      el("input", { id: "x-share", type: "number", min: 1, max: 50, step: 1, value: 10, inputmode: "numeric" }));
    kind.addEventListener("change", () => { shareField.hidden = kind.value !== "holdout"; });
    const create = el("form", { class: "form-grid" },
      el("div", { class: "field" }, el("label", { for: "x-name" }, "Name"), el("input", { id: "x-name", required: true, maxlength: 80, placeholder: "October revenue test" })),
      el("div", { class: "field" }, el("label", { for: "x-kind" }, "Kind"), kind),
      el("div", { class: "field" }, el("label", { for: "x-place" }, "Where"),
        el("select", { id: "x-place" }, el("option", { value: "" }, "Every rail"), Object.entries(PLACEMENTS).map(([v, l]) => el("option", { value: v }, l)))),
      shareField,
      el("button", { class: "btn primary", type: "submit" }, "Start test"));
    create.addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        await api("/v1/experiments", {
          method: "POST",
          body: {
            name: $("#x-name").value.trim(), kind: kind.value, placement: $("#x-place").value || null,
            holdout_share: Number($("#x-share").value) / 100,
          },
        });
        toast("Test started.");
        route();
      } catch (err) {
        toast(err.message);
      }
    });

    const out = el("div", { class: "power-out", role: "status" });
    const power = el("form", { class: "form-grid" },
      el("div", { class: "field" }, el("label", { for: "pw-v" }, "Visitors a day"), el("input", { id: "pw-v", type: "number", min: 1, value: 400, inputmode: "numeric" })),
      el("div", { class: "field" }, el("label", { for: "pw-m" }, "Smallest lift worth finding (%)"), el("input", { id: "pw-m", type: "number", min: 1, max: 90, step: 1, value: 5, inputmode: "numeric" })),
      el("div", { class: "field" }, el("label", { for: "pw-h" }, "Within (days)"), el("input", { id: "pw-h", type: "number", min: 7, max: 365, value: 28, inputmode: "numeric" })),
      el("button", { class: "btn", type: "submit" }, "Check"));
    power.addEventListener("submit", async (e) => {
      e.preventDefault();
      const mde = Number($("#pw-m").value) / 100;
      const horizon = Number($("#pw-h").value);
      const r = await api("/v1/power", { method: "POST", body: { daily_visitors: Number($("#pw-v").value), rel_mde: mde, horizon_days: horizon } });
      if (r.status) {
        out.replaceChildren(el("p", {}, `Not enough order data yet to say: ${r.status}.`));
        return;
      }
      const weeks = r.days_needed / 7;
      const span = weeks < 1 ? "under a week" : weeks > 104 ? "more than two years" : `about ${Math.round(weeks)} ${Math.round(weeks) === 1 ? "week" : "weeks"}`;
      out.replaceChildren(
        el("p", { class: "verdict" }, r.conclusive_within_horizon
          ? [el("b", { class: "ok-text" }, "Ready to test. "), `Finding a ${Math.round(mde * 100)}% lift takes ${span} at your traffic.`]
          : [el("b", {}, "Too small to test revenue in that time. "), `A ${Math.round(mde * 100)}% lift needs ${span}. In ${horizon} days you could find lifts of ${pct(r.mde_at_horizon, 0)} or more. A preference test needs far fewer shoppers.`]),
        el("p", { class: "hint" }, `Based on ${fmt.format(r.basis_visitors)} visitors in the last 28 days, averaging ${inr2.format(r.mean_rpv)} each. ` +
          "It assumes the next weeks look like those. A few very large orders make revenue noisier than this estimate allows, so expect to need longer if your store gets them."));
    });

    return [
      pageHead("Tests", "Is Mitate earning its place?", "A revenue test holds some visitors out and compares. A preference test mixes two rankers in one list and counts clicks: it needs far fewer shoppers, but measures preference, not revenue."),
      items.length ? el("div", {}, items) : el("p", { class: "empty-note" }, "No tests yet."),
      section("Start a test", null, create),
      section("Can my store measure this?", "Uses the spread of your own revenue per visitor over the last 28 days.", power, out),
    ];
  }

  // ---------------------------------------------------------------- catalog

  const FORMATS = [
    ["mitate", "Mitate CSV", "Columns: id, title, category, price, stock, and optionally description, brand, image_url."],
    ["shopify", "Shopify export", "Shopify admin › Products › Export › CSV for Excel, Numbers or other programs."],
    ["woocommerce", "WooCommerce export", "WooCommerce › Products › Export, with all columns."],
  ];

  async function catalog() {
    const fileInput = el("input", { type: "file", id: "c-file", accept: ".csv,text/csv", class: "visually-hidden" });
    const chosen = el("p", { role: "status" }, "No file chosen.");
    const replace = el("input", { type: "checkbox", id: "c-replace" });
    const report = el("div", { role: "status" });
    let file = null;
    const pick = (f) => {
      file = f;
      chosen.textContent = f ? `${f.name}, ${fmt.format(Math.round(f.size / 1024))} KB` : "No file chosen.";
    };
    fileInput.addEventListener("change", () => pick(fileInput.files[0] || null));
    const drop = el("div", { class: "drop" },
      el("p", {}, "Drop a CSV file here, or"),
      el("label", { class: "btn", for: "c-file" }, "Choose a file"), fileInput, chosen);
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0] || null); });

    const form = el("form", {},
      el("fieldset", { class: "formats" }, el("legend", { class: "eyebrow" }, "File format"),
        FORMATS.map(([v, l, how], i) => el("label", {}, el("input", { type: "radio", name: "fmt", value: v, checked: i === 0 }), el("b", {}, l), el("small", {}, how)))),
      el("div", { class: "section" }, drop),
      el("label", { class: "check", for: "c-replace" }, replace, el("span", {}, "This file is my whole catalog"),
        el("span", { class: "help" }, "Products missing from it stop being recommended. Leave this off to add or update only.")),
      el("div", { class: "row", style: "margin-block-start: var(--s4)" }, el("button", { class: "btn primary", type: "submit" }, "Import")),
      report);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!file) { toast("Choose a CSV file first."); return; }
      report.replaceChildren(el("p", { class: "muted" }, "Importing…"));
      try {
        const fmtv = form.querySelector("input[name=fmt]:checked").value;
        const r = await api(`/v1/catalog/import/${fmtv}?replace=${replace.checked}`, { method: "POST", body: await file.arrayBuffer(), headers: { "Content-Type": "text/csv" } });
        report.replaceChildren(section("Import report", null,
          el("ul", { class: "report-list", role: "list" },
            el("li", {}, `${fmt.format(r.upserted)} products added or updated.`),
            el("li", {}, `${fmt.format(r.deactivated)} no longer sold.`),
            el("li", {}, `${fmt.format(r.n_active)} products are live now.`),
            r.errors.length ? el("li", { class: "warn-text" }, `${fmt.format(r.errors.length)} rows skipped:`) : null),
          r.errors.length ? el("ul", { class: "report-list mono small" }, r.errors.slice(0, 50).map((x) => el("li", {}, x))) : null));
      } catch (err) {
        report.replaceChildren(el("p", { class: "err" }, err.message));
      }
    });
    return [pageHead("Catalog", "Your products.", "Import a product file. Titles, categories and descriptions are what Mitate reads first, so recommendations work before any shopper arrives."), form];
  }

  // ---------------------------------------------------------------- privacy

  async function privacy() {
    const exportId = el("input", { id: "pv-export", autocomplete: "off", spellcheck: "false", required: true });
    const exportForm = el("form", { class: "form-grid" },
      el("div", { class: "field" }, el("label", { for: "pv-export" }, "Shopper id"), exportId),
      el("button", { class: "btn", type: "submit" }, "Download their data"));
    exportForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const id = exportId.value.trim();
      const data = await api(`/v1/shoppers/${encodeURIComponent(id)}/export`);
      const blob = new Blob([JSON.stringify(data, null, 1)], { type: "application/json" });
      const a = el("a", { href: URL.createObjectURL(blob), download: `mitate-shopper-export-${new Date().toISOString().slice(0, 10)}.json` });
      a.click();
      URL.revokeObjectURL(a.href);
      toast(`Exported ${fmt.format(data.events?.length ?? 0)} events.`);
    });

    const eraseId = el("input", { id: "pv-erase", autocomplete: "off", spellcheck: "false", required: true });
    const confirm = el("input", { id: "pv-confirm", autocomplete: "off", spellcheck: "false", required: true });
    const receipt = el("div", { role: "status" });
    const eraseBtn = el("button", { class: "btn", type: "submit", disabled: true }, "Erase permanently");
    const sync = () => { eraseBtn.disabled = !eraseId.value.trim() || confirm.value.trim() !== eraseId.value.trim(); };
    eraseId.addEventListener("input", sync);
    confirm.addEventListener("input", sync);
    const eraseForm = el("form", { class: "form-grid" },
      el("div", { class: "field" }, el("label", { for: "pv-erase" }, "Shopper id"), eraseId),
      el("div", { class: "field" }, el("label", { for: "pv-confirm" }, "Type it again to confirm"), confirm),
      eraseBtn);
    eraseForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const r = await api(`/v1/shoppers/${encodeURIComponent(eraseId.value.trim())}`, { method: "DELETE" });
      receipt.replaceChildren(el("div", { class: "section" },
        el("p", { class: "eyebrow" }, `Receipt · ${stamp.format(new Date())}`),
        el("p", {}, `${fmt.format(r.events_deleted ?? 0)} events deleted. Future events for this shopper are dropped on arrival, and the model retrains without them.`)));
      eraseForm.reset();
      sync();
    });

    return [
      pageHead("Privacy", "Shoppers' data, on request.", "Under India's DPDP Act a shopper can ask what you hold about them, and ask you to erase it. Use the same id your store sends to Mitate."),
      section("Export", "A JSON file of every event Mitate holds for this shopper.", exportForm),
      section("Erase", "Deletes their events and test exposure, and keeps them out from now on. This cannot be undone.", eraseForm, receipt),
      section("What Mitate never stores", null, el("ul", {},
        el("li", {}, "Names, emails, phone numbers or addresses."),
        el("li", {}, "Raw shopper ids: they are pseudonymised with a key only this store holds."),
        el("li", {}, "Anything from shoppers who did not consent, beyond the product on screen."))),
    ];
  }

  // ---------------------------------------------------------------- models

  async function models() {
    const [st, list] = await Promise.all([api("/v1/status"), api("/v1/models")]);
    return [
      pageHead("Models", "Every model this store has trained.", "Versions are kept so you can see what changed. Only the live one serves shoppers."),
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Version", "Stage", "Events", "Sources", "Merging", "Fit", "Trained"].map((h, i) => el("th", { class: i === 2 || i === 5 ? "num" : null }, h)))),
        el("tbody", {}, [...list].sort((a, b) => b.version - a.version).map((m) => el("tr", { class: m.active ? "live" : null },
          el("td", {}, `v${m.version}${m.active ? " · live" : ""}`),
          el("td", {}, m.stage === "catalog_only" ? "Catalog only" : "Behavioural"),
          el("td", { class: "num" }, fmt.format(m.n_events)),
          el("td", { class: "mono" }, m.active_sources.join(" · ") || "catalog"),
          el("td", {}, rankerName(m.ranker)),
          el("td", { class: "num" }, `${m.fit_seconds.toFixed(1)} s`),
          el("td", {}, stamp.format(new Date(m.created_at * 1000)))))))),
      section("Sources and thresholds", null, sourcesTable(st)),
    ];
  }

  // ---------------------------------------------------------------- shell and routing

  const ROUTES = { home, rules, pairings, tests, catalog, privacy, models };
  let keyHandler = null;
  let renderToken = 0;
  let firstRender = true;

  function setPending(n) {
    const b = $("#pending-count");
    b.hidden = !n;
    b.textContent = String(n);
    b.setAttribute("aria-label", `${n} pairings waiting`);
  }

  document.addEventListener("keydown", (e) => keyHandler?.(e));

  async function route() {
    const name = (location.hash.slice(1) || "home").split("?")[0];
    const page = ROUTES[name] ? name : "home";
    for (const a of document.querySelectorAll(".rail a")) {
      a.dataset.route === page ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current");
    }
    keyHandler = null;
    const token = ++renderToken;
    main.setAttribute("aria-busy", "true");
    try {
      const nodes = await ROUTES[page]();
      if (token !== renderToken) return; // a newer navigation won
      main.replaceChildren(...nodes.flat().filter(Boolean));
    } catch (e) {
      if (token !== renderToken) return;
      main.replaceChildren(pageHead("Something went wrong", "This page could not load.", e.message));
    } finally {
      main.removeAttribute("aria-busy");
    }
    document.title = `${page === "home" ? "Home" : page[0].toUpperCase() + page.slice(1)} · Mitate`;
    if (!firstRender) main.focus({ preventScroll: true });
    firstRender = false;
    window.scrollTo(0, 0);
  }

  async function start() {
    $("#signin").hidden = true;
    $("#shell").hidden = false;
    const st = await api("/v1/status");
    $("#store-id").textContent = st.merchant_id;
    api("/v1/graph").then((g) => setPending(g.filter((e) => e.status === "proposed").length)).catch(() => {});
    route();
  }

  window.addEventListener("hashchange", route);
  if (key) start().catch(signOut);
  else signOut();
})();
