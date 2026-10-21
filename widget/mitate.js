/*! Mitate storefront widget. No dependencies, no fonts, no third-party requests.
 *
 *   <script src="https://YOUR-MITATE/widget/mitate.js" defer
 *           data-key="pk_..." data-api="https://YOUR-MITATE"
 *           data-url-template="/products/{id}" data-currency="INR" data-heading="h2"></script>
 *   <div data-mitate="similar" data-item="SKU1"></div>
 *   <div data-mitate="complements" data-items="SKU1,SKU2" data-title="Goes well with"></div>
 *   <div data-mitate="next"></div>
 *   <div data-mitate="returning" data-k="6"></div>
 *
 *   Mitate.setConsent("granted")          // from your consent banner: granted|denied|withdrawn
 *   Mitate.setShopper("your-opaque-id")   // logged-in shoppers only; never an email or phone
 *   Mitate.track("view", "SKU1")          // product page view; also "cart", "order" (+ value)
 *   Mitate.set({ ageStatus: "minor" })    // switches to the minor-safe path; also region
 *
 * Rails render inside Shadow DOM, inherit the page's font and colour, and size themselves with
 * container queries. Restyle through custom properties on the placement element:
 * --mitate-gap, --mitate-radius, --mitate-media-ratio, --mitate-media-bg.
 *
 * Privacy: without consent the widget stores nothing in the browser and sends no identifiers;
 * recommendations then use only the product on screen. Every reason shown is a counted fact from
 * the store's own data, and counts appear only when at least 5 shoppers stand behind them.
 *
 * Each render dispatches a bubbling "mitate:render" event ({response, ms, serverMs}) on the
 * placement element, for stores that want to inspect what was served.
 */
(function () {
  "use strict";
  var script = document.currentScript;
  var d = (script && script.dataset) || {};
  var cfg = {
    key: d.key,
    api: d.api || "",
    urlTemplate: d.urlTemplate || "/products/{id}",
    currency: d.currency || "INR",
    locale: d.locale || undefined,
    heading: /^h[2-6]$/.test(d.heading || "") ? d.heading : "h2",
    consent: "unknown",
    ageStatus: "unknown",
    region: null,
    shopperId: null,
    xray: false,
    titles: {
      similar: "Similar items",
      complements: "Goes well with",
      next: "You may also like",
      returning: "For you",
    },
  };
  var SESSION_KEY = "mitate.session";
  var SVG = "http://www.w3.org/2000/svg";
  // one 16-unit stroke glyph per reason; drawn in currentColor
  var GLYPH = {
    bought_with: "M3 11h4l2-6h4M11 3l2 2-2 2",
    viewed_next: "M2 8h10M9 5l3 3-3 3",
    similar: "M3 6c2-2 4 2 6 0s4-2 4 0M3 10c2-2 4 2 6 0s4-2 4 0",
    pairs_with: "M6 10l4-4M5 7l-1 1a2.5 2.5 0 003.5 3.5l1-1M11 9l1-1A2.5 2.5 0 008.5 4.5l-1 1",
    bought_before: "M4 6a4.5 4.5 0 118 2.5M4 3v3h3",
    popular: "M3 13V9M7 13V5M11 13V7",
    new: "M8 3v10M3 8h10",
    featured: "M5 3h6l-1 4 2 2H4l2-2zM8 9v4",
  };
  var SCOPE = {
    bought_with: "Distinct shoppers in this store who bought both.",
    viewed_next: "Distinct shoppers in this store who viewed one right after the other.",
    popular: "Distinct shoppers in this store over the last 7 days.",
    similar: "From the catalog: the shared category, brand and price band.",
    pairs_with: "A pairing this store chose.",
    bought_before: "From your own orders here. Only you see this.",
    featured: "Placed here by the store.",
    new: "Added to the catalog in the last 14 days.",
  };
  var COUNTED = { bought_with: 1, viewed_next: 1, popular: 1 };
  var CSS =
    ":host{display:block;font:inherit;color:inherit;container-type:inline-size;--g:var(--mitate-gap,1rem)}" +
    ".r{margin-block:1.5rem}" +
    ".h{display:flex;align-items:end;justify-content:space-between;gap:1rem;margin-block-end:.75rem}" +
    ".t{font:inherit;font-weight:600;font-size:1.15em;line-height:1.25;margin:0}" +
    ".n{display:none;gap:.25rem}" +
    ".n button{font:inherit;color:inherit;background:none;border:1px solid color-mix(in srgb,currentColor 35%,transparent);border-radius:6px;min-inline-size:2rem;min-block-size:2rem;cursor:pointer}" +
    ".n button:disabled{opacity:.35;cursor:default}" +
    ".l{display:grid;grid-auto-flow:column;grid-auto-columns:min(62%,14rem);gap:var(--g);margin:0;padding:0 0 .5rem;list-style:none;overflow-x:auto;scroll-snap-type:x mandatory;overscroll-behavior-x:contain;scrollbar-width:thin}" +
    ".l:focus-visible{outline:2px solid;outline-offset:4px}" +
    "@container (min-width:30rem){.l{grid-auto-flow:row;grid-template-columns:repeat(auto-fill,minmax(9.5rem,1fr));overflow:visible}}" +
    "@container (min-width:48rem){.many .l{grid-auto-flow:column;grid-template-columns:none;grid-auto-columns:calc((100% - 5 * var(--g)) / 6);overflow-x:auto}.many .n{display:flex}}" +
    ".c{position:relative;display:flex;flex-direction:column;gap:.3rem;scroll-snap-align:start;min-inline-size:0}" +
    ".m{inline-size:100%;block-size:auto;aspect-ratio:var(--mitate-media-ratio,4/5);object-fit:cover;border-radius:var(--mitate-radius,6px);background:var(--mitate-media-bg,color-mix(in srgb,currentColor 6%,transparent));margin-block-end:.2rem}" +
    ".a{color:inherit;font-weight:500;line-height:1.3;text-decoration:none}" +
    ".a::after{content:'';position:absolute;inset:0;border-radius:var(--mitate-radius,6px)}" +
    ".a:hover{text-decoration:underline;text-underline-offset:.18em}" +
    ".a:focus-visible{outline:none}.a:focus-visible::after{outline:2px solid;outline-offset:3px}" +
    ".p{font-variant-numeric:tabular-nums}" +
    ".y,.w,.f{color:color-mix(in srgb,currentColor 75%,transparent);font-size:.85em;line-height:1.35}" +
    ".y{display:flex;gap:.35em;align-items:start;font-variant-numeric:tabular-nums}" +
    ".y svg{flex:none;inline-size:1em;block-size:1em;margin-block-start:.12em;fill:none;stroke:currentColor;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}" +
    ".f{font-size:.75em;letter-spacing:.04em;text-transform:uppercase}" +
    "details{position:relative;z-index:1;font-size:.8em}" +
    "summary{inline-size:max-content;cursor:pointer;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:.2em}" +
    "details p{margin:.35rem 0 0}" +
    ".in{animation:i .24s cubic-bezier(.22,1,.36,1) both;animation-delay:calc(var(--i) * 30ms)}" +
    "@keyframes i{from{opacity:0;translate:0 6px}}" +
    "@media (prefers-reduced-motion:reduce){.in{animation-duration:.12s;animation-delay:0s}@keyframes i{from{opacity:0}}}";

  var sheet = null;
  var roots = new WeakMap();
  var settled = new WeakSet();
  var queue = [];
  var timer = 0;

  function storage() {
    try {
      return window.sessionStorage;
    } catch (e) {
      return null; // storage blocked: run without a session id
    }
  }

  function sessionId() {
    if (cfg.consent !== "granted") return null; // no identifier without consent
    var s = storage();
    var id = s && s.getItem(SESSION_KEY);
    if (!id) {
      id = (crypto.randomUUID && crypto.randomUUID()) || String(Math.random()).slice(2);
      if (s) s.setItem(SESSION_KEY, id);
    }
    return id;
  }

  function personal() {
    return cfg.consent === "granted" ? cfg.shopperId : null;
  }

  function post(path, body) {
    return fetch(cfg.api + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Mitate-Key": cfg.key },
      body: JSON.stringify(body),
      keepalive: true,
    }).then(function (r) {
      if (!r.ok) throw new Error("mitate " + r.status);
      var st = /dur=([\d.]+)/.exec(r.headers.get("Server-Timing") || "");
      return r.json().then(function (j) {
        j._serverMs = st ? parseFloat(st[1]) : null;
        return j;
      });
    });
  }

  function event(type, itemId, extra) {
    var e = {
      session_id: sessionId() || "anonymous",
      item_id: String(itemId),
      type: type,
      consent: cfg.consent,
      age_status: cfg.ageStatus,
    };
    if (personal()) e.shopper_id = personal();
    if (cfg.region) e.context = { region: cfg.region };
    for (var k in extra || {}) e[k] = extra[k];
    return post("/v1/events", { events: [e] }).catch(function () {});
  }

  function money(v) {
    if (v === null || v === undefined) return "";
    try {
      var whole = v % 1 === 0; // ₹1,299 rather than ₹1,299.00; paise only when the price has them
      return new Intl.NumberFormat(cfg.locale, {
        style: "currency",
        currency: cfg.currency,
        minimumFractionDigits: whole ? 0 : undefined,
        maximumFractionDigits: whole ? 0 : undefined,
      }).format(v);
    } catch (e) {
      return String(v);
    }
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text; // never innerHTML: titles and reasons come from the catalog
    return n;
  }

  function glyph(code) {
    var s = document.createElementNS(SVG, "svg");
    s.setAttribute("viewBox", "0 0 16 16");
    s.setAttribute("aria-hidden", "true");
    var p = document.createElementNS(SVG, "path");
    p.setAttribute("d", GLYPH[code] || GLYPH.similar);
    s.appendChild(p);
    return s;
  }

  function root(node) {
    var box = roots.get(node);
    if (box) return box;
    var r = node.attachShadow({ mode: "open" });
    if (r.adoptedStyleSheets && "replaceSync" in CSSStyleSheet.prototype) {
      if (!sheet) {
        sheet = new CSSStyleSheet();
        sheet.replaceSync(CSS);
      }
      r.adoptedStyleSheets = [sheet];
    } else {
      r.appendChild(el("style", "", CSS));
    }
    box = r.appendChild(el("div"));
    roots.set(node, box);
    return box;
  }

  function card(it, i, res, animate) {
    var li = el("li", animate ? "c in" : "c");
    li.style.setProperty("--i", Math.min(i, 6));
    if (it.image_url) {
      var img = el("img", "m");
      img.src = it.image_url;
      img.alt = ""; // decorative: the title sits right below
      img.loading = "lazy";
      img.decoding = "async";
      img.width = 400;
      img.height = 500;
      li.appendChild(img);
    }
    if (it.reason_code === "featured") li.appendChild(el("span", "f", "Featured by the store"));
    var a = el("a", "a", it.title);
    a.href = cfg.urlTemplate.replace("{id}", encodeURIComponent(it.id));
    a.addEventListener("click", function () {
      event("click", it.id, { impression_id: res.impression_id });
    });
    li.appendChild(a);
    if (it.price !== null && it.price !== undefined) li.appendChild(el("span", "p", money(it.price)));
    if (it.reason && it.reason_code !== "featured") {
      var y = el("p", "y");
      y.appendChild(glyph(it.reason_code));
      y.appendChild(el("span", "", it.reason));
      li.appendChild(y);
      var scope = SCOPE[it.reason_code];
      if (scope) {
        var why = el("details", "w");
        why.appendChild(el("summary", "", "Why?"));
        why.appendChild(el("p", "", it.reason + "."));
        why.appendChild(el("p", "", scope));
        if (COUNTED[it.reason_code]) {
          why.appendChild(el("p", "", "Counts appear only when at least 5 shoppers stand behind them."));
        }
        li.appendChild(why);
      }
    }
    return li;
  }

  function render(node, res, ms) {
    var r = root(node);
    r.textContent = "";
    var empty = !res.items.length; // holdout arm, or nothing worth showing: no padding, no rail
    node.hidden = empty;
    if (!empty) {
      var animate = !settled.has(node);
      settled.add(node);
      var title = node.dataset.title || cfg.titles[res.placement];
      var id = "mitate-" + res.impression_id;
      var sec = el("section", res.items.length > 6 ? "r many" : "r");
      sec.setAttribute("aria-labelledby", id);
      var head = el("div", "h");
      var h = el(/^h[2-6]$/.test(node.dataset.heading || "") ? node.dataset.heading : cfg.heading, "t", title);
      h.id = id;
      head.appendChild(h);
      var list = el("ul", "l");
      list.id = id + "-l";
      list.tabIndex = 0;
      list.setAttribute("aria-label", title);
      var nav = el("div", "n");
      [["Previous", -1, "←"], ["Next", 1, "→"]].forEach(function (b) {
        var btn = el("button", "", b[2]);
        btn.type = "button";
        btn.setAttribute("aria-label", b[0] + " items");
        btn.setAttribute("aria-controls", list.id);
        btn.addEventListener("click", function () {
          list.scrollBy({ left: b[1] * list.clientWidth * 0.9, behavior: "smooth" });
        });
        nav.appendChild(btn);
      });
      function edges() {
        nav.children[0].disabled = list.scrollLeft <= 1;
        nav.children[1].disabled = list.scrollLeft + list.clientWidth >= list.scrollWidth - 1;
      }
      list.addEventListener("scroll", edges, { passive: true });
      head.appendChild(nav);
      res.items.forEach(function (it, i) {
        list.appendChild(card(it, i, res, animate));
      });
      sec.appendChild(head);
      sec.appendChild(list);
      r.appendChild(sec);
      requestAnimationFrame(edges);
    }
    node.dispatchEvent(
      new CustomEvent("mitate:render", {
        bubbles: true,
        detail: { response: res, ms: ms, serverMs: res._serverMs },
      })
    );
  }

  function request(node) {
    var items = (node.dataset.items || node.dataset.item || "")
      .split(",")
      .map(function (s) { return s.trim(); })
      .filter(Boolean);
    var body = {
      placement: node.dataset.mitate,
      item_ids: items,
      consent: cfg.consent,
      age_status: cfg.ageStatus,
      k: parseInt(node.dataset.k || "8", 10),
    };
    var sid = sessionId();
    if (sid) body.session_id = sid;
    if (personal()) body.shopper_id = personal();
    if (cfg.region) body.region = cfg.region;
    if (cfg.xray) body.xray = true;
    return body;
  }

  function flush() {
    timer = 0;
    var nodes = queue.splice(0, 8); // the API batches up to 8 placements per call
    if (queue.length) timer = setTimeout(flush, 0);
    var t0 = performance.now();
    return post("/v1/recommend/batch", { requests: nodes.map(request) })
      .then(function (r) {
        var ms = performance.now() - t0;
        r.responses.forEach(function (res, i) {
          res._serverMs = r._serverMs;
          render(nodes[i], res, ms);
        });
      })
      .catch(function () {
        nodes.forEach(function (n) { n.hidden = true; });
      });
  }

  function want(node) {
    if (queue.indexOf(node) < 0) queue.push(node);
    if (!timer) timer = setTimeout(flush, 0);
  }

  var io =
    "IntersectionObserver" in window
      ? new IntersectionObserver(
          function (entries) {
            entries.forEach(function (e) {
              if (e.isIntersecting) {
                io.unobserve(e.target);
                want(e.target);
              }
            });
          },
          { rootMargin: "600px" }
        )
      : null;

  function scan() {
    document.querySelectorAll("[data-mitate]").forEach(function (n) {
      if (io && !settled.has(n)) io.observe(n);
      else want(n);
    });
  }

  function refresh() {
    document.querySelectorAll("[data-mitate]").forEach(function (n) {
      if (settled.has(n) || !io) want(n);
    });
  }

  window.Mitate = {
    set: function (o) {
      for (var k in o) cfg[k] = o[k];
    },
    setConsent: function (c) {
      cfg.consent = c;
      if (c !== "granted") {
        var s = storage();
        if (s) s.removeItem(SESSION_KEY);
      }
      refresh();
    },
    setShopper: function (id) {
      cfg.shopperId = id || null;
      refresh();
    },
    track: function (type, itemId, opts) {
      var extra = {};
      if (opts && opts.value !== undefined) extra.value = opts.value;
      return event(type, itemId, extra);
    },
    scan: scan, // call after adding placement elements to the page
    refresh: refresh,
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scan);
  else scan();
})();
