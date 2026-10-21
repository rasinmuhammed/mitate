// Misata Kitchen: a hash-routed storefront over products.json with Mitate rails on every page,
// plus the demo frame (shopper switch, consent, host skins and the x-ray ledger).
(async function () {
  "use strict";

  const DEMO = window.DEMO;
  const products = await (await fetch("products.json")).json();
  const byId = new Map(products.map((p) => [p.id, p]));
  const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
  const count = new Intl.NumberFormat("en-IN");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const $ = (s) => document.querySelector(s);
  const view = $("#view");

  Mitate.set({ key: DEMO.publicKey });

  // ---------------------------------------------------------------- small helpers

  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v === null || v === undefined) continue;
      if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) n.append(c);
    return n;
  }

  const leafName = (category) => category.split(" > ").pop();
  const deptOf = (category) => category.split(" > ")[0];
  const productHref = (id) => `#/p/${encodeURIComponent(id)}`;
  const categoryHref = (path) => `#/c/${encodeURIComponent(path)}`;
  const seal = () => el("span", { class: "seal mark", "aria-hidden": "true" });

  // a seal's resting angle varies by item, so stamps look placed by hand (brief 7.6.3)
  function tiltFor(id) {
    let h = 0;
    for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) | 0;
    return `${((Math.abs(h) % 41) / 10 - 2 - 1.5).toFixed(1)}deg`;
  }

  function plate(p, attrs) {
    return el("img", { src: p.image, alt: "", width: 400, height: 500, decoding: "async", ...attrs });
  }

  function storage() {
    try {
      return window.sessionStorage;
    } catch {
      return null;
    }
  }

  function toast(...content) {
    const region = $(".toast-region");
    const t = el("div", { class: "toast" }, ...content);
    region.append(t);
    let timer;
    const arm = () => { timer = setTimeout(() => t.remove(), 4000); };
    t.addEventListener("pointerenter", () => clearTimeout(timer));
    t.addEventListener("pointerleave", arm);
    arm();
  }

  // ---------------------------------------------------------------- catalog shape

  const leaves = [...new Set(products.map((p) => p.category))].sort();
  const depts = [...new Set(leaves.map(deptOf))];
  const firstOf = new Map();
  for (const p of products) if (!firstOf.has(p.category) && p.stock > 0) firstOf.set(p.category, p);

  const deptNav = $(".depts ul");
  for (const d of depts) deptNav.append(el("li", {}, el("a", { href: categoryHref(d) }, d)));

  // ---------------------------------------------------------------- cart and provenance

  const cart = new Map(); // id -> quantity; lives only as long as the tab
  const via = new Map(); // id -> title of the rail the shopper picked it from
  const cartCount = $("#cart-count");

  function updateCartCount(bump) {
    let n = 0;
    for (const q of cart.values()) n += q;
    cartCount.textContent = String(n);
    if (bump && !reduceMotion.matches) {
      cartCount.classList.remove("bump");
      void cartCount.offsetWidth; // restart the animation
      cartCount.classList.add("bump");
    }
  }

  // A click inside a rail reaches the page retargeted to the placement element; the composed
  // path still holds the card's link, which tells us which product came from which rail.
  document.addEventListener("click", (e) => {
    const host = e.target.closest?.("[data-mitate]");
    if (!host) return;
    const link = e.composedPath().find((n) => n instanceof HTMLAnchorElement);
    const m = link && /#\/p\/(.+)$/.exec(link.getAttribute("href") || "");
    if (m) via.set(decodeURIComponent(m[1]), host.dataset.title);
  });

  // ---------------------------------------------------------------- consent and shopper

  const consentBar = $("#consent");
  const consentChip = $("#consent-state");
  let consent = storage()?.getItem("demo.consent") === "granted" ? "granted" : "unknown";

  function applyConsent(c, announce) {
    consent = c;
    Mitate.setConsent(c);
    const s = storage();
    if (s) c === "granted" ? s.setItem("demo.consent", c) : s.removeItem("demo.consent");
    consentBar.hidden = c !== "unknown";
    consentChip.dataset.state = c;
    consentChip.textContent = { granted: "Personal: on", denied: "Personal: off", unknown: "Personal: ask" }[c];
    consentChip.setAttribute("aria-expanded", String(!consentBar.hidden));
    if (announce) {
      toast(c === "granted"
        ? "Personalisation on. Rails can now use this visit and the chosen shopper's history."
        : "Personalisation off. Rails use only the product on screen.");
    }
  }
  consentBar.addEventListener("click", (e) => {
    const b = e.target.closest("[data-consent]");
    if (b) { applyConsent(b.dataset.consent, true); route(); }
  });
  consentChip.addEventListener("click", () => {
    consentBar.hidden = !consentBar.hidden;
    consentChip.setAttribute("aria-expanded", String(!consentBar.hidden));
    if (!consentBar.hidden) consentBar.querySelector("button").focus();
  });
  applyConsent(consent, false);

  const shopperSelect = $("#shopper");
  DEMO.shoppers.forEach((s, i) => shopperSelect.append(new Option(`Returning shopper ${i + 1}`, s)));
  shopperSelect.addEventListener("change", () => {
    Mitate.setShopper(shopperSelect.value || null);
    if (shopperSelect.value && consent !== "granted") {
      consentBar.hidden = false;
      toast("This shopper's history stays unused until personalisation is allowed.");
    }
    route();
  });

  // ---------------------------------------------------------------- host skins

  const skinSelect = $("#skin");
  skinSelect.addEventListener("change", () => {
    const apply = () => { document.documentElement.dataset.skin = skinSelect.value; };
    if (document.startViewTransition && !reduceMotion.matches) document.startViewTransition(apply);
    else apply();
  });

  // ---------------------------------------------------------------- x-ray

  const xraySwitch = $("#xray");
  const ledgers = new WeakMap(); // placement element -> its ledger
  const lastRender = new WeakMap(); // placement element -> last mitate:render detail

  const HIDDEN = {
    blocked: "blocked by the store",
    out_of_stock: "out of stock",
    not_delivered: "not delivered to this region",
    on_screen_or_in_cart: "already on screen or in the cart",
    price_band: "outside the price band",
    category_cap: "over the per-category cap",
  };
  // "full" and "no_<source>" are the shipped rankers trained on pooled OTTO stores
  const rankerName = (r) =>
    r === "rrf" ? "rank fusion (no ranker for these sources)"
    : r === "full" ? "learned ranker"
    : r.startsWith("no_") ? `learned ranker, without ${r.slice(3)}`
    : r;
  const STAGE = { catalog_only: "catalog only", behavioural: "learning from shoppers" };

  function titleOf(id) {
    return byId.get(id)?.title ?? id;
  }

  function evidence(item) {
    const e = item.evidence || {};
    switch (item.reason_code) {
      case "bought_with": return `${e.shoppers} shoppers · with ${titleOf(e.with)}`;
      case "viewed_next": return `${e.shoppers} shoppers · after ${titleOf(e.after)}`;
      case "similar": return (e.shared || []).join(", ");
      case "pairs_with": return (e.categories || []).map((c) => leafName(c || "?")).join(" → ") + " · approved";
      case "bought_before": return `${e.times}× in your orders`;
      case "popular": return `${e.shoppers_7d} shoppers in 7 days`;
      case "new": return `${e.age_days} days in the catalog`;
      case "featured": return "pinned by the store";
      default: return "no fact clears the 5-shopper bar: no reason shown";
    }
  }

  function ago(ts) {
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 90) return "just now";
    if (s < 5400) return `${Math.round(s / 60)} min ago`;
    if (s < 129600) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} days ago`;
  }

  function ledger(node, detail) {
    const res = detail.response;
    const x = res.xray;
    const facts = (pairs) => el("dl", { class: "xr-facts" },
      pairs.filter(Boolean).map(([k, v]) => el("div", {}, el("dt", {}, k), el("dd", {}, v))));
    const timing = [`${Math.round(detail.ms)} ms round trip`];
    if (detail.serverMs != null) timing.push(`${detail.serverMs.toFixed(1)} ms server`);
    const head = el("div", { class: "xr-head" },
      seal(), el("b", {}, "X-ray"), el("span", { class: "placement" }, res.placement),
      el("span", { class: "xr-ms" }, timing.join(" · ")));
    if (!x) {
      return el("div", { class: "xr" }, head,
        el("p", { class: "xr-hidden" }, "This store has not switched the x-ray on."));
    }
    const hiddenTotal = Object.values(x.hidden).reduce((a, b) => a + b, 0);
    const hiddenText = hiddenTotal
      ? `Rules held back ${hiddenTotal} better-scored ${hiddenTotal === 1 ? "item" : "items"}: ` +
        Object.entries(x.hidden).filter(([, n]) => n).map(([k, n]) => `${n} ${HIDDEN[k] || k}`).join(", ") + "."
      : "Rules held nothing back.";
    const items = el("ol", { class: "xr-items", role: "list" }, x.items.map((it, i) => {
      const p = byId.get(it.id);
      return el("li", {},
        el("span", { class: "n" }, String(i + 1).padStart(2, "0")),
        p ? plate(p, { loading: "lazy" }) : el("span"),
        el("span", { class: "t" }, titleOf(it.id)),
        el("span", { class: "ev" },
          it.pinned ? el("span", { class: "pin" }, "pinned ") : null,
          el("code", {}, it.reason_code || "none"), " ", evidence(it)));
    }));
    return el("div", { class: "xr", role: "region", "aria-label": `X-ray: ${node.dataset.title}` },
      head,
      facts([
        ["Stage", STAGE[x.stage] || x.stage],
        ["Ranker", rankerName(x.ranker)],
        ["Model", res.model_version != null ? `v${res.model_version}, trained ${ago(x.trained_at)}` : "none"],
        ["Learned from", `${count.format(x.events_learned_from)} events`],
        ["Personal", x.history_events ? `${x.personalisation} · ${x.history_events} events` : x.personalisation],
        ["Sources", x.sources.join(" · ") || "catalog"],
        res.arm ? ["Experiment arm", res.arm] : null,
        ["Engine time", `${x.server_ms.toFixed(1)} ms`],
      ]),
      el("p", { class: "xr-hidden" }, hiddenText),
      items);
  }

  function drawLedger(node) {
    ledgers.get(node)?.remove();
    ledgers.delete(node);
    const detail = lastRender.get(node);
    if (!xraySwitch.checked || !detail || node.hidden || !node.isConnected) return;
    const l = ledger(node, detail);
    node.after(l);
    ledgers.set(node, l);
  }

  document.addEventListener("mitate:render", (e) => {
    lastRender.set(e.target, e.detail);
    drawLedger(e.target);
  });

  function setXray(on) {
    xraySwitch.checked = on;
    document.documentElement.classList.toggle("xray-on", on);
    Mitate.set({ xray: on });
    if (on) Mitate.refresh(); // ask again with the x-ray flag; rails do not re-animate
    else document.querySelectorAll(".xr").forEach((n) => n.remove());
  }
  xraySwitch.addEventListener("change", () => setXray(xraySwitch.checked));
  document.addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() !== "x" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest("input, select, textarea, [contenteditable]")) return;
    setXray(!xraySwitch.checked);
  });

  // ---------------------------------------------------------------- views

  function rail(placement, title, items) {
    return el("div", {
      "data-mitate": placement,
      "data-title": title,
      "data-items": items && items.length ? items.join(",") : null,
    });
  }

  function card(p) {
    return el("article", { class: "card" },
      plate(p, { loading: "lazy", "data-plate": p.id }),
      el("a", { href: productHref(p.id) }, p.title),
      el("div", { class: "meta" },
        el("span", { class: "price" }, money.format(p.price)),
        p.stock > 0 ? el("span", {}, p.material) : el("span", { class: "oos" }, "Out of stock")));
  }

  function home() {
    const pick = (cat) => products.find((p) => p.category === cat && p.stock > 0);
    const trio = ["Cookware > Kadai & Woks", "Tea & Coffee > Coffee Filters", "Table > Bowls"]
      .map(pick).filter(Boolean);
    const personal = consent === "granted" && shopperSelect.value;
    return [
      el("section", { class: "hero" },
        el("div", { class: "hero-copy" },
          el("p", { class: "hero-count" }, `${count.format(products.length)} pieces · ${leaves.length} categories`),
          el("h1", { class: "title" }, "Pans that season. Tins that stack. Tea that gets made."),
          el("p", { class: "hero-lede" },
            "Cast iron, brass, steel and stoneware for kitchens in daily use. " +
            "Every suggestion on this site says why it is there."),
          el("a", { class: "s-btn buy", href: categoryHref("Cookware") }, "Shop cookware")),
        el("div", { class: "trio" }, trio.map((p) => el("a", { href: productHref(p.id), "aria-label": p.title },
          plate(p, { fetchpriority: "high", "data-plate": p.id }))))),
      personal ? rail("returning", "Picked for you", null) : null,
      rail("next", "What shoppers are looking at", null),
      el("h2", { class: "section-title" }, "Shop by department"),
      el("div", { class: "depts-index" }, depts.map((d) => el("section", { class: "dept" },
        el("h3", {}, d),
        el("ul", { class: "leaves", role: "list" }, leaves.filter((l) => deptOf(l) === d).map((l) => {
          const p = firstOf.get(l);
          const n = products.filter((q) => q.category === l).length;
          return el("li", {}, el("a", { class: "leaf", href: categoryHref(l) },
            p ? plate(p, { loading: "lazy", "data-plate": p.id }) : null,
            el("b", {}, leafName(l)), el("small", {}, `${n} pieces`)));
        }))))),
    ];
  }

  function listing(path) {
    const items = products.filter((p) => p.category === path || p.category.startsWith(path + " > "));
    if (!items.length) return home();
    const dept = deptOf(path);
    const isLeaf = path.includes(" > ");
    return [
      el("nav", { class: "crumbs", "aria-label": "Breadcrumb" },
        el("a", { href: "#/" }, "Home"), el("span", { "aria-hidden": "true" }, "/"),
        isLeaf ? [el("a", { href: categoryHref(dept) }, dept), el("span", { "aria-hidden": "true" }, "/")] : null,
        el("span", { "aria-current": "page" }, leafName(path))),
      el("div", { class: "list-head" },
        el("h1", { class: "title" }, leafName(path)),
        el("p", {}, `${items.length} pieces`)),
      el("nav", { class: "filters", "aria-label": `${dept} categories` },
        el("a", { href: categoryHref(dept), "aria-current": isLeaf ? null : "page" }, `All ${dept.toLowerCase()}`),
        leaves.filter((l) => deptOf(l) === dept).map((l) =>
          el("a", { href: categoryHref(l), "aria-current": l === path ? "page" : null }, leafName(l)))),
      el("ul", { class: "grid", role: "list" }, items.map((p) => el("li", {}, card(p)))),
    ];
  }

  function product(id, morph) {
    const p = byId.get(id);
    if (!p) return home();
    const inStock = p.stock > 0;
    const frame = el("div", { class: "product-plate" },
      plate(p, { fetchpriority: "high", style: morph ? "view-transition-name: plate" : null }));
    const status = el("p", { class: "visually-hidden", role: "status" });
    const add = el("button", { class: "s-btn buy", type: "button", disabled: !inStock }, inStock ? "Add to cart" : "Out of stock");
    const buy = el("button", { class: "s-btn", type: "button", disabled: !inStock }, "Buy now");

    function addToCart() {
      cart.set(p.id, (cart.get(p.id) || 0) + 1);
      Mitate.track("cart", p.id);
      updateCartCount(true);
      add.textContent = `In your cart (${cart.get(p.id)})`;
      status.textContent = `${p.title} added to cart.`;
      const from = via.get(p.id);
      if (from && !frame.querySelector(".stamp")) {
        frame.append(el("div", { class: "stamp", style: `--tilt: ${tiltFor(p.id)}` },
          seal(), el("span", {}, `Chosen from “${from}”`)));
      }
    }
    add.addEventListener("click", addToCart);
    buy.addEventListener("click", () => { addToCart(); location.hash = "#/cart"; });

    return [
      el("nav", { class: "crumbs", "aria-label": "Breadcrumb" },
        el("a", { href: "#/" }, "Home"), el("span", { "aria-hidden": "true" }, "/"),
        el("a", { href: categoryHref(deptOf(p.category)) }, deptOf(p.category)), el("span", { "aria-hidden": "true" }, "/"),
        el("a", { href: categoryHref(p.category) }, leafName(p.category))),
      el("section", { class: "product" },
        frame,
        el("div", { class: "product-info" },
          el("h1", { class: "title" }, p.title),
          el("p", { class: "price" }, money.format(p.price)),
          el("dl", { class: "facts" },
            el("dt", {}, "Material"), el("dd", {}, p.material),
            el("dt", {}, "Category"), el("dd", {}, leafName(p.category)),
            el("dt", {}, "Availability"), el("dd", {}, inStock ? "In stock" : "Out of stock")),
          el("div", { class: "actions" }, add, buy),
          status,
          el("p", { class: "blurb" },
            `A synthetic listing. The picture is drawn from the product's id, so the same piece always looks the same.`))),
      rail("complements", "Goes well with", [p.id]),
      rail("similar", "Similar pieces", [p.id]),
      rail("next", "Shoppers went on to view", [p.id]),
    ];
  }

  function cartView() {
    const ids = [...cart.keys()];
    if (!ids.length) {
      return [
        el("section", { class: "empty" },
          el("h1", { class: "title" }, "Your cart is empty"),
          el("a", { class: "s-btn", href: "#/" }, "Back to the shop")),
        rail("next", "What shoppers are looking at", null),
      ];
    }
    let total = 0;
    const lines = ids.map((id) => {
      const p = byId.get(id);
      const q = cart.get(id);
      total += p.price * q;
      const qty = el("select", { "aria-label": `Quantity of ${p.title}` },
        [1, 2, 3, 4, 5].map((n) => el("option", { value: n, selected: n === q }, String(n))));
      qty.addEventListener("change", () => { cart.set(id, Number(qty.value)); updateCartCount(false); route(); });
      const remove = el("button", { type: "button" }, "Remove");
      remove.addEventListener("click", () => { cart.delete(id); updateCartCount(false); route(); });
      const from = via.get(id);
      return el("li", { class: "line" },
        plate(p, { loading: "lazy" }),
        el("div", {},
          el("a", { href: productHref(id) }, p.title),
          from ? el("p", { class: "via" },
            el("span", { class: "seal mark", "aria-hidden": "true", style: `--seal-tilt: ${tiltFor(id)}` }),
            `Chosen from “${from}”`) : null),
        el("div", { class: "line-end" },
          el("span", { class: "price" }, money.format(p.price * q)), qty, remove));
    });
    const place = el("button", { class: "s-btn buy", type: "button" }, "Place order");
    place.addEventListener("click", () => {
      for (const [id, q] of cart) Mitate.track("order", id, { value: byId.get(id).price * q });
      cart.clear();
      updateCartCount(false);
      route();
      toast(seal(), "Order placed. Nothing was charged: this store is synthetic.");
    });
    return [
      el("div", { class: "list-head" }, el("h1", { class: "title" }, "Cart")),
      el("section", { class: "cart" },
        el("ul", { class: "lines", role: "list" }, lines),
        el("aside", { class: "summary", "aria-label": "Order summary" },
          el("dl", {},
            el("dt", {}, "Subtotal"), el("dd", {}, money.format(total)),
            el("dt", {}, "Delivery"), el("dd", {}, "Free"),
            el("dt", { class: "total" }, "Total"), el("dd", { class: "total" }, money.format(total))),
          place,
          el("small", {}, "Placing an order records it for the recommender. No payment is taken."))),
      rail("complements", "Goes well with your cart", ids),
    ];
  }

  // ---------------------------------------------------------------- routing

  let morphFrom = null; // the clicked plate, for a shared-element transition into the product page
  view.addEventListener("click", (e) => {
    const a = e.target.closest("a[href^='#/p/']");
    morphFrom = a?.closest(".card, .leaf, .trio a, li")?.querySelector("img[data-plate]") || null;
  });

  let first = true;
  let shown = null; // the hash on screen: re-rendering it (shopper or consent change) is not a new view
  function render(morph) {
    const hash = location.hash || "#/";
    let m;
    let content;
    if ((m = /^#\/p\/(.+)$/.exec(hash))) {
      const id = decodeURIComponent(m[1]);
      if (hash !== shown && byId.has(id)) Mitate.track("view", id);
      content = product(id, morph);
    }
    else if ((m = /^#\/c\/(.+)$/.exec(hash))) content = listing(decodeURIComponent(m[1]));
    else if (hash === "#/cart") content = cartView();
    else content = home();
    view.replaceChildren(...[content].flat().filter(Boolean));
    const dept = /^#\/c\/(.+)$/.exec(hash);
    for (const a of deptNav.querySelectorAll("a")) {
      const active = dept && deptOf(decodeURIComponent(dept[1])) === a.textContent;
      active ? a.setAttribute("aria-current", "true") : a.removeAttribute("aria-current");
    }
    window.scrollTo(0, 0);
    if (!first) view.focus({ preventScroll: true });
    first = false;
    shown = hash;
    Mitate.scan();
  }

  function route() {
    const source = morphFrom;
    if (source && document.startViewTransition && !reduceMotion.matches) {
      source.style.viewTransitionName = "plate";
      const t = document.startViewTransition(() => render(true));
      t.finished.finally(() => {
        view.querySelectorAll("[style*='view-transition-name']").forEach((n) => { n.style.viewTransitionName = ""; });
      });
    } else {
      render(false);
    }
    morphFrom = null;
  }

  // shareable demo states: ?skin=night&xray=1
  const params = new URLSearchParams(location.search);
  if (params.has("skin") && [...skinSelect.options].some((o) => o.value === params.get("skin"))) {
    skinSelect.value = params.get("skin");
    document.documentElement.dataset.skin = skinSelect.value;
  }
  if (params.get("xray") === "1") setXray(true);

  window.addEventListener("hashchange", route);
  updateCartCount(false);
  route();
})();
