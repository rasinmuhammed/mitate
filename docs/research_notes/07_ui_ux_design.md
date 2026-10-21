# 07: UI and UX design direction for the widget, demo storefront and dashboard

Scope: visual and interaction design for Mitate's three surfaces: the storefront widget (`widget/`), the demo storefront (`serve/demo_site/`) and the merchant dashboard (`dashboard/`).
Review date: 2026-10-20. Every factual claim links to its source. Labels used below:

- **[measured]**: measured for this note. The method is stated next to the number.
- **[secondary]**: read in a secondary report of the primary source.
- **[unverified]**: I could not confirm it.
- **[judgement]**: a design decision, not a finding.

Fonts, library sizes and Baseline status were measured on 2026-10-20 from the Google Fonts repository, jsDelivr and the `web-features` npm package v3.43.0. Commands and versions are given in §3.

---

## Executive summary

1. **Direction: "Hakogaki" (箱書き, the inscribed box).** In tea culture, a prized object's wooden box carries an inscription that names the object and says who vouched for it. Mitate's product is a choice plus an honest record of why it was made, so the visual system treats every recommendation as an object with a provenance label. The palette is sumi ink on unbleached paper, with a single vermilion seal that marks "chosen". Counted facts are set in a mono face, like a ledger. (§7.1)
2. **Trust is the aesthetic.** Gen Z distrusts polish and AI-made gloss, and trusts work that shows its proof ([eMarketer on Bentley-Gallup 2026](https://www.emarketer.com/content/gen-z-ai-ad-skepticism-puts-brand-authenticity-under-microscope), [Hiebing](https://hiebing.com/blog/why-gen-z-is-rejecting-brand-perfection), both [secondary]). Mitate's honest counted reasons and its x-ray view are that proof. They should be the most designed elements on every surface, not footnotes.
3. **The widget is a quiet guest.** It loads zero fonts and makes zero third-party requests, and it inherits the host's type and colour through Shadow DOM, so it looks native in any Shopify or WooCommerce theme. Budget: ≤ 5 KB JS + ≤ 1.5 KB CSS, gzip (§7.9).
4. **Reasons are visible by default and specific.** Text explanations beat visual ones ([Kouki et al. 2019](https://linqs.org/assets/resources/kouki-tiis20.pdf)). Item- and content-based reasons beat social ones in that study. In an A/B test on JD.com, specific feature-level explanations lifted CTR (4.34%) while a generic "People also viewed" label (3.22%) did no better than none (3.20%) ([Zhang et al. SIGIR 2014](https://www.cs.cmu.edu/~glai1/papers/yongfeng-guokun-sigir14.pdf)).
5. **Type: Gloock, Anek Latin, IBM Plex Mono.** All three are OFL and all carry the ₹ glyph [measured]. Display is Gloock at 16.5 KB. UI is Anek Latin by Mumbai's Ek Type at 38.3 KB. Ledger figures use IBM Plex Mono at 9.9 KB per weight. All sizes are self-hosted WOFF2 Latin subsets. JetBrains Mono, Martian Mono, Red Hat Mono, Spline Sans Mono, Instrument Serif and Hanken Grotesk all lack ₹ (U+20B9) and are ruled out for an INR product [measured]. Devanagari and other Indic scripts fall back to system fonts, because a self-hosted Anek Devanagari subset costs ~206–224 KB [measured].
6. **Colour: OKLCH tokens checked against WCAG 2.2 AA, with APCA Lc as an internal target.** WCAG 2 stays the legal gate ([Roselli, Apr 2026](https://adrianroselli.com/2026/04/wcag3-contrast-as-of-april-2026.html)). The current widget's muted text, `color-mix(currentColor 60%, transparent)`, **fails 4.5:1** on common light themes (4.27:1 for `#222` on white; 3.71:1 for `#333`). 75% passes on every host pair tested [measured] (§3.6).
7. **Motion is short, interruptible and rare.** Durations are 120–360 ms with ease-out and CSS transitions, plus one spring built with `linear()`. No animation on keyboard-triggered actions ([Kowalski](https://emilkowal.ski/ui/great-animations)). No scroll-jacking ([NN/g 2023](https://www.nngroup.com/articles/scrolljacking-101/)). Fully replaced under `prefers-reduced-motion`.
8. **Dashboard: a calm ledger, not a KPI bento.** It is table- and sentence-first, with progressive disclosure capped at two levels ([Nielsen 2006](https://www.nngroup.com/articles/progressive-disclosure/)). Uncertainty is shown as ranges and frequency-style dot plots, which non-experts read better than error bars ([Kay et al. CHI 2016](https://idl.uw.edu/papers/when-ish-is-my-bus), [Hullman et al. 2015](https://idl.uw.edu/papers/hops)). A plain verdict appears when a store is too small to test.
9. **Stack: vanilla, no build.** Use Baseline CSS (container queries, `:has()`, layers, `oklch()`, `color-mix()`, popover, `@starting-style`, same-document view transitions, `linear()`) and treat non-Baseline features as enhancements: scroll-driven animations, cross-document view transitions, anchor positioning. The only library worth adding is uPlot (19.8 KB br) for dashboard time series. Preact+htm (4.9 KB br) is optional, if the dashboard's state outgrows hand-written DOM code (§8).
10. **Never fabricate.** Show no false urgency, disguised promotion or invented popularity. India's CCPA lists all three among 13 prohibited dark patterns ([Mondaq summary](https://www.mondaq.com/ccpas-guidelines-on-dark-patterns-an-overview/1844140)). Merchant-pinned items are labelled "Featured by the store", and counts are shown only at ≥ 5 shoppers (`Evidence.min_support` in `mitate/explain.py`).

---

## 1. Gen Z and current web design culture

### 1.1 What the 2025–2026 discourse says (weak evidence, consistent signal)

Most "trend" material comes from agencies and vendors, so I treat it as a description of the mood, not as data.

- **Anti-AI craft and visible imperfection.**
  - Creative Bloq's 2025 and 2026 roundups frame the turn to texture, organic lettering and "intentional human imperfections" as a backlash against "AI's 'ugly minimalism'" ([2025](https://www.creativebloq.com/design/graphic-design/10-graphic-design-trends-to-watch-in-2025-plus-4-to-avoid), [2026](https://creativebloq.com/design/graphic-design/texture-warmth-and-tactile-rebellion-the-big-graphic-design-trends-for-2026)).
  - Agency pieces repeat the claim that the purple-to-magenta "AI startup" gradient now reads as generic ([Visuapex](https://www.visuapexcreatives.com/2026-design-trends-anti-ai-craft/), [secondary]).
- **Type as the interface.**
  - Kinetic and variable type, broken grids and "visual personality" lead Envato's 2026 list ([Envato](https://elements.envato.com/learn/web-design-trends)).
  - Figma's trend page makes similar points ([Figma](https://www.figma.com/resource-library/web-design-trends/)).
  - Shopify's own design blog notes a tension between rejecting digital sterility and a return of ornament ([Shopify](https://www.shopify.com/blog/graphic-design-trends)).
- **Pinterest Predicts 2026.**
  - Based on search and save data, it names three drivers: "emotional comfort, intentional curation, and grounded optimism".
  - It reports that nearly four in ten Gen Z users want to curate a personal identity rather than copy aesthetics ([Campaign Asia](https://www.campaignasia.com/article/pinterest-predicts-the-biggest-gen-z-trends-of-2026/506572), [Inside Retail](https://insideretail.co.nz/2026/01/07/pinterest-predicts-2026-gen-z-drives-a-year-of-maximalism-and-modern-expression/), both [secondary]).
  - "Intentional curation" is close to a literal translation of *mitate*.

### 1.2 Trustworthy vs try-hard: the better evidence

- **AI-made advertising.** Per the Bentley University–Gallup 2026 survey:
  - 66% of US 18–29-year-olds view businesses' use of AI to create ads negatively.
  - 73% of adult Gen Z call AI-generated people or voices in ads unacceptable.
  - Source: [eMarketer](https://www.emarketer.com/content/gen-z-ai-ad-skepticism-puts-brand-authenticity-under-microscope), [secondary].
- **Polish.** In YouthInsight's "I Hate Ads" report (Australians aged 14–29), 40% say over-polished ads make them trust a brand less ([LBB](https://lbbonline.com/news/we-are-different-youth-insight-i-hate-ads-report), [secondary]).
- **Proof over polish.** Hiebing reports that Gen Z trusts brands that "show their work" ([Hiebing](https://hiebing.com/blog/why-gen-z-is-rejecting-brand-perfection), [secondary]).
- **What this means for Mitate [judgement].** Try-hard reads as decoration that claims something it cannot back up: gloss, fake 3D, gradients, invented urgency. Trustworthy reads as specificity: exact counts, named sources, visible method. Mitate's honest explanations are both its differentiator and its brand look.

### 1.3 The new generic: what "AI slop" looks like in 2026

- **Anthropic names the default.**
  - Unguided generation converges on "Inter fonts, purple gradients on white backgrounds, and minimal animations".
  - Banning one default tends to produce the next most common choice; Space Grotesk is the example given.
  - Source: [Claude blog, Nov 2025](https://claude.com/blog/improving-frontend-design-through-skills).
  - Two of that post's suggestions do not fit Mitate: "load from Google Fonts" conflicts with Mitate's no-third-party-request rule, and "avoid system fonts" does not apply to the widget, which should inherit the host's fonts.
- **Prestige styles are now templates too.**
  - Agent-ready DESIGN.md clones of Vercel and Nothing circulate as packages and plugins ([Vercel-inspired DESIGN.md on npm](https://cdn.jsdelivr.net/npm/oh-my-opencode@4.18.2/dist/skills/frontend/references/design/vercel.md), [shadcn.io "Nothing" design](https://www.shadcn.io/design/nothing)).
  - [judgement] A copy of Linear, Vercel or Nothing (near-black UI, Geist or Inter, hairline borders, a bento grid of KPI cards, a dot-matrix accent) is now itself a slop tell. Mitate needs a direction derived from its own idea.

### 1.4 Fads vs durable craft [judgement, informed by §1.1–1.3 and §2]

| Fad (date-stamps a UI) | Durable craft (ages well) |
|---|---|
| Glassmorphism, aurora/mesh gradients, purple-to-pink | A small, named palette in a perceptual space (OKLCH) with one disciplined accent |
| Bento KPI grids as the default dashboard | Information hierarchy: sentence first, then the one chart that answers it |
| Scroll-jacking, smooth-scroll libraries, WebGL heroes | Native scroll; motion that explains change and stays under 300 ms |
| Emoji section headers, sparkle icons, "AI-powered" badges | Plain verbs and nouns; honest status |
| Y2K chrome, glitch, stickers everywhere | One tactile motif used with purpose (here, the seal) |
| Inter, Space Grotesk, Instrument Serif + Inter "AI landing" pairing | A typeface chosen for coverage (₹, Indic), figures and character |
| Fake grain overlays via SVG turbulence on every surface | Texture that comes from the imagery metaphor (print, ink), applied sparingly |

### 1.5 India: the context that should shape every decision

- **Who shops.**
  - Gen Z is about 40% of India's e-retail shoppers, and 70% of Gen Z shoppers discover brands online.
  - Gen Z browses fewer search listings before buying, and over 90% of Gen Z digital transactors prefer UPI.
  - More than 270 million Indians shopped online in 2024.
  - Source: [Bain, How India Shops Online 2025, 26 Mar 2025](https://www.bain.com/insights/how-india-shops-online-2025/).
  - UPI processed 24.07 billion transactions in September 2026 ([Outlook Money](https://www.outlookmoney.com/banking/upi-processes-2407-billion-transactions-worth-rs-2937-lakh-crore-in-september), [secondary]).
- **Who is online.**
  - IAMAI–Kantar's 2025 report counts 958 million active internet users, 57% of them rural ([tele.net](https://tele.net.in/indias-internet-user-base-crosses-950-million-in-2025-as-per-iamai-report/), [secondary]).
  - The 2024 edition found 98% of users accessed content in Indic languages ([Business Today](https://www.businesstoday.in/amp/technology/news/story/indias-internet-revolution-key-insights-from-kantar-and-iamai-report-461043-2025-01-16), [secondary]).
- **Devices and networks.**
  - Counterpoint reports that premium phones (₹30,000+) were 22% of units shipped in 2025, so about 78% were below that ([GSMArena](https://www.gsmarena.com/counterpoint_1_in_5_smartphones_shipped_in_india_in_2025_were_premium_models-news-71366.php), [secondary]).
  - Alex Russell's 2026 reference phone is a Samsung Galaxy A24 4G-class device (Helio G99 / Exynos 1330 equivalents).
  - His India P75 network figures are 6.2 Mbps down, 2.1 Mbps up and 85 ms RTT ([Performance Inequality Gap 2026](https://infrequently.org/2025/11/performance-inequality-gap-2026/)).
  - **Design for that phone and that network.**
- **Vernacular.**
  - Meesho's designers found that users "scan words and only read the ones they're familiar with" and rewrote copy for simplicity in each language ([Meesho design blog](https://meesho.io/blog/my-first-6-months-as-a-product-designer-at-meesho)).
  - Meesho added 8 vernacular languages ([Meesho](https://meesho.io/blog/meesho-adds-8-new-vernacular-languages-to-its-platform)).
  - Mitate's reason strings are templates over counted facts, which makes them translatable. Keep them short and number-led.
- **Where the widget will live [measured].** I fetched Indian D2C home pages on 2026-10-20 with a Galaxy A24 Chrome user-agent and measured the decompressed HTML only. Most are Shopify stores, so the widget lands on pages that are already heavy.

  | Site | HTML (decompressed) | `<script>` tags | Shopify | Primary font in CSS |
  |---|---|---|---|---|
  | sleepyowl.co | 901 KB | 81 | yes | Rubik, Knockout |
  | bluetokaicoffee.com | 383 KB | 106 | yes | Brandon Grotesque, Cormorant |
  | bombayshavingcompany.com | 1,370 KB | 125 | yes | Poppins |
  | snitch.com | 310 KB | 23 | yes | custom var |
  | boat-lifestyle.com | 2,815 KB | 109 | yes | Metropolis |
  | mokobara.com | 1,524 KB | 123 | yes | Neurial Grotesk, Karla, Montserrat |
  | bluorng.com | 879 KB | 89 | yes | Assistant + custom |
  | thewholetruthfoods.com | 453 KB | 100 | no | Obviously Narrow |
  | myntra.com | 621 KB | 29 | no | Figtree |

  Meesho, Nykaa and The Souled Store returned bot-challenge or app-shell responses and are excluded.

  - Shopify allows an app to lower storefront Lighthouse performance by no more than 10 points ([Shopify](https://shopify.dev/docs/apps/build/performance)).
  - Shopify weights product pages at 40% of that score ([Shopify storefront](https://shopify.dev/docs/apps/build/performance/storefront)).
  - Product pages are exactly where Mitate renders.

**Implications [judgement].**

- Gen Z India is fast, mobile-first, social-led, UPI-native and visually literate. They are not starved for spectacle; they are drowning in it.
- What stands out is calm confidence, real specifics and speed on a ₹15k phone.
- Indian D2C sites lean on loud display grotesks and heavy pages. A restrained, fast and honest rail will look *more* premium by contrast.

---

## 2. Skills world-class product designers use now

### 2.1 Typography systems

- **Variable fonts and optical size.** They let one file cover a range, but only if the axes you ship are the ones you use. Pinning axes cut Fraunces from 122.8 KB to 32.5 KB and Newsreader from 134.6 KB to 55.3 KB [measured, §3.5].
- **Figures.** Use `font-variant-numeric: tabular-nums` for prices and counts. Gloock, Anek Latin and Geist have a `tnum` feature; monospaced faces are tabular by construction [measured].
- **Script coverage is a product requirement.** ₹ (U+20B9) must exist in every face that sets prices; six popular OFL faces fail this [measured].
- **Wrapping.** `text-wrap: balance` for headings is Baseline (2024) and `text-wrap: pretty` for paragraphs is Chromium + Safari 26 ([web-features 3.43.0](https://www.npmjs.com/package/web-features)). Linear sets headings in Inter Display "for more expression", with body text in regular Inter ([Linear](https://linear.app/now/how-we-redesigned-the-linear-ui)). Using an optical display cut is a craft move worth copying, not the font.

### 2.2 Layout and grid

- **Container queries are Baseline Widely available** (Aug 2025). They are the right tool for a widget whose slot width is unknown.
- **`subgrid` is Widely available** (Mar 2026). It aligns card internals across a rail (title, price and reason on shared baselines).
- **Style queries reached Baseline in May 2026**, all per [web-features](https://www.npmjs.com/package/web-features). They allow `@container style(--density: compact)` theming hooks.

### 2.3 Colour systems

- **Perceptual spaces.**
  - Linear moved theme generation from HSL to LCH and now derives a whole theme from three inputs (base, accent, contrast) instead of 98 variables ([Linear](https://linear.app/now/how-we-redesigned-the-linear-ui)).
  - Its 2026 refresh moved defaults to "a warmer, less saturated gray", warning that too warm turns muddy ([Linear, 12 Mar 2026](https://linear.app/now/behind-the-latest-design-refresh)).
  - Stripe built its accessible palette in a perceptually uniform space for the same reason ([Stripe](https://stripe.com/blog/accessible-color-systems)).
- **Contrast.**
  - The WCAG 3 contrast algorithm is "yet to be determined" as of April 2026, and APCA was removed from the 2023 draft.
  - Roselli's advice is to use APCA only with colours that also pass WCAG 2 ([Roselli](https://adrianroselli.com/2026/04/wcag3-contrast-as-of-april-2026.html)).
  - Mitate's tokens (§7.3) pass WCAG 2 AA and report APCA Lc as a second check.
- **CSS.** `oklch()` and `color-mix()` are Widely available (Nov 2025). Relative colour syntax (Sept 2024) and `light-dark()` (May 2024) are Newly available. `contrast-color()` became Newly available only in April 2026 ([web-features](https://www.npmjs.com/package/web-features)).

### 2.4 Motion

- **Kowalski's rules** ([great animations](https://emilkowal.ski/ui/great-animations)):
  - UI motion "should also usually be shorter than 300ms".
  - `ease-out` for feedback.
  - Interruptible CSS transitions rather than keyframes.
  - Respect `prefers-reduced-motion`.
  - "Never animate keyboard initiated actions".
  - Weigh how often a motion will be seen. He presents Raycast's lack of animation as correct.
- **Sonner's craft details.**
  - Transitions instead of keyframes so rapid toasts retarget smoothly.
  - Hover gaps filled with pseudo-elements so the hover state doesn't drop.
  - Velocity-based swipe dismissal (threshold 0.11).
  - Timers that pause while the tab is hidden.
  - `@starting-style` noted as the modern way to do enter states.
  - Source: [Building a toast component](https://emilkowal.ski/ui/building-a-toast-component).
- **Freiberg's "Invisible details"**: interruptibility, spatial consistency (motion shows where things come from), and frequency, i.e. removing motion from repeated actions because it becomes "a cognitive burden" ([Every](https://every.to/p/invisible-details-of-interaction-design)).
- **Springs in CSS.** `linear()` approximates springs and bounces ([Chrome for Developers](https://developer.chrome.com/docs/css-ui/css-linear-easing-function)) and is Baseline Widely available (June 2026). No JS spring library is needed.

### 2.5 Interaction design and accessibility (WCAG 2.2)

The nine success criteria new in 2.2 include:

- 2.4.11 Focus Not Obscured (AA). Sticky consent bars must not hide focus.
- 2.5.7 Dragging Movements (AA). Swipe-to-approve needs buttons.
- 2.5.8 Target Size (Minimum) 24×24 CSS px (AA).
- 3.3.8 Accessible Authentication (AA). The dashboard's paste-a-key login is fine; allow paste and password managers.

Source: [W3C](https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/).

### 2.6 Performance as UX

- **Thresholds.** Core Web Vitals "good" thresholds: LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1, at the 75th percentile ([web.dev](https://web.dev/articles/vitals)).
- **Russell's 2026 budgets** for a 3-second first load on the P75 device:
  - JS-light (markup-centric) pages: ~0.3 MiB of JS.
  - JS-heavy pages: ~0.62 MiB.
  - Source: [Russell](https://infrequently.org/2025/11/performance-inequality-gap-2026/).
- **The widget is a guest inside someone else's budget,** so its own budget must be two orders of magnitude smaller (§7.9).

### 2.7 Information design for data

- **Encodings.**
  - Length and 2D position are the preattentive encodings to use for quantities on dashboards ([NN/g video](https://www.nngroup.com/videos/data-visualizations-dashboards/)).
  - Quantile dot plots cut the variance of non-experts' probability estimates by about 1.15× versus density plots ([Kay et al. 2016](https://idl.uw.edu/papers/when-ish-is-my-bus)) and led to better decisions ([Fernandes et al. 2018](https://mucollective.northwestern.edu/project/uncertainty-bus)).
- **Error bars.**
  - Hypothetical outcome plots beat error bars and violin plots for judgements across two or three quantities ([Hullman et al. 2015](https://idl.uw.edu/papers/hops)).
  - Error bars are often misread as the probability of the estimate ([Kale et al. 2018](https://mucollective.northwestern.edu/files/2018-HOPsTrends-InfoVis.pdf), cited via the HOPs project page).
- **Uncertainty and trust.** Stating uncertainty numerically lowered public trust only slightly, and mostly when it was stated verbally ([van der Bles et al. PNAS 2020](https://research.rug.nl/en/publications/the-effects-of-communicating-uncertainty-on-public-trust-in-facts/); review in [RSOS 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6549952/)). **So say "between −2% and +9%", not "the result is uncertain".**

### 2.8 Copy and UX writing [judgement, with sources]

- Lead with the answer.
- Use the merchant's words: "orders", not "events"; "shoppers", not "users".
- Put numbers before adjectives, and use one idea per sentence.
- Vague source labels such as Amazon's "and more" confused users ([NN/g 2018](https://www.nngroup.com/articles/recommendation-guidelines/)).
- Indian number formatting is built in: `Intl.NumberFormat('en-IN')` renders ₹12,34,567, and compact notation renders "12L" [measured with Node]. The long compact form produced "12 million", so avoid `compactDisplay: "long"` for en-IN.

### 2.9 Designing explanations and trust in recommenders

- **Seven explanation aims.** Tintarev and Masthoff list transparency, scrutability, trust, effectiveness, persuasiveness, efficiency and satisfaction. These can conflict: persuasion is not the same as helping someone judge fit ([Tintarev & Masthoff, UMUAI 2012](https://doi.org/10.1007/s11257-011-9117-5), [secondary] via [this review](https://arxiv.org/pdf/2006.08672)). **Mitate optimises for effectiveness and transparency**, not persuasion.
- **Social explanations.** "X friends like this" style explanations shifted willingness to try an item. There was "little correlation" between that and the rating after trying ([Sharma & Cosley, WWW 2013](https://arxiv.org/abs/1304.3405)). Social proof moves clicks more than it predicts satisfaction.
- **Kouki et al. (TiiS 2019, ~198 last.fm users)** ([PDF](https://linqs.org/assets/resources/kouki-tiis20.pdf)):
  - People "prefer item-centric but not user-centric or socio-centric explanations".
  - At most 3–4 explanation styles (mean preferred 4.32).
  - "Textual explanations are better" than Venn diagrams or other visual formats in static recommendation contexts.
  - Novices liked popularity-based and content-based explanations more.
- **Tag clouds.** Content-based tag-cloud explanations raised perceived transparency and satisfaction at a cognitive cost ([Gedikli, Jannach & Ge, IJHCS 2014](https://doi.org/10.1016/j.ijhcs.2013.12.007)).
- **Explanations in production (JD.com, phone category, users who hovered the list)** ([Zhang et al. SIGIR 2014](https://www.cs.cmu.edu/~glai1/papers/yongfeng-guokun-sigir14.pdf)):
  - CTR was 4.34% with specific feature-level explanations, 3.22% with "People also viewed" and 3.20% with no explanation (p = 0.033 and 0.041 vs the other two).
  - Explanations on the recommend/disrecommend indicator raised add-to-cart (p = 0.037).
  - Caveats: 2014, one category, and only hovering users were counted.
- **Herlocker et al. (CSCW 2000)** is widely cited for a histogram-of-neighbours explanation being most persuasive. [unverified]: I could not access the full text.

---

## 3. Tools and techniques for a no-build, vanilla stack served by FastAPI

### 3.1 Baseline status of the CSS and DOM features that matter [measured from `web-features` 3.43.0, 2026-10-20]

| Feature | Status | Support (first version) | Use in Mitate |
|---|---|---|---|
| Container queries (size) | **Widely** (since 2025-08) | Chrome 105, Firefox 110, Safari 16 | Widget layout by slot width; dashboard panels |
| Container style queries | Newly (2026-05) | Chrome 111, Firefox 151, Safari 18 | Optional theme/density hooks |
| `:has()` | **Widely** (2026-06) | Chrome 105, Firefox 121, Safari 15.4 | State styling without JS |
| Cascade layers | **Widely** | Chrome 99, Firefox 97, Safari 15.4 | `@layer reset, tokens, base, components, utilities` |
| Nesting | **Widely** (2026-06) | Chrome 120, Firefox 117, Safari 17.2 | Authoring without a preprocessor |
| `oklch()`, `color-mix()` | **Widely** (2025-11) | Chrome 111, Firefox 113, Safari 15.4/16.2 | All tokens |
| Relative colour syntax | Newly (2024-09) | Chrome 125, Firefox 128, Safari 18 | Hover/pressed variants; fallback to fixed token |
| `light-dark()` | Newly (2024-05) | Chrome 123, Firefox 120, Safari 17.5 | Dashboard/demo tokens (keep `[data-theme]` override) |
| `@property` | Newly (2024-07) | Chrome 85, Firefox 128, Safari 16.4 | Animatable custom props (seal rotation, progress) |
| `@starting-style`, `transition-behavior` | Newly (2024-08) | Chrome 117, Firefox 129, Safari 17.4–17.5 | Enter transitions for popovers, disclosures, rail items |
| `linear()` easing | **Widely** (2026-06) | Chrome 113, Firefox 112, Safari 17.2 | Spring token |
| Popover API | Newly (2025-01) | Chrome 116, Firefox 125, Safari 17 / iOS 18.3 | "Why?" receipt on desktop, menus |
| Invoker commands (`command`/`commandfor`) | Newly (2025-12) | Chrome 135, Firefox 144, Safari 26.2 | Use with a 10-line JS fallback (iOS < 26.2 still common) |
| Interest invokers | **Not Baseline** | Chromium 142 only | Do not use |
| Same-document View Transitions | Newly (2025-10) | Chrome 111, Firefox 144, Safari 18 | Dashboard tab and theme switches |
| Cross-document View Transitions | **Not Baseline** | Chrome 126, Safari 18.2; no Firefox | Demo product-page navigation, as enhancement only |
| Scroll-driven animations | **Not Baseline** | Chrome 115, Safari 26; no Firefox | Rail scroll progress only, as enhancement |
| Anchor positioning | Core APIs Newly (2026-01) per [web.dev](https://web.dev/blog/web-platform-01-2026); the aggregate feature is still "not Baseline" in web-features 3.43.0 because of sub-features | Chrome 125, Firefox 147, Safari 26 | Desktop "Why?" popover placement with a centred fallback |
| `text-wrap: balance` / `pretty` | Newly / not Baseline | balance everywhere; pretty Chrome 117 and Safari 26 | Headings / paragraphs |
| `@scope` | Newly (2026-03) | Chrome 143, Firefox 146, Safari 26.4 | Not needed where Shadow DOM is used |
| `sibling-count()`/`sibling-index()` | Newly (2026-08) | Chrome 138, Firefox 154, Safari 26.2 | CSS-only stagger, with no-stagger fallback |
| `content-visibility` | Newly (2025-09) | Chrome 108, Firefox 130, Safari 26 | Long dashboard tables |
| `contrast-color()` | Newly (2026-04) | Chrome 147, Firefox 146, Safari 26 | Not yet: too new for the widget's audience |
| `interpolate-size` / `calc-size()` | Not Baseline | Chromium only | Height animation of `<details>`, as enhancement only |
| Customizable `<select>` | Not Baseline | Chrome 135, Safari 27 | Do not use |
| Shadow DOM, `::part`, constructed stylesheets | **Widely** | Shadow DOM since Safari 10; constructed sheets since Safari 16.4 | Widget isolation |
| Web Animations API | **Widely** | Chrome 84, Firefox 75, Safari 14 | Choreographed enters; FLIP |
| `<dialog>`, `inert`, exclusive `<details name>` | Widely / Widely / Newly (2024-09) | | Dashboard modals; accordion groups |

Interop 2026 focuses on anchor positioning, view transitions (including cross-document) and scroll-driven animations. Expect these to firm up during 2026–27, but do not depend on them yet ([WebKit](https://webkit.org/blog/17818/announcing-interop-2026/), [Mozilla](https://hacks.mozilla.org/2026/02/launching-interop-2026/)).

### 3.2 JavaScript libraries: what each costs [measured]

Method: fetched from jsDelivr on 2026-10-20 and compressed locally (`gzip -9`; Brotli quality 11 via the `brotli` Python module).

| Library (version) | Min | gzip | br | Verdict |
|---|---|---|---|---|
| Motion (`animate`, mini) | n/a | n/a | n/a | Docs state **3.2 KB mini / 22 KB hybrid** ([motion.dev](https://motion.dev/docs/animate)). The jsDelivr `+esm` build did not tree-shake (46.6 KB gz), so it is not measured here |
| GSAP 3.15.0 core | 72.9 KB | 28.3 KB | 25.7 KB | No. Free including former Club plugins ([Webflow](https://webflow.com/blog/gsap-becomes-free)), but the licence bars competing visual animation tools ([GSAP licence](https://gsap.com/community/standard-license/)) and the payload is overkill |
| Lenis 1.3 | 18.8 KB | 5.4 KB | 4.9 KB | No. It is smooth-scroll/scroll-jacking; see NN/g ([2023](https://www.nngroup.com/articles/scrolljacking-101/)) |
| uPlot 1.6.32 (+ CSS) | 51.1 KB | 22.0 KB (+0.8) | 19.8 KB (+0.6) | **Yes**, dashboard time series |
| Observable Plot 0.6.17 + d3 7 (UMD) | 209 KB + 280 KB | 68.7 + 92.5 KB | 58.8 + 77.7 KB | No: ~136 KB br is too heavy for this dashboard |
| Preact 10 | 11.3 KB | 4.8 KB | 4.4 KB | Optional |
| htm/preact standalone 3.1 | 13.2 KB | 5.3 KB | 4.9 KB | **Optional** for the dashboard if state grows; no build needed |
| Lit core 3.3.3 | 15.7 KB | 6.1 KB | 5.5 KB | No: the widget is one component and does not need reactive templating |
| Alpine 3 | 55.9 KB | 19.9 KB | 18.1 KB | No |
| petite-vue 0.4 | 16.9 KB | 7.1 KB | 6.5 KB | No |
| lottie-web light 5 | 168 KB | 46.5 KB | 40.3 KB | No: there is no motion designer in the loop, and SVG + CSS covers the needs |
| Rive canvas-lite | | | 222 KB br incl. WASM per [Rive](https://rive.app/docs/runtimes/runtime-sizes.md) | No |
| Floating UI dom + core 1.x | 22.6 KB | 8.8 KB | 8.0 KB | Only if anchor-positioning fallback proves insufficient |

### 3.3 Widget isolation and host-theme adaptation (Shadow DOM)

**What crosses the shadow boundary.** Inherited properties (`font`, `color`, `line-height`, `letter-spacing`, `direction`) and custom properties cross into a shadow root. Authors expose structure through `part` attributes, styled from outside with `::part()` ([CSS Shadow Parts](https://drafts.csswg.org/css-shadow-parts), [CSS-Tricks](https://css-tricks.com/?p=305838)). This is what Mitate needs: the host's font and colour arrive automatically, and nothing in the host's `ul li` or `a` rules can break the widget.

**Recommended contract [judgement]:**

- **Shadow root.** Open shadow root on each `[data-mitate]` host. One constructed stylesheet shared by all roots via `adoptedStyleSheets` (Widely available).
- **`:host` styles.** `display:block; container-type:inline-size; color:inherit; font:inherit`. Card radius and spacing scale in `em`, so they follow the host's type size.
- **Public tokens.** Defaults are derived from `currentColor`:
  - `--mitate-accent` (default `currentColor`).
  - `--mitate-muted` (default `color-mix(in oklab, currentColor 75%, transparent)`).
  - `--mitate-rule` (default `color-mix(in oklab, currentColor 14%, transparent)`).
  - `--mitate-radius` (default `0.5em`) and `--mitate-gap` (default `0.75em`).
  - `--mitate-surface` (default `transparent`).
- **Public parts.** `rail`, `heading`, `card`, `media`, `title`, `price`, `reason`, `why`, `featured`, `nav`. Document them as the styling API.
- **Heading level.** Merchant-settable via `data-heading-level`, default `h2`. The current hard-coded `h3` can break the host's outline.
- **No interactive nesting.** Use a "stretched link" on the title (an `::after` covering the card). The "Why?" control sits outside the link, because a button inside `<a>` is invalid and confuses screen readers.

### 3.4 Font strategy without third-party requests

Self-host WOFF2 from the FastAPI static mount for the dashboard and demo:

- `font-display: swap`.
- `size-adjust` / `ascent-override` on a local fallback to minimise layout shift. `font-size-adjust` is Newly available as of 2024-07.
- `unicode-range` splits.
- Preload only the one text face.

The widget ships **no** fonts.

### 3.5 Font candidates: measured [measured]

Method:

- **Source and tools.** Google Fonts repo TTFs, subset with fontTools 4.67.0.
- **Glyph set.** Basic Latin, Latin-1, General Punctuation, ₹ U+20B9, € and arrows.
- **Processing.** All OpenType layout features kept, hinting removed, WOFF2 output.
- **Axis pinning.** Applied with `varLib.instancer` where stated.

| Face (licence) | Axes shipped | WOFF2 | ₹ | `tnum` | Notes |
|---|---|---|---|---|---|
| **Gloock** (OFL, Duarte Pinto) | static | **16.5 KB** | yes | yes | "contemporary high-contrast serif … inspired by newspaper headlines" ([GF](https://github.com/google/fonts/tree/main/ofl/gloock)) |
| **Anek Latin** (OFL, Ek Type, Mumbai) | wght 400–700, wdth 100 | **38.3 KB** | yes | yes | Anek covers Latin + 9 Indic scripts; "a … refreshing interpretation of India's letter traditions" ([GF](https://github.com/google/fonts/tree/main/ofl/aneklatin)) |
| Anek Latin (display cut) | wght 700, wdth 75–100 | 33.7 KB | yes | yes | Width axis for condensing headings in narrow containers |
| Anek Latin (full) | wght 300–800, wdth 75–125 | 98.2 KB | yes | yes | Too heavy |
| **IBM Plex Mono** Regular / Medium (OFL) | static | **9.9 / 9.8 KB** | yes | mono | Ledger figures, codes, x-ray |
| Geist / Geist Mono (OFL) | wght 400–600 | 29.1 / 23.3 KB | yes | yes / mono | Good, but strongly associated with Vercel (§1.3) |
| Fraunces (OFL) | opsz 144, SOFT 0, WONK 0, wght 300–700 | 32.5 KB | yes | no | Alternative display serif; very common 2021–24 |
| Newsreader (OFL) | opsz 72, wght 300–700 | 55.3 KB | yes | yes | Alternative if long-form reading is ever needed |
| Bricolage Grotesque (OFL) | opsz/wdth pinned, wght 300–700 | 86.9 KB | yes | yes | Heavy |
| Mona Sans (OFL) | wdth 75–125, wght 300–700 | 115.7 KB | yes | yes | Heavy |
| Recursive (OFL) | MONO 0–1, CASL 0, wght 300–800 | 74.5 KB | yes | no | One file covering sans + mono; considered and not chosen |
| Fragment Mono (OFL) | static | 16.6 KB | yes | mono | Alternative mono |
| JetBrains Mono, Martian Mono, Red Hat Mono, Spline Sans Mono | | 21.9–39.1 KB | **no** | | Ruled out: no ₹ |
| Instrument Serif, Hanken Grotesk | | 14.4 KB / n/a | **no** | | Ruled out: no ₹ |
| Shippori Mincho / Zen Old Mincho (OFL), subset to `見立て` only | static | **1.1 KB** | | | Brand mark glyphs; or ship the mark as SVG paths (0 font bytes) |
| Anek Devanagari, Devanagari block only | wght 300–700, wdth 100 | 224 KB | | | **Too heavy**: use system Devanagari (Noto on Android, Kohinoor on iOS) via the font stack |

### 3.6 Muted-text contrast inside an unknown host [measured]

WCAG 2 contrast was computed for `currentColor` mixed with transparent over the host background.

| Host text on background | Base | 60% (current widget) | 70% | **75%** | 80% |
|---|---|---|---|---|---|
| `#111` on `#fff` | 18.88 | 4.94 | 7.07 | 8.52 | 10.27 |
| `#222` on `#fff` | 15.91 | **4.27** | 5.89 | 6.98 | 8.29 |
| `#333` on `#fff` | 12.63 | **3.71** | 4.94 | 5.74 | 6.71 |
| `#3a3a3a` on `#f7f4ee` | 10.36 | **3.38** | **4.39** | 5.03 | 5.80 |
| `#eee` on `#111` | 16.28 | 6.35 | 8.29 | 9.39 | 10.58 |
| `#ddd` on `#1e1e1e` | 12.27 | 5.26 | 6.66 | 7.45 | 8.29 |

**Change the widget default to 75%.** The reason line is the product's differentiator, so it must never be the faintest text on the page.

### 3.7 Generated product imagery (SVG)

- **Rendering cost.** SVG filters are costly on mobile and turbulence cost grows with `numOctaves` ([MDN-derived docs](https://docs.w3cub.com/svg/reference/attribute/numoctaves)). Animated `feTurbulence` is "pretty horrific for performance" on iOS per practitioners ([GreenSock forum](https://greensock.com/forums/topic/33075-gsap-and-feturbulence-mobile-performance/), [secondary]).
- **Texture approach.** Build print texture from `<pattern>` halftone dots and layer offsets, not filters (§7.7).

---

## 4. Reference products: what each does well, and what to take

| Reference | What it does well (sourced) | Take for Mitate | Leave |
|---|---|---|---|
| Linear | LCH theme from 3 inputs; Inter Display for headings; 2026 "calmer" refresh with dimmer sidebar, softer separators, warmer grey; "Structure should be felt not seen" ([2024](https://linear.app/now/how-we-redesigned-the-linear-ui), [2026](https://linear.app/now/behind-the-latest-design-refresh)) | Generate themes from few inputs; let navigation recede | The look itself (§1.3) |
| Vercel / Geist | Geist Sans + Mono under OFL; Swiss-inspired "simplicity, minimalism, and speed" ([Vercel font](https://vercel.com/font)) | A type family that spans UI and mono | Geist itself (association) |
| Raycast | No animation in a keyboard-driven tool, which Kowalski cites as correct ([Kowalski](https://emilkowal.ski/ui/great-animations)) | No motion on keyboard flows (pairing review, rules) | |
| Emil Kowalski (Sonner, Vaul) | Interruptible transitions, hover-gap fill, velocity-based dismissal, timers paused when hidden ([toast write-up](https://emilkowal.ski/ui/building-a-toast-component)) | Toasts and sheets in the dashboard | |
| Rauno Freiberg | Interruptibility, spatial consistency, frequency-aware motion ([Every](https://every.to/p/invisible-details-of-interaction-design)) | X-ray reveal originates from the rail it explains | |
| Paco Coursey / cmdk | Unstyled, accessible command-menu primitive ([cmdk](https://github.com/pacocoursey/cmdk)) | A `⌘K`/`/` jump-to for the dashboard (later) | |
| Stripe | Perceptual colour tooling for accessible palettes ([Stripe](https://stripe.com/blog/accessible-color-systems)) | Contrast computed, never eyeballed | |
| Shopify Polaris (2025) | Unified, framework-agnostic web components across Admin, Checkout and POS; stable since 2025-10 ([Shopify](https://shopify.dev/changelog/polaris-unified-web-components-are-now-stable)) | If Mitate becomes an embedded Shopify app, App Home should use Polaris web components; the stand-alone dashboard should feel as calm | Polaris styling in the stand-alone dashboard |
| Teenage Engineering | Constraints as aesthetic; monospace as "precision and technical honesty" ([guide](https://blakecrosley.com/es/guides/design/teenage-engineering), [secondary]) | Mono for counted facts | Orange-on-grey pastiche |
| Nothing | One red accent used as a rare signal ([analysis](https://madegooddesigns.com/nothing-brand-guidelines/), [secondary]) | The vermilion seal as the *only* saturated colour | Dot-matrix type |
| Family (Benji Taylor) | "Simplicity → fluidity → delight", in that order ([secondary summary](https://skills.sh/cristicretu/family-taste-skill/design-with-taste); original at benji.org/family-values, [unverified]) | Order of work: hierarchy first, motion last | |
| Airbnb 2025 | "Dimensional and beautifully animated interface" with new 3D iconography ([TechBuzz](https://techbuzzireland.com/2025/05/14/redesigned-airbnb-app-now-available/), [secondary]); read by commentators as a skeuomorphic revival ([HospitalityNet](https://www.hospitalitynet.org/opinion/4127228.html)) | Tactile, crafted object imagery, via the still-life plates of §7.7 | 3D rendering cost |
| Gumroad | Committed neo-brutalism against generic SaaS styling ([teardown](https://marketingonmonday.substack.com/p/marketing-teardown-gumroad), [secondary]) | Commit fully to one idea | Neo-brutalism (now templated) |
| Are.na | No ads, no likes, no algorithmic recommendations ([Are.na](https://www.are.na/)) | A calm, collector's tone; dignity for objects | |
| Awwwards SOTY 2025 | Listings show "Messenger" (abeto) and "Lando Norris" (OFF+BRAND) with SOTY badges ([Awwwards](https://www.awwwards.com/websites/%23e2de00/?page=5)); the official announcement is [unverified] | Choreography restraint | WebGL and scroll spectacle on product pages |
| Indian D2C (§1.5) | Strong display type and brand voice | Confident type scale | Heavy pages; 100+ scripts |

**Recommendation interfaces:**

- **Amazon.**
  - "View or edit browsing history" placed next to recommendations ([NN/g 2018](https://www.nngroup.com/articles/recommendation-guidelines/)).
  - "Bought in past month" counts are rounded and capped (e.g. "100K+") ([Marketplace Pulse, 17 Mar 2023](https://www.marketplacepulse.com/articles/amazon-made-sales-data-public)).
  - Mitate should likewise show coarse counts when they are large (e.g. "1,200+").
- **Netflix.** Row-level "Because you watched" labels help users judge relevance. Removing an item from history shows that it will "no longer be used to make recommendations" ([NN/g 2018](https://www.nngroup.com/articles/recommendation-guidelines/)).
- **Spotify.** Users preferred genre-coherent Daily Mixes over the more varied Discover Weekly ([NN/g 2018](https://www.nngroup.com/articles/recommendation-guidelines/)). This argues for one clear relation per rail ("Goes well with", "Similar"), never a mixed rail.
- **Pinterest.** A rejected item was replaced at once by a placeholder, with no reflow ([NN/g 2018](https://www.nngroup.com/articles/recommendation-guidelines/)).
- **TikTok.** "Why this video" (Dec 2022) lists broad factors. Critics called it too coarse ([Axios](https://www.axios.com/2022/12/20/tiktok-explain-video-recommendations-algorithm), [Popular Science](https://www.popsci.com/technology/tiktok-why-this-video/)). Mitate's per-item counted facts are more specific.
- **Myntra.** It has published research on personalised similar-item and shop-the-look retrieval ([arXiv 2008.11638](https://arxiv.org/abs/2008.11638)). The UI details are [unverified].
- **Meesho.** Simplicity for first-time online shoppers, and vernacular copy written for scanning ([Meesho tech](https://meesho.io/tech), [design blog](https://meesho.io/blog/my-first-6-months-as-a-product-designer-at-meesho)).

---

## 5. Recommendation and e-commerce UX evidence

### 5.1 Baymard

- **Product-page cross-sell content.** 68% of desktop sites in Baymard's benchmark give too little information in product-page cross-sells ([Baymard, 23 Mar 2021](https://baymard.com/research-articles/product-page-suggestions-information)).
  - **Essential on every cross-sell item:** a high-quality thumbnail (8% lack one), a fully visible title (55% fail), price (15% lack it) and rating with count (50% lack it).
  - **Show when warranted:** variations and brand.
  - **Mobile** needs the same attributes as desktop.
  - **Implication for Mitate:** no title truncation by default (wrap up to three lines), price always shown, and an image slot once catalogs supply images. Ratings are out of scope until the merchant supplies them.
- **Cart cross-sell relevance.** 52% of desktop sites show irrelevant cart cross-sells, or ones driven only by "what other customers bought". One bad suggestion makes users ignore the whole section ([Baymard, 12 Jan 2021](https://baymard.com/research-articles/product-recommendations-cart)).
  - **Six fixes:** no fixed count, caution with alternatives, clear context labels, same use-case/theme, compatibility first, context-aware offers.
  - **Implication for Mitate:** a rail renders as many items as pass the relevance and rules bar. Zero means the rail is hidden.
- **Checkout cross-sells.** Injected checkout cross-sell steps provoked extreme frustration in 66% of the users who hit Amazon's step ([Baymard](https://baymard.com/checkout-usability/benchmark/step-type/cross-sell)). **Mitate never renders in checkout.**
- **Mobile carousels.** Partially visible items signal more content. Pairing this with visible controls helps first-time and older users ([Baymard thumbnails](https://baymard.com/research-articles/truncating-product-gallery-thumbnails), [page control](https://baymard.com/learn/page-control-ui), [secondary] via search snippets).
- **Hover.** Information that appears only on hover adds effort (product-page attributes article above). **Reasons are visible, not hover-only.**

### 5.2 NN/g on recommended content ([Harley, 4 Nov 2018](https://www.nngroup.com/articles/recommendation-guidelines/))

- **Be specific about the data used.** For example "Based on your past history", never "and more".
- **Keep groupings specific.**
- **Offer feedback and edit controls,** and reflect feedback immediately.
- **Placement.** Users missed personalised sections that sat below generic promotions.

### 5.3 What Mitate must never do (fabrication, dark patterns)

- **India.** The CCPA's Guidelines for Prevention and Regulation of Dark Patterns (30 Nov 2023) list 13 patterns ([Mondaq](https://www.mondaq.com/ccpas-guidelines-on-dark-patterns-an-overview/1844140)). They include false urgency (invented scarcity *or popularity*), basket sneaking, confirm-shaming, interface interference, disguised advertisement and nagging.
  - A June 2025 advisory asked platforms to self-audit within 3 months ([Storyboard18](https://stgwww.storyboard18.com/digital/ccpa-warns-e-comm-firms-over-dark-patterns-self-audit-within-3-months-69249.htm)).
  - CCPA has since penalised seven platforms ([Bar & Bench](https://www.barandbench.com/law-firms/view-point/from-nudge-to-notice-what-indias-dark-pattern-orders-mean-for-digital-business), [secondary]).
- **US.** The FTC's rule (16 CFR 465, effective 21 Oct 2024) bans fake reviews and fake social-media influence indicators ([Hogan Lovells](https://www.hoganlovells.com/en/publications/ftc-publishes-final-rule-banning-fake-consumer-reviews-and-testimonials)).
- **Rules for the UI [judgement, aligned with the engine]:**
  1. Every displayed count comes from `Evidence` and is backed by at least `min_support` (default 5) distinct shoppers.
  2. No timers, "only N left" or "N people viewing now" in any Mitate surface, including the demo.
  3. Merchant pins are labelled "Featured by the store", because CCPA treats ads disguised as organic content as a dark pattern.
  4. The consent prompt has two neutral, equal-weight buttons with no confirm-shaming copy.
  5. The "Why?" receipt states its window and scope: "this store's orders, last 90 days". Use the engine's real window.

---

## 6. Dashboard and data UX for non-technical small merchants

- **Progressive disclosure, two levels maximum** ([Nielsen 2006](https://www.nngroup.com/articles/progressive-disclosure/)).
  - Level 1 holds the sentence and number a founder needs: "Mitate is learning from 1,284 orders. 'Goes well with' now uses your shoppers' behaviour."
  - Level 2, behind an explicit "Details", holds model versions, source lists, fit times and raw parameters.
- **Empty states** ([Kaplan 2021](https://www.nngroup.com/articles/empty-state-interface-design/)) do three jobs:
  - Say the status: loading vs nothing yet vs error. Never show "No records" while still loading.
  - Teach in context.
  - Offer the direct action: "Import your products (Shopify → Products → Export)".
- **Stage honesty.**
  - The engine's real states are `untrained`, `catalog_only` and `behavioural`, plus per-source activation thresholds (`mitate/engine.py`, `serve/service.py`).
  - The dashboard shows exactly these, in merchant words, with the true count to the next threshold.
  - Progress is never smoothed or estimated, and there are no fake progress bars.
- **"Your store is too small to test this".**
  - Eppo, GrowthBook and Statsig all document MDE and power ([Eppo](https://docs.geteppo.com/statistics/sample-size-calculator/mde/), [GrowthBook](https://docs.growthbook.io/statistics/power), [Statsig](https://docs.statsig.com/en/experiments/power-analysis)). None of their docs describe an in-product "underpowered" verdict ([unverified] beyond docs).
  - That gap is Mitate's chance to do it well:
    1. Lead with the verdict.
    2. Then give the weeks needed as a range.
    3. Then the alternative: interleaving needs less traffic but measures preference, not revenue.
  - Interleaving's often-quoted 10–100× sensitivity gains come only from large platforms ([DoorDash](https://careers.doordash.com/blog/doordash-experimentation-with-interleaving-designs), [secondary]). The repo already records that limit as decision D13 in `docs/research_ledger.md`. The UI must repeat it, not hide it.
- **Showing uncertainty.**
  - Use an interval bar on a shared axis with a zero line, plus a sentence.
  - For "how long would a test take?", show a quantile dot plot of 20 dots, each a plausible duration (§2.7).
  - Keep p-values and posterior probabilities at level 2.
- **Charts.**
  - uPlot for time series (events per day, orders, model refits).
  - Plain HTML/CSS bars for everything categorical. Length encodes quantity, per §2.7.
  - No pie or donut charts, no 3D, no gauges.

---

## 7. Design brief

### 7.1 Direction: "Hakogaki" (箱書き)

**Idea.**

- **Mitate.** In tea practice, *mitate* means seeing an ordinary object anew and choosing it for a role, for example a water gourd chosen as a flower vase ([Osaka art-flavor](https://art-flavor.osaka-info.jp/column/mitate-adapting-heirlooms-as-tableware/), [secondary]).
- **Hakogaki.** A tea object's storage box carries an inscription recording its name, maker and who vouched for it ([Toledo Museum, authenticating box](https://emuseum.toledomuseum.org/objects/57262/authenticating-box-tomobako); dealer notes on [hakogaki](https://turuta.jp/ceramics-story/?p=25009), [secondary]).
- **The mapping.** The rail is the *mitate*, and the reason is the *hakogaki*: a short, factual inscription that names its evidence.

**Visual language [judgement]:**

- **Paper and ink.** An unbleached paper ground (*kinari*), warm ink (*sumi*), and hairline rules like a ledger or a box lid's grain. In dark mode the paper becomes an ink-stone ground and the text becomes paper.
- **One seal.** Vermilion (*shu*) is the only saturated colour. It appears as a small seal that marks the "chosen" moment: the brand mark, an approved pairing, the current stage. It never decorates and never signals an error.
- **Three voices of type.**
  - A high-contrast headline serif (Gloock) for the voice of the curator: names, big numerals.
  - An Indian-made grotesque (Anek Latin) for the interface.
  - A mono (IBM Plex Mono) for the inscription: counted facts, windows, versions, latencies.
- **Objects as still lifes.** Generated SVG plates for the synthetic catalog (§7.7).
- **The Indian bridge.** The rubber stamp on a shop's bill book is as familiar in Indian commerce as the hanko is in Japan. The seal reads as "verified, paid, chosen" in both cultures without kitsch.

**Why it fits Gen Z (§1).** It is specific instead of glossy and it shows its work through receipts and the x-ray. It has one tactile motif used with intent, the seal, and confident editorial type. It is curated in Pinterest's "intentional curation" sense, and it is fast on a ₹15k phone.

**How it avoids slop.**

- Everything traces back to Mitate's own idea.
- No Inter, Space Grotesk or Geist.
- No gradients, no glass and no neon accents.
- No bento KPI grid and no emoji.
- **No orientalist decoration.** No waves, cherry blossoms, faux-brush fonts or scattered kanji. The only Japanese text is `見立て` in the brand mark.

### 7.2 Type

**Families** (OFL; sizes from §3.5):

| Role | Face | File(s) | Bytes | Where |
|---|---|---|---|---|
| Display / numerals | Gloock Regular | `gloock-latin.woff2` | 16.5 KB | Dashboard + demo, ≥ 28 px only |
| UI / text | Anek Latin, wght 400–700, wdth 100 | `anek-latin-text.woff2` | 38.3 KB | Dashboard + demo (preload) |
| Display condensing | Anek Latin, wght 700, wdth 75–100 | `anek-latin-display.woff2` | 33.7 KB | Demo only (product names in narrow containers) |
| Ledger | IBM Plex Mono 400 + 500 | `plex-mono-{400,500}-latin.woff2` | 9.9 + 9.8 KB | Dashboard + demo x-ray |
| Brand glyphs | `見立て` from Shippori Mincho, or SVG paths | `mitate-mark.woff2` / inline SVG | 1.1 KB / ~1 KB | Brand mark only |
| Indic scripts | System (`"Noto Sans Devanagari", "Kohinoor Devanagari", system-ui`) | none | 0 | Vernacular copy |
| Widget | `font: inherit` | none | 0 | Host's font |

**Per-surface font totals:** dashboard ≈ 75 KB; demo ≈ 109 KB; widget 0.

**Scale** (rem at 16 px root; fluid steps use `clamp()`):

| Token | Size | Line height | Face / weight | Use |
|---|---|---|---|---|
| `--t-micro` | 0.75rem (12) | 1.35 | Plex Mono 500, +0.02em, uppercase | Seal labels, table heads (never below 12 px) |
| `--t-small` | 0.875rem (14) | 1.45 | Anek 400 | Meta, reason line in dashboard |
| `--t-body` | 1rem (16) | 1.5 | Anek 400 | Body. Dashboard moves up from today's 15 px |
| `--t-lead` | 1.25rem (20) | 1.4 | Anek 500 | Answer-first sentences |
| `--t-h3` | clamp(1.375rem, 1.2rem + 0.8vw, 1.625rem) | 1.25 | Anek 600 | Card/section titles |
| `--t-h2` | clamp(1.75rem, 1.4rem + 1.6vw, 2.25rem) | 1.15 | Gloock | Page titles |
| `--t-h1` | clamp(2.25rem, 1.6rem + 3vw, 3.5rem) | 1.05 | Gloock, −0.01em | Demo product name, dashboard home headline |
| `--t-numeral` | clamp(3rem, 2rem + 5vw, 6rem) | 0.95 | Gloock, `tnum` | The one big number per view (orders learned from, weeks needed) |

**Rules:**

- `font-variant-numeric: tabular-nums` on every price and count.
- `text-wrap: balance` on h1–h3 and `pretty` on paragraphs.
- Max measure 66ch.
- The ₹ sign is never synthesised: every face above has it.
- **Kinetic type with a job.** Demo product names switch to the condensed cut via `@container (inline-size < 26rem) { font-stretch: 85% }` instead of wrapping into four lines. Do not animate width on scroll.

### 7.3 Colour tokens (OKLCH, hex fallback) [measured contrast]

Contrast is computed against `--paper`. APCA uses the 0.0.98G formula, validated against reference values: black on white Lc 106.04, white on black −107.88, `#888` on white 63.06.

**Light ("kinari")**

| Token | OKLCH | Hex | WCAG vs paper | APCA Lc | Use |
|---|---|---|---|---|---|
| `--paper` | `oklch(97.5% 0.010 85)` | `#faf6ef` | n/a | n/a | Page |
| `--raised` | `oklch(99.3% 0.005 85)` | `#fefdf9` | n/a | n/a | Cards, popovers |
| `--sunk` | `oklch(94.5% 0.014 82)` | `#f2ece3` | n/a | n/a | Wells, table stripes, inputs |
| `--ink` | `oklch(23.5% 0.014 60)` | `#231d18` | 15.54 | 98.6 | Text |
| `--ink-2` | `oklch(43% 0.016 62)` | `#564e47` | 7.57 | 83.2 | Secondary text |
| `--ink-3` | `oklch(52% 0.015 66)` | `#6f6760` | 5.14 (4.70 on sunk) | 72.6 | Meta, placeholders |
| `--rule` | `oklch(88.5% 0.013 80)` | `#ded8d0` | 1.32 | n/a | Hairlines (decorative only) |
| `--rule-strong` | `oklch(62% 0.014 70)` | `#8c857d` | 3.39 | 59.1 | Input borders, focus-adjacent UI (≥ 3:1 non-text) |
| `--shu` | `oklch(54.5% 0.185 34)` | `#c43512` | 5.04 (4.61 on sunk) | 70.6 | Seal, brand mark, primary action fill |
| `--on-shu` | `oklch(98.5% 0.010 85)` | `#fdfaf3` | 5.19 on shu | −77.7 | Text on seal / primary button |
| `--ai` (indigo) | `oklch(44% 0.120 262)` | `#2c4f94` | 7.34 | 82.1 | Links, data series 1 |
| `--ok` | `oklch(50% 0.110 152)` | `#267543` | 5.30 | 73.2 | Success text/icons |
| `--warn` | `oklch(52% 0.110 75)` | `#8d5e00` | 5.23 | 72.8 | Warning text |
| `--warn-fill` (haldi) | `oklch(83% 0.130 82)` | `#f1bf5b` | ink on it 9.80 | 71.0 | Warning chips (ink text) |
| `--danger` | `oklch(52% 0.180 18)` | `#ba2840` | 5.64 | 73.8 | Errors, always with icon + text |
| `--focus` | `oklch(50% 0.170 262)` | `#285cc2` | 5.75 | 75.3 | 2 px focus ring + 2 px paper offset |

**Dark ("sumi")**

| Token | OKLCH | Hex | WCAG vs paper | APCA Lc |
|---|---|---|---|---|
| `--paper` | `oklch(17.5% 0.008 60)` | `#13100d` | n/a | n/a |
| `--raised` | `oklch(21.5% 0.009 60)` | `#1d1915` | n/a | n/a |
| `--sunk` | `oklch(14.5% 0.007 60)` | `#0c0907` | n/a | n/a |
| `--ink` | `oklch(93.5% 0.011 85)` | `#ede9e1` | 15.68 | −93.3 |
| `--ink-2` | `oklch(82% 0.012 80)` | `#c8c3bc` | 10.87 | −70.4 |
| `--ink-3` | `oklch(74% 0.012 78)` | `#afaaa3` | 8.22 | −56.1 |
| `--rule` | `oklch(30% 0.010 65)` | `#312d29` | 1.39 | n/a |
| `--rule-strong` | `oklch(52% 0.012 70)` | `#6e6862` | 3.44 | n/a |
| `--shu` | `oklch(73% 0.160 38)` | `#fa7f59` | 7.47 | −52.2 |
| `--on-shu` | `oklch(18% 0.030 40)` | `#1d0d07` | 7.44 on shu | 54.0 |
| `--ai` | `oklch(78% 0.095 262)` | `#96b8f5` | 9.45 | −63.0 |
| `--ok` | `oklch(77% 0.120 152)` | `#75ca8e` | 9.60 | −63.9 |
| `--warn` | `oklch(80% 0.110 80)` | `#e3b667` | 10.07 | −66.4 |
| `--danger` | `oklch(75% 0.140 18)` | `#fa878c` | 8.02 | −55.3 |
| `--focus` | `oklch(78% 0.110 262)` | `#91b7fe` | 9.44 | −63.0 |

**Rules.**

- All tokens are in sRGB gamut, so no P3 clipping surprises.
- `--shu` and `--danger` are close in hue (34° vs 18°), so `--shu` is never used for status, and danger always carries an icon and words.
- Shadows in dark mode become a lighter `--raised` surface plus a `--rule` border.
- Data series beyond `--ai` use ink tints, plus direct labels instead of legends.

**Demo store "host skins"** are deliberately *not* Mitate's palette, to prove the widget adapts:

- *Brass & teak*: light, warm.
- *Steel thali*: cool grey, high contrast.
- *Night bazaar*: dark.
- *Plain*: browser defaults.

### 7.4 Space, shape, depth, motion

- **Space** (4 px base): `--s-1:2px --s0:4px --s1:8px --s2:12px --s3:16px --s4:24px --s5:32px --s6:48px --s7:64px --s8:96px`. Page gutter 16 px at 390 px wide, 32 px from 768 px, and content max 72rem.
- **Radius.**
  - `--r-hair: 2px` (inputs, chips).
  - `--r-1: 6px` (buttons).
  - `--r-2: 10px` (cards, sheets).
  - Seals are circles or 4 px rounded squares.
  - No pill buttons, except filter chips.
- **Borders.** 1 px `--rule` for structure and 1 px `--rule-strong` for inputs. Rules are preferred to cards: most dashboard groupings are sections separated by hairlines, not boxes.
- **Elevation.**
  - Light: `--lift: 0 1px 0 oklch(20% 0.02 60 / .05), 0 10px 28px -12px oklch(20% 0.02 60 / .22)`. Use it only for popovers, sheets and the x-ray ledger.
  - Dark: `--raised` plus a border.
- **Focus.** `outline: 2px solid var(--focus); outline-offset: 2px`. Never removed, and never hidden under sticky bars (WCAG 2.4.11, via `scroll-padding`).
- **Motion tokens:**

| Token | Value | Use |
|---|---|---|
| `--d-1` | 120 ms | Press, hover colour, toggle |
| `--d-2` | 180 ms | Popover, disclosure, tooltip |
| `--d-3` | 240 ms | Rail enter, x-ray reveal, toasts |
| `--d-4` | 360 ms | Stage seal move, view transitions |
| `--stagger` | 30 ms per item, max 6 items | Rail enter only |
| `--ease-out` | `cubic-bezier(0.22, 1, 0.36, 1)` | Default for enters and feedback |
| `--ease-in-out` | `cubic-bezier(0.65, 0, 0.35, 1)` | Elements that move while on screen |
| `--ease-exit` | `cubic-bezier(0.4, 0, 1, 1)`, at 0.75× the enter duration | Exits |
| `--ease-spring` | `linear(0, 0.08, 0.255, 0.457, 0.644, 0.796, 0.907, 0.98, 1.022, 1.041, 1.046, 1.042, 1.033, 1.024, 1.016, 1.009, 1.004, 1.001, 1)` | Seal stamp, stage move. Damping ratio 0.7, about 4.6% overshoot [measured: sampled analytic damped spring] |

- **Reduced motion.** Under `prefers-reduced-motion: reduce`, all transforms become opacity-only at ≤ 120 ms. Stagger, view transitions and spring overshoot are removed.
- **Keyboard actions** (pairing review, tab changes via arrow keys, rule toggles) **never animate.**
- **Only `transform` and `opacity` animate.** The single exception is the `<details>` height, which uses `interpolate-size` where available and is instant elsewhere.

### 7.5 Component inventory

**Widget (Shadow DOM, light-touch):**

| Component | Notes |
|---|---|
| Rail | `<section aria-labelledby>`; configurable heading level; renders only when items exist (holdout or empty means nothing); item count follows the data (Baymard: no fixed count) |
| Layout | `@container` < 30rem: horizontal scroll-snap rail with the last card peeking ~40%. ≥ 30rem: grid of 3–6 columns, no scroll. ≥ 48rem with more than 6 items: scroll with prev/next buttons (`aria-controls`, 24 px+ targets) |
| Card | Media slot (aspect-ratio box, `width`/`height` set, `loading=lazy`, `decoding=async`) shown only when the API supplies images; title (wraps up to 3 lines, no truncation); price (`tabular-nums`, `Intl` currency); reason line; stretched link on the title |
| Reason line | Reason glyph (inline SVG `currentColor`, one per `reason_code`) plus text. The number sits first and is set in `font-variant-numeric: tabular-nums` (host font) |
| "Why?" disclosure | `<details>` below the card: an inline receipt with the full fact, window, scope and "We show counts only when at least 5 shoppers stand behind them." Zero JS; keyboard native |
| Featured label | "Featured by the store" for `reason_code = featured` (merchant pin) |
| States | Loading renders nothing. Error and empty render nothing (`hidden`). Optional `data-reserve` sets `min-block-size` for above-the-fold placements |
| Telemetry | Click events as today; no impression pixels; no storage without consent (unchanged) |

**Demo storefront:**

| Component | Notes |
|---|---|
| Synthetic-store banner | Keep the existing honest notice, restyled as a ruled strip (not a yellow alarm) |
| Header | Store wordmark (host skin), "Browse as" shopper picker, host-skin switcher, x-ray toggle |
| Home | Hero with an editorial line and three still-life plates; category index; "For you" and "Popular right now" rails |
| Product page | Still-life plate (4:5), name (Gloock/condensed Anek per skin), price, add-to-cart/buy; then rails in order: Goes well with → Similar → Shoppers went on to view |
| Consent bar | Two equal buttons, plain copy, does not cover focus (2.4.11), dismissed state remembered per session only after consent |
| X-ray ledger | Per rail: stage, model version, personalised (yes/no and why: consent/shopper), experiment arm, latency (client fetch ms; plus `Server-Timing` when exposed), per-item `reason_code` and evidence count, rules applied (pinned/blocked/stock/region). Page-level summary as a bottom sheet on mobile |
| Cart toast | Sonner-style, 4 s, pauses on hover and hidden tab |

**Dashboard:**

| Area | Components |
|---|---|
| Shell | Left rail nav (collapses to bottom tab bar under 40rem): Home, Rules, Pairings, Tests, Catalog, Privacy; Settings/Models under "Advanced". Store name + `見立て` mark. Theme toggle (system/light/dark) |
| Home | Answer-first headline; stage track (§7.6); "What changed" feed (retrain, sources switched on, imports); next best action; level-2 "Details" (model registry table) |
| Rules | Grouped forms with live preview ("Customers will no longer see 3 products"); product pickers by name, not raw IDs; inline validation; sticky save bar that doesn't obscure focus |
| Pairings | Review queue: one proposed pair at a time, both category names large, model rationale in mono, Approve (A) / Reject (R) / Skip (S) buttons plus keys; swipe on touch only as an addition (2.5.7); approved list as a ledger table |
| Tests | Experiment list (status, days running, plain verdict); result view (interval bar + sentence; level-2 statistics); create flow with power check inline |
| Power check | Inputs (visitors/day, smallest lift worth detecting, horizon); verdict sentence; quantile dot plot of weeks needed; interleaving alternative with its caveat |
| Catalog | Drag-or-pick upload (with button alternative), format picker with tiny export instructions, import report as a readable list (not `<pre>`) |
| Privacy | Erasure and export flows by shopper ID, confirmation with typed ID, receipts with timestamps |
| Feedback | Toasts (non-blocking), inline errors next to fields, `role="status"` live regions |

### 7.6 Signature interactions

1. **The rail settles (widget).**
   - On first intersection (IntersectionObserver, `rootMargin: 600px`, which also defers the API call), cards fade from 0 to 1 and rise 6 px.
   - Timing: `--d-3`, `--ease-out`, stagger `min(sibling-index(), 6) * 30ms`, with no stagger where unsupported.
   - It runs once per page view, never on refresh after consent changes. Reduced motion gives opacity only.
   - There is no skeleton shimmer, so nothing flashes and nothing shifts.
2. **The inscription (reason reveal).**
   - The reason is always visible. "Why?" expands an inline receipt beneath the card. Its first line repeats the fact in full; the second gives the scope and window; the third gives the threshold rule.
   - In the dashboard and demo the receipt is set in Plex Mono. In the widget it uses the host font with tabular numbers.
   - The disclosure height animates only where `interpolate-size` exists. Elsewhere it is instant, which is fine.
3. **The seal (chosen).**
   - When a pairing is approved by pointer, or a recommended item is added to cart in the demo, a 20 px vermilion seal stamps onto the item.
   - Motion: scale 1.15→1 and rotate −4°→−1.5° over `--d-4` with `--ease-spring`. The seal's resting angle varies by item id (±2°), so seals look hand-placed.
   - Keyboard approvals show the seal instantly, with no motion.
4. **X-ray (lifting the lid, demo).**
   - The toggle (a header switch, or the `X` key) lifts each rail 4 px. A ledger "prints" beneath it via `clip-path: inset(0 0 100% 0)` → `inset(0)` over `--d-3`.
   - Ledger rows line up with cards by index numerals (①②③), not connector lines.
   - The motion starts at the rail it explains (spatial consistency). Reduced motion makes it appear instantly.
   - Latency is shown as measured (`fetch` duration). Server time appears only if the API sends `Server-Timing` with `Timing-Allow-Origin`.
5. **Host-skin swap (demo).**
   - Switching skins runs a same-document View Transition: a cross-fade at `--d-4`, instant under reduced motion.
   - The widget re-themes with zero configuration, which proves the "inherits your theme" claim live.
   - On product-page navigation, a cross-document transition keeps the still-life plate in place (`view-transition-name: plate-{id}`), as an enhancement in Chromium and Safari only.
6. **Stage track (dashboard home).**
   - Three stations: *Catalog only* → *Learning from shoppers* → *Learned ranking*. The labels map to engine states and source activations, never invented.
   - The store's seal sits at its true position, with the exact remaining count in Gloock numerals ("412 more events").
   - When a retrain changes stage, the seal glides to the next station (`--d-4`, spring) once. A one-line "What changed" entry explains the effect in shopper terms.

### 7.7 Product imagery for the synthetic catalog: "still-life plates"

- **Deterministic.** `seed = hash(product_id)`. The same SKU always draws the same plate. Generate client-side from a shared sprite of silhouettes, or server-side at catalog build and cache it as static SVG.
- **Grammar.**
  - Category → silhouette. Tawa: a disc plus handle. Kadai: a deep bowl with two loop handles. Katori: a small flared bowl. Dabara-tumbler. Filter-coffee filter. Lota. Masala dabba: a lid with seven wells. Ladle. Knife. Mortar and pestle. Jar.
  - Material → two or three spot inks. Cast iron: graphite + oil sheen. Brass: ochre + highlight. Copper. Steel: cool grey. Terracotta. Wood. Stoneware.
  - The seed sets scale (±8%), angle (±6°), highlight position, ground colour from the material's complement, and a ground pattern from {plain, halftone, ruled}.
- **Print feel without filters.**
  - A second ink layer offset by 1–1.5 px (riso misregistration).
  - Halftone shadow via `<pattern>` circles.
  - No `feTurbulence` (§3.7). Keep each plate ≤ 2 KB and the silhouette sprite ≤ 12 KB, gzip.
- **Composition.** Aspect 4:5, the object at a 30% optical-centre offset, and one consistent light direction (top-left) across the catalog. That consistency is what makes generated art look art-directed.
- **Honesty and accessibility.** Plates are decorative, so `alt=""` where the title sits adjacent. The footer says illustrations are generated for a synthetic store. Never imitate product photography.

### 7.8 Voice and copy

Write one fact per sentence, number first, in merchant words.

Reason templates must be ≤ 48 characters in the widget:

- "34 shoppers bought this with Cast-iron tawa"
- "You bought this 3 times"
- "Same brand, same price band"
- "Pairs with tawas (approved by the store)"
- "New this fortnight"

**Never write:**

- "Recommended for you!"
- "Trending 🔥"
- "Our AI thinks…"
- Any "only N left" or "N people viewing" message.

Dashboard verdicts:

- "Ready to test: about 3–5 weeks."
- "Too small to test revenue right now. You'd need about 9–14 months. Try a preference test instead."

### 7.9 Accessibility and performance budgets

| Surface | JS (gzip) | CSS (gzip) | Fonts | Requests | Field targets (P75) |
|---|---|---|---|---|---|
| Widget | **≤ 5 KB** (today 2.37 KB) | **≤ 1.5 KB** (today 0.51 KB) | 0 | 1 API call per page (batch placements), 0 third-party | No long task > 50 ms; CLS contribution 0 below the fold; storefront Lighthouse drop ≤ 2 points (Shopify allows 10) |
| Demo | ≤ 30 KB | ≤ 15 KB | ≤ 110 KB | ≤ 12 on first load | LCP ≤ 2.5 s and INP ≤ 200 ms on Russell's 9/3/100 Mbps profile with Galaxy A24-class CPU throttling |
| Dashboard | ≤ 60 KB incl. uPlot (app ≤ 35 KB) | ≤ 15 KB | ≤ 80 KB | ≤ 10 | INP ≤ 200 ms; tables > 200 rows use `content-visibility` |

**Accessibility:**

- WCAG 2.2 AA throughout.
- Targets ≥ 24 px, and ≥ 44 px for primary touch actions.
- Focus is always visible and never obscured.
- All drag or swipe actions have buttons.
- Live regions for async results.
- Heading outline correct with the host's.
- `lang` on vernacular strings.
- The widget passes axe on three host skins plus a forced-colours check.

### 7.10 Do not do (slop tells)

- Inter, Roboto, Space Grotesk, Geist or Instrument Serif. Also no Google Fonts or any CDN request from shopper pages.
- Purple/blue-to-pink gradients, mesh/aurora backgrounds, glassmorphism, neon glows, "glow on hover".
- A bento grid of KPI cards as the dashboard home; sparkline confetti; donut charts; gauges.
- Emoji in headings, sparkle ✨ icons for "AI", "Powered by AI" badges, robot illustrations.
- Skeleton shimmers that flash for 80 ms; spinners where nothing is slow; toasts for things the user just watched happen.
- Scroll-jacking, Lenis, parallax heroes, cursor followers, magnetic buttons.
- Hover-only information; truncated product titles; fixed-count rails padded with weak items.
- Japanese pastiche: waves, sakura, faux-brush fonts, decorative kanji.
- Grain overlays everywhere; SVG turbulence filters.
- False urgency, fake counts, invented popularity, unlabelled pins, confirm-shaming consent copy.
- Rounded-2xl-everything with uniform shadows; 3 px coloured left borders on cards; uppercase tracking-wide labels on every element.
- Copy like "Unlock insights", "Supercharge your store", "Seamless", "Leverage".

---

## 8. Tools decision table

| Need | Use | Don't use | Why | Baseline note |
|---|---|---|---|---|
| Widget isolation | Shadow DOM + constructed stylesheet + `::part` + custom props | iframes; global class names | Inherit host font/colour, block host CSS collisions, no extra document | Widely |
| Widget layout | Container queries, scroll-snap, `subgrid` | Media queries; JS measuring | Slot width is unknown | Widely |
| Theming | `oklch()`, `color-mix()`, `light-dark()`, relative colours, `@layer` | Sass, CSS-in-JS | No build; perceptual tokens | Widely / Newly |
| Enter/exit | CSS transitions + `@starting-style`; WAAPI for sequences | GSAP, Motion (not needed) | Interruptible, 0 KB | Newly / Widely |
| Springs | `linear()` token | JS spring libs | 0 KB | Widely |
| Page transitions | Same-document View Transitions (dashboard, skins); cross-document as enhancement (demo) | SPA router for transitions | Progressive | Newly / not Baseline |
| Scroll effects | None; scroll-driven animation for rail progress only as enhancement | Lenis, scroll-jacking | NN/g; Firefox lacks it | Not Baseline |
| Popovers and menus | `popover` + anchor positioning with centred fallback | Floating UI unless needed | Native top layer, light-dismiss | Newly / partial |
| Commands | `command`/`commandfor` + tiny fallback | Bespoke click wiring everywhere | Declarative | Newly (Dec 2025) |
| Disclosures | `<details>` (+ `name` for groups) | Custom accordions | Accessible by default | Widely / Newly |
| Dialogs | `<dialog>` + `inert` | Custom modals | Focus handling built in | Widely |
| Dashboard state | Vanilla modules; Preact+htm (4.9 KB br) if views grow past ~6 stateful screens | React + bundler, Alpine, Lit | No build; small | n/a |
| Charts | uPlot (19.8 KB br) for time series; HTML/CSS bars and dot plots | Observable Plot + d3 (~136 KB br), Chart.js | Size; honesty of encodings | n/a |
| Vector motion | SVG + CSS | Lottie (40 KB br + JSON), Rive (222 KB br) | No motion designer pipeline; size | n/a |
| Fonts | Self-hosted WOFF2 subsets (§7.2), `font-display: swap`, metric overrides | Google Fonts / any CDN | DPDP no-third-party stance; speed | Widely |
| Product art | Seeded SVG still-life plates with `<pattern>` halftone | Raster AI images; `feTurbulence` | Honest, tiny, consistent | Widely |
| Numbers | `Intl.NumberFormat('en-IN')` | Hand formatting | Lakh/crore grouping | Widely |
| Long tables | `content-visibility: auto` | Virtual-scroll libs | Native | Newly |

---

## 9. Self-review checklist for the implementer

**Capture before calling it done.** Take screenshots for each surface at **390×844** and **1440×900**, in **light and dark**:

- Widget on all four demo host skins.
- Demo home, product page and product page with x-ray on.
- Dashboard home, rules, pairings, tests (running and "too small"), catalog, privacy.
- One set with `prefers-reduced-motion: reduce` and one with forced colours (Windows High Contrast emulation).

**Visual**

- [ ] Only one saturated colour per view, and it is the seal or the primary action.
- [ ] No text below 12 px; body is 16 px in the dashboard.
- [ ] All prices and counts use tabular figures; ₹ renders in the intended face, not a fallback.
- [ ] Headings balanced; no orphaned single words in hero lines at 390 px.
- [ ] Every item in §7.10 is absent: search the CSS for `gradient`, `backdrop-filter`, `Inter`, `blur(`.
- [ ] Still-life plates share one light direction; the same SKU renders the same plate twice.

**Honesty**

- [ ] Every reason shown traces to a `reason_code` plus evidence. No count below 5 is displayed.
- [ ] Pinned items show "Featured by the store".
- [ ] No timers, scarcity or "viewing now" anywhere.
- [ ] Stage labels and counts equal what `/v1/...` returns; nothing is interpolated or estimated.
- [ ] Power verdict wording matches the calculator output; interleaving caveat present.

**Accessibility (keyboard-only pass, then a screen reader pass)**

- [ ] Tab order follows reading order on every page; focus ring visible on paper, sunk and raised surfaces in both themes.
- [ ] Focus never hidden by the consent bar or sticky save bar (2.4.11).
- [ ] Rail scrollable by keyboard, with prev/next buttons ≥ 24 px; the "Why?" disclosure opens with Enter/Space.
- [ ] Pairing review fully operable by A/R/S and by buttons; nothing requires dragging (2.5.7).
- [ ] Widget heading level correct inside the host outline; one link per card; the accessible name is not a wall of text.
- [ ] axe: zero serious or critical issues on all screenshots' states.
- [ ] Contrast spot-check with the token tables (§7.3) and the widget's 75% muted rule on all host skins.

**Motion**

- [ ] No animation exceeds 360 ms; enters use ease-out, exits are shorter.
- [ ] Keyboard-triggered changes are instant.
- [ ] Reduced motion: no transforms, no stagger, no view transitions.
- [ ] Rapidly toggling x-ray or skins never janks or leaves half-states (transitions interruptible).

**Performance** (Chrome DevTools: "Low-tier mobile" CPU, network 9 Mbps / 3 Mbps / 100 ms)

- [ ] Widget JS ≤ 5 KB gz, CSS ≤ 1.5 KB gz (`gzip -9 -c widget/mitate.js | wc -c`).
- [ ] Demo LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1; dashboard INP ≤ 200 ms on rules save and pairing approve.
- [ ] No request to any origin other than the store/Mitate API from shopper pages (DevTools → Network → filter "-domain:").
- [ ] Fonts: only the preloaded text face blocks first render; total font bytes within §7.2.
- [ ] Lighthouse on a stock Shopify Dawn page with and without the widget: drop ≤ 2 points (average of 3 runs).

---

## Could not verify

- Herlocker et al. 2000 "histogram of neighbours is most persuasive": full text not accessed.
- Exact Netflix interleaving sensitivity figure (the Netflix tech blog returned 403). The 10–100× figures are via DoorDash and the repo's D13.
- Awwwards Site of the Year 2025 official winner: only badge listings were seen.
- Benji Taylor's "Family Values" original wording: only secondary summaries were seen.
- Gen Z trend statistics from agency blogs (Net-Craft, Letterhend, DesignRush) were not used for any decision.
- Current per-GB mobile data price in India: latest found is 2021–22. Device CPU and latency are the binding constraints regardless (§1.5).
- Myntra and Meesho recommendation UI specifics beyond their published papers and design blogs.
