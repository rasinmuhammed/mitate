# 06 — India DPDP Act 2023 + DPDP Rules 2025: what applies to Mitate

Review date: **2026-10-19**. This is an engineering reading for product design, not legal advice. Have Indian counsel review the contract templates and the treatment of children's data before launch.

## Primary sources (fetched and read)

| Instrument | Official source |
|---|---|
| Digital Personal Data Protection Act, 2023 (No. 22 of 2023). Gazette, 11 Aug 2023 | https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf |
| **G.S.R. 843(E)**, MeitY, "New Delhi, the 13th November, 2025": Act commencement notification (Gazette Extraordinary No. 757, Thursday 13 Nov 2025) | https://egazette.gov.in/WriteReadData/2025/267647.pdf |
| **G.S.R. 846(E)**, MeitY, 13 Nov 2025: **Digital Personal Data Protection Rules, 2025** (Gazette Extraordinary No. 760, dated Thursday 13 Nov 2025; gazette ID CG-DL-E-14112025-267650; digitally signed 2025-11-14 10:43 IST) | https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf |
| PIB release "Government notifies DPDP Rules…" (Release ID 2190014, posted 14 Nov 2025) and PIB backgrounder (17 Nov 2025: "notified … on 14 November 2025") | https://www.pib.gov.in/PressReleasePage.aspx?PRID=2190014 · https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf |

**On the date:** the Gazette masthead and the notification both say **13 November 2025**. PIB says "notified on 14 November 2025", which is when the e-Gazette was uploaded and signed. Commencement runs "from the date of publication", so plan on the earlier date: **13 Nov 2026** and **13 May 2027**. Some commentators use the 14th.

---

## 1. Phased commencement (exact)

**Act: G.S.R. 843(E):**
- (a) **On publication (13 Nov 2025):** s.1(2), s.2, ss.18–26, 35, 38–43, s.44(1) and (3). These cover definitions, the Data Protection Board, and the rule-making powers.
- (b) **One year (≈13 Nov 2026):** s.6(9) (Consent Manager registration) and s.27(1)(d).
- (c) **Eighteen months (≈13 May 2027):** "sections 3 to 5, sub-sections (1) to (8) and (10) of section 6, sections 7 to 10, sections 11 to 17, section 27 except clause (d) of sub-section (1) …, sections 28 to 34, 36, 37 and sub-section (2) of section 44". **All substantive duties on Data Fiduciaries and Processors fall here.**

**Rules: G.S.R. 846(E), Rule 1:**
- "(2) Rules 1, 2 and 17 to 21 shall come into force on the date of their publication in the Official Gazette." These cover the Board.
- "(3) Rule 4 shall come into force one year after the date of publication of this Gazette." This covers Consent Managers, **≈13 Nov 2026, 25 days from today**.
- "(4) Rules 3, 5 to 16, 22 and 23 shall come into force eighteen months after the date of publication of this Gazette." This covers notice, security, breach, retention, children, SDF, rights, transfer and research. **≈13 May 2027.**

**Watch items (secondary sources, not verified against the Gazette):**
- **12-month compression proposal.** MeitY reportedly consulted from 23 Jan 2026, with comments due 4 Feb 2026, on cutting 18 months to 12 (the last date would become 13 Nov 2026). As of Sept–Oct 2026 several trackers report **no amending notification** (chambers.com, 2026-02-23; mirvolegal.com, 2026-09-20; consentos.in, 2026-10-09; dpdprules.org, verified 2026-10-18). Engineering hedge: **be feature-complete by 13 Nov 2026.**
- **Removal of Difficulties Order.** dpdprules.org alone reports a "Digital Personal Data Protection (Removal of Difficulties) Order, 2026", S.O. 5458(E), ~5–6 Oct 2026. It reportedly makes wording fixes only: s.9(1) "child or of a person with disability", and "data audit" in s.10(2)(c)(ii). It does not change commencement. **UNVERIFIED.**
- **Corrigendum G.S.R. 892(E)** (Dec 2025) is reported. Not reviewed.
- **Data Protection Board:** MeitY invited applications on 6 May 2026. No appointments were reported as of Sept 2026 (secondary).

---

## 2. Roles: who is the Data Fiduciary?

- s.2(i) "**Data Fiduciary**" means any person who alone or in conjunction with other persons determines the purpose and means of processing of personal data. s.2(k) "**Data Processor**" means any person who processes personal data on behalf of a Data Fiduciary.
- s.2(t) "personal data" means "any data about an individual who is identifiable by or in relation to such data". **Pseudonymous shopper IDs are personal data** when the merchant can link them to an account, phone, email or cookie, which a merchant can always do. Rule 6(1)(a) treats "encryption, obfuscation, masking or the use of virtual tokens" as *security measures*, not as an exit from the Act. Only truly anonymised aggregates are plausibly out of scope, e.g. category→category co-purchase counts with k-anonymity thresholds.
- **Merchant = Data Fiduciary.** It decides to personalise and owns the shopper relationship.
- **Mitate = Data Processor** when it processes a merchant's events solely to serve that merchant's recommendations under contract. s.8(2): "A Data Fiduciary may engage, appoint, use or otherwise involve a Data Processor … only under a valid contract."
- **Mitate becomes a Data Fiduciary** for any processing *it* decides on, such as pooling events across merchants to train a shared model, using customer data to tune defaults, product analytics on shopper-level data, or using any merchant data in the public benchmark. **Design rule:** per-merchant data isolation by default. Any cross-merchant learning uses only anonymised, aggregated signals (e.g. category-level graphs), or requires explicit merchant opt-in plus a shopper-facing notice listing that purpose.
- s.8(1): the Fiduciary stays responsible "irrespective of any agreement to the contrary" for processing "undertaken by it or on its behalf by a Data Processor". Expect merchants to push obligations down to Mitate by contract.
- s.11(1)(b): on request, the Fiduciary must give "the identities of all other Data Fiduciaries and **Data Processors** with whom the personal data has been shared". Merchants will name Mitate.

## 3. Lawful basis, notice and consent (from 13 May 2027)

- s.4(1): processing only "(a) for which the Data Principal has given her consent; or (b) for certain legitimate uses." s.7(a) covers data "voluntarily provided" for a "specified purpose" where she has not objected. Behavioural recommendations built from browsing telemetry are **not safely covered by s.7(a)**. Plan for the merchant to collect **consent** that names personalised recommendations as a purpose.
- s.6(1): consent must be "free, specific, informed, unconditional and unambiguous with a clear affirmative action … limited to such personal data as is necessary for such specified purpose".
- s.6(4): withdrawal must be "at any time, with the ease of doing so being comparable to the ease with which such consent was given". s.6(6): on withdrawal the Fiduciary must "cease and cause its Data Processors to cease processing".
- s.5 + **Rule 3**: the notice must be standalone and "understandable independently". It must include "(i) an itemised description of such personal data; and (ii) the specified purpose or purposes of, and specific description of the goods or services to be provided or uses to be enabled by, such processing". It must link to withdrawal, rights and a complaint route. s.5(3)/6(3): English or any Eighth Schedule language.
- **Mitate deliverables:**
  - a merchant-facing notice snippet that itemises event types (`view`, `add_to_cart`, `purchase`, `search`, device/session ID) and the purpose ("personalised product recommendations on this store");
  - a `consent_state` flag on every event;
  - a hard switch to **non-personalised** recommendations (popularity, or item-to-item from the current page only) when consent is absent or withdrawn;
  - a `DELETE /shoppers/{id}` plus suppression list.

## 4. Purpose limitation, minimisation, retention and erasure

- Purpose limitation and minimisation flow from s.6(1) ("limited to such personal data as is necessary") and s.8(7). Collect only event type, item, timestamp and pseudonymous ID. Do **not** ingest name, email, phone, address or precise location.
- s.8(7): erase "upon the Data Principal withdrawing her consent or as soon as it is reasonable to assume that the specified purpose is no longer being served, whichever is earlier", and "(b) cause its Data Processor to erase".
- **Rule 8(1) + Third Schedule:** a fixed **3-year** inactivity erasure ("three years from the date on which the Data Principal last approached the Data Fiduciary … or the commencement of the [Rules], whichever is latest"). This applies only to an "e-commerce entity having not less than **two crore** registered users in India" (and to large gaming and social-media intermediaries). The definition excludes "a seller offering her goods or services for sale on a marketplace". **Small stores are below the threshold**, so the general s.8(7) standard applies to them. Rule 8(2): a 48-hour advance notice before such erasure.
- **Rule 8(3), a minimum retention floor:** the Fiduciary "shall retain, in respect of any processing … undertaken by it or on its behalf by a Data Processor, such personal data, associated traffic data and other logs of the processing for a minimum period of **one year** from the date of such processing, for the purposes as specified in the Seventh Schedule". Illustration Case 2 makes the Processor keep data and logs for one year too. **Rule 6(1)(e)** also requires keeping "logs and personal data for a period of one year" for breach detection.
- **Mitate design implication:** separate the *serving store* (feature tables) from an *append-only processing log*. The serving store has short, merchant-configurable TTLs (e.g. 90–180 days for behavioural features) and is erased on withdrawal. The processing log is access-restricted, encrypted, kept **≥ 365 days**, then purged. Erasure requests remove the shopper from the serving and training stores at once. The legally required log copy is held under restricted access until its one-year floor ends.

## 5. Security and breach notification

- **Rule 6(1):** minimum safeguards are:
  - (a) encryption, obfuscation, masking or virtual tokens;
  - (b) access control;
  - (c) access logging, monitoring and review;
  - (d) backups and continuity;
  - (e) 1-year log retention;
  - (f) "appropriate provision in the contract … between such Data Fiduciary and such a Data Processor … for taking reasonable security safeguards";
  - (g) technical and organisational measures.
- **Rule 7, breach:**
  - **To each affected Data Principal:** "without delay", covering the nature, extent and timing, likely consequences, mitigation, protective steps, and a contact.
  - **To the Board:** "(a) without delay" an initial description, and "(b) **within seventy-two hours** of becoming aware of the breach" a detailed report covering facts, causes, mitigation, findings on the person responsible, remediation, and the notices sent to principals.
  - There is no materiality threshold. Every personal data breach is in scope.
- The clock starts when the **Fiduciary** becomes aware, so the Mitate DPA must commit to notifying the merchant within a short window (e.g. ≤ 24 h) with the Rule 7(2)(b) fields pre-filled.

## 6. Children: the highest-risk area for a recommender

- s.2(f): "**child**" means an individual who has not completed the age of **eighteen** years.
- s.9(1): "obtain **verifiable consent of the parent**" before processing **any** personal data of a child. Rule 10 sets out due diligence: the parent must be an identifiable adult, checked via reliable identity and age details already held, or via details or a virtual token from an authorised entity or DigiLocker.
- s.9(2): no processing "likely to cause any detrimental effect on the well-being of a child".
- **s.9(3): "A Data Fiduciary shall not undertake tracking or behavioural monitoring of children or targeted advertising directed at children."**
- **Exemptions (Rule 12 + Fourth Schedule)** cover only healthcare, educational institutions, crèches, child transport, and purposes such as legal functions, subsidies, email-only accounts, real-time location for safety, blocking detrimental content, and "confirmation by the Data Fiduciary that the Data Principal is not a child". **There is no exemption for e-commerce personalisation.** Parental consent does not lift the s.9(3) ban either, because s.9(1) and s.9(3) are separate duties.
- **Penalty:** up to **₹200 crore** (Schedule item 3).
- **Mitate design implications:**
  - every event and profile carries an `age_status ∈ {adult_verified, adult_declared, minor, unknown}` set by the merchant;
  - for `minor`, Mitate builds **no profile**, does no session-behaviour modelling, and does no targeted promotions. It serves only contextual, non-personalised recommendations (popularity, or item-to-item from the item currently in view, with no user history). Whether even session-level, within-visit personalisation counts as "behavioural monitoring" is **UNCLEAR**, so treat it as prohibited for minors;
  - merchants in child-heavy categories (toys, school supplies, teen fashion) should default `unknown` to the minor-safe path;
  - document the policy in the paper and the docs.

## 7. Significant Data Fiduciary (SDF)

- s.10(1): the Central Government notifies SDFs based on "the volume and sensitivity of personal data processed; risk to the rights of Data Principal; … sovereignty and integrity … electoral democracy … security of the State; and public order".
- s.10(2) duties: an India-based DPO, an independent data auditor, periodic DPIA and audit.
- **Rule 13:** an annual DPIA and audit, with a report to the Board (13(1)–(2)). **13(3):** "due diligence to verify that technical measures including **algorithmic software** adopted by it … are not likely to pose a risk to the rights of Data Principals". **13(4):** localisation of specified data and its traffic data.
- Small merchants and Mitate are unlikely to be notified, but a large merchant customer could be. Mitate should ship a model card and DPIA input pack (data used, features, the minor-safe path, evaluation and bias notes) so SDF customers can meet Rule 13(3).
- **Penalty:** up to ₹150 crore.

## 8. Other relevant items

- **Data Principal rights (ss.11–14; Rule 14).** These are access (a summary plus the list of processors), correction and erasure (s.12(3): erase "unless retention … is necessary for the specified purpose or for compliance with any law"), grievance redressal, and nomination. Rule 14(3) requires the grievance system to respond "within a reasonable period not exceeding **ninety days**". Mitate needs per-shopper **export** and **erase** APIs that merchants can call.
- **Contact (Rule 9):** the Fiduciary publishes the business contact of its DPO or responsible person.
- **Cross-border (s.16; Rule 15):** transfer is allowed unless the Central Government restricts it by order. Hosting outside India is permitted today, but keep an India region option for SDF customers (Rule 13(4)).
- **Startups (s.17(3)):** the Government "may … notify certain Data Fiduciaries or class of Data Fiduciaries, **including startups**" to be exempt from s.5, s.8(3), s.8(7), s.10 and s.11. **No such notification was found.** Do not rely on it.
- **Research exemption (s.17(2)(b); Rule 16; Second Schedule):** the Act does not apply to processing "necessary for research, archiving or statistical purposes if the personal data is not to be used to take any decision specific to a Data Principal" and if the processing follows the Second Schedule standards (lawful, limited, accurate, retained only as needed, secured, accountable). This is relevant only if Mitate ever builds a benchmark from *Indian merchants' real data*. The public datasets in 05 are non-Indian and anonymised, and are very likely not "personal data" under s.2(t).
- **Penalties (Schedule to s.33):**
  - security safeguards failure (s.8(5)): up to **₹250 crore**;
  - breach-notification failure (s.8(6)): **₹200 crore**;
  - children (s.9): **₹200 crore**;
  - SDF duties (s.10): **₹150 crore**;
  - anything else: **₹50 crore**.
- **Consent Managers (Rule 4, from ≈13 Nov 2026):** these must be Indian companies registered with the Board. Optional for Mitate, but the consent-state API should accept consent artefacts from a Consent Manager later.

## 9. Compliance checklist for the Mitate product (target date 13 Nov 2026 as a hedge; legal deadline ≈13 May 2027)

1. **DPA template** (merchant = Fiduciary, Mitate = Processor). It covers processing only on instructions, Rule 6 safeguards, a breach notice to the merchant within ≤ 24 h carrying the Rule 7 fields, erasure on instruction or withdrawal, the 1-year log floor (Rule 8(3)), a sub-processor list, and audit cooperation.
2. **Event schema:** add `consent_state`, `age_status`, `purpose`, and `merchant_id`, with hard tenant isolation. Accept no direct identifiers.
3. **Minor-safe and no-consent serving path:** contextual and popularity recommendations only, with no profile writes.
4. **Erase and export APIs**, a suppression list, and a propagation SLA into features and models (retrain or unlearn cadence documented).
5. **Retention:** configurable feature TTLs plus a restricted ≥ 365-day processing log, then purge.
6. **Security:** encryption at rest and in transit, RBAC, access logs reviewed, backups.
7. **Cross-merchant learning** only on anonymised aggregates (category graphs, k ≥ threshold), or with explicit opt-in.
8. **SDF pack:** model card, DPIA inputs, and description of the algorithmic risk controls.
