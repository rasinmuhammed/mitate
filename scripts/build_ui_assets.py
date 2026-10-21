"""Self-hosted UI fonts and the brand mark, reproducibly.

    uv run --with "fonttools[woff]==4.60.1" python scripts/build_ui_assets.py

Sources are OFL fonts from google/fonts at a pinned commit, checked by SHA-256. Outputs go to
``ui/fonts/`` (WOFF2 subsets plus each family's OFL.txt), ``ui/mark.svg`` and ``ui/seal.svg``.
Subsets cover Latin, Latin-1, general punctuation, the rupee sign and a few arrows: everything the
UI prints. No font is ever fetched from a third-party origin at run time
(docs/research_notes/07_ui_ux_design.md, 3.4).
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path

import httpx
from fontTools import subset
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

COMMIT = "47d54ca1ebb14fd864695f1d39dc734f39017c33"
RAW = f"https://raw.githubusercontent.com/google/fonts/{COMMIT}/ofl"
SOURCES = {  # path under ofl/ -> SHA-256
    "gloock/Gloock-Regular.ttf": "c3a67e0212f7da01764796ae721cb9bab3e122d142354f97cf9c63fb763e6d77",
    "gloock/OFL.txt": "321e28d45a430e017d9d1d6a47ef1fd261aad354723b014cf1933a765036c311",
    "aneklatin/AnekLatin[wdth,wght].ttf": (
        "ef7077abf2166add6ab6a64b4a4a4407859bf4f9e5b9058b51ac01ada136b295"
    ),
    "aneklatin/OFL.txt": "de62be758bba267e653a5f457dfebb4680957e2c343eb3e7de15aff4746bed05",
    "ibmplexmono/IBMPlexMono-Regular.ttf": (
        "6a3412f058c7d8dfd9170c41e85ade48e5156ecb89356110ca57a0a27734af46"
    ),
    "ibmplexmono/IBMPlexMono-Medium.ttf": (
        "a9b4c49bb299e05b5f6c481e7fb5e78943d2793249a0c8874ab574a2d1ea6755"
    ),
    "ibmplexmono/OFL.txt": "7e6b2818edbd8f6a01ae80641cc8f16a51080d08fb4e532be3a0b6f74adb07da",
    "shipporimincho/ShipporiMincho-Bold.ttf": (
        "63bc4eddc74793f671c3ab827c5175e773ffbe569d0bf50ee65375ea9e3bc286"
    ),
    "shipporimincho/OFL.txt": "41fba056279be5f45ff9a99e44b7b53897b42732f5806d8e666e0ab49ac6bd38",
}
UNICODES = (
    "U+0020-007E,U+00A0-00FF,U+0131,U+0152-0153,U+02C6,U+02DA,U+02DC,U+2010-2027,U+2030-203A,"
    "U+2044,U+20B9,U+2122,U+2190-2193,U+2212,U+2215,U+FEFF"
)
CACHE = Path(os.environ.get("MITATE_CACHE", Path.home() / ".cache" / "mitate")) / "fonts" / COMMIT
OUT = Path(__file__).resolve().parent.parent / "ui"


def fetch(rel: str) -> bytes:
    path = CACHE / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(f"{RAW}/{rel}", timeout=120, follow_redirects=True)
        r.raise_for_status()
        path.write_bytes(r.content)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCES[rel]:
        sys.exit(f"{rel}: SHA-256 {digest} does not match the pinned {SOURCES[rel]}")
    return data


def woff2(font: TTFont, dest: Path, features: str = "kern,liga,tnum,lnum,case,ss01") -> int:
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = features.split(",")
    opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]  # keep copyright and licence records (OFL)
    opts.notdef_outline = True
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=subset.parse_unicodes(UNICODES))
    sub.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    dest.write_bytes(buf.getvalue())
    return dest.stat().st_size


def instance(data: bytes, axes: dict[str, float | tuple[float, float]]) -> TTFont:
    """Limit a variable font's axes, then round-trip it so the subsetter sees plain tables."""
    buf = io.BytesIO()
    instancer.instantiateVariableFont(TTFont(io.BytesIO(data)), axes).save(buf)
    return TTFont(io.BytesIO(buf.getvalue()))


def has_rupee(font: TTFont) -> bool:
    return 0x20B9 in font.getBestCmap()


def mark(text: str, dest: Path) -> None:
    """Glyphs from Shippori Mincho Bold as SVG outlines: no font is needed to draw the mark."""
    font = TTFont(io.BytesIO(fetch("shipporimincho/ShipporiMincho-Bold.ttf")))
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    upm = font["head"].unitsPerEm
    paths, x = [], 0
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyphs)
        # font units are y-up; SVG is y-down with the ascent at the top of the em box
        glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, x, upm * 0.88)))
        paths.append(pen.getCommands())
        x += glyphs[name].width
    d = " ".join(paths)
    dest.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {x} {upm}" role="img"'
        f' aria-label="{text}"><path fill="currentColor" d="{d}"/></svg>\n'
    )


def main() -> None:
    fonts = OUT / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    for fam in ("gloock", "aneklatin", "ibmplexmono", "shipporimincho"):
        (fonts / f"OFL-{fam}.txt").write_bytes(fetch(f"{fam}/OFL.txt"))
    sizes = {}
    g = TTFont(io.BytesIO(fetch("gloock/Gloock-Regular.ttf")))
    assert has_rupee(g), "Gloock lost the rupee sign"
    sizes["gloock-latin.woff2"] = woff2(g, fonts / "gloock-latin.woff2")
    anek = fetch("aneklatin/AnekLatin[wdth,wght].ttf")
    text = instance(anek, {"wdth": 100, "wght": (400, 700)})
    assert has_rupee(text), "Anek Latin lost the rupee sign"
    sizes["anek-latin-text.woff2"] = woff2(text, fonts / "anek-latin-text.woff2")
    display = instance(anek, {"wdth": (75, 100), "wght": 700})
    sizes["anek-latin-display.woff2"] = woff2(display, fonts / "anek-latin-display.woff2")
    for w in ("Regular", "Medium"):
        f = TTFont(io.BytesIO(fetch(f"ibmplexmono/IBMPlexMono-{w}.ttf")))
        assert has_rupee(f), "IBM Plex Mono lost the rupee sign"
        name = f"plex-mono-{400 if w == 'Regular' else 500}-latin.woff2"
        sizes[name] = woff2(f, fonts / name)
    mark("見立て", OUT / "mark.svg")  # the wordmark
    mark("見", OUT / "seal.svg")  # one character, legible inside a 16 px seal
    for name, n in sizes.items():
        print(f"{name:28s} {n / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
