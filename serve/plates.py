"""Still-life plates: deterministic SVG illustrations for the synthetic demo catalog.

The demo store has no photographs, and imitating product photography would misrepresent it, so
every product gets a printed-plate illustration instead (docs/research_notes/07_ui_ux_design.md,
7.7). The category picks the silhouette, the material picks the inks, and a hash of the product
id picks scale, tilt, ground colour and ground pattern, so a product always draws the same plate.
One light direction (top left) runs across the whole catalog; the shadow is a halftone and a
second ink is offset by 1.5 px like a risograph misregistration. No raster, no SVG filters.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

W, H = 400, 500
CX, CY = 200, 285  # optical centre: the object sits a little below the middle

# material -> (body ink, highlight ink)
INKS: dict[str, tuple[str, str]] = {
    "Cast Iron": ("#2e2c2a", "#5b554f"),
    "Iron": ("#3b3733", "#6a625a"),
    "Tri-ply Steel": ("#a2aaaf", "#e4e9ec"),
    "Steel": ("#9fa7ac", "#e1e6e9"),
    "Stainless": ("#a7afb4", "#e6ebee"),
    "Non-stick": ("#34302e", "#615a55"),
    "Hard Anodised": ("#45403c", "#746b64"),
    "Aluminium": ("#b7bbbc", "#eef0f0"),
    "Brass": ("#b98f33", "#ecd28b"),
    "Copper": ("#ad5d36", "#e8a47c"),
    "Carbon Steel": ("#5c6063", "#9da2a5"),
    "Damascus": ("#686c6f", "#b7bcbf"),
    "Teak": ("#82532f", "#b98656"),
    "Acacia": ("#956538", "#c99a64"),
    "Bamboo": ("#c09658", "#e6c993"),
    "Wooden": ("#8c5c34", "#c08f5c"),
    "Silicone": ("#6f8f78", "#b5cdb9"),
    "Stoneware": ("#9c8670", "#d2bfa9"),
    "Ceramic": ("#ddd5c6", "#fbf8f2"),
    "Enamel": ("#2b5574", "#6f97b8"),
    "Glass": ("#a9c4c8", "#eef7f8"),
    "Insulated": ("#4e6357", "#8ca397"),
    "Chikmagalur": ("#a77d4c", "#d9b98b"),
    "Coorg": ("#6e4630", "#a9774f"),
    "Araku": ("#8a4a2a", "#c98a5c"),
}
# grounds: pale papers; light-bodied objects get the deeper ones so they still read
GROUNDS_LIGHT = ["#efe6d6", "#e9dfcf", "#e6e1d7", "#ece2d9", "#e3e6dd", "#efe2cc"]
GROUNDS_DEEP = ["#d9c9ad", "#cbbfae", "#c9cdc3", "#d7c3b5", "#c6c3b8", "#ddc89f"]
SECOND_INK = "#c43512"  # the misregistered line, in the brand's seal vermilion at low opacity
HANDLE = "#2a2623"
WOOD = "#8c5c34"


def _seed(product_id: str) -> int:
    return int.from_bytes(hashlib.sha256(product_id.encode()).digest()[:8], "big")


class Rand:
    """Deterministic draws from a product id (a tiny xorshift; the stdlib RNG is overkill)."""

    def __init__(self, product_id: str) -> None:
        self.x = _seed(product_id) or 1

    def u(self) -> float:
        x = self.x
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.x = x
        return (x % 1_000_003) / 1_000_003

    def pick(self, xs: list[str]) -> str:
        return xs[int(self.u() * len(xs)) % len(xs)]


def _light(hex_: str) -> bool:
    r, g, b = (int(hex_[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 165


# ---------------------------------------------------------------------------- silhouettes
# Each returns SVG markup drawn around (0, 0) in body ink ``b`` with highlight ``h``.
Shape = Callable[[str, str], str]


def pan(b: str, h: str) -> str:
    return (
        f'<rect x="92" y="-14" width="118" height="22" rx="10" fill="{HANDLE}" '
        'transform="rotate(-9)"/>'
        f'<ellipse cx="-30" cy="0" rx="128" ry="62" fill="{b}"/>'
        f'<ellipse cx="-30" cy="-6" rx="108" ry="48" fill="{h}" opacity=".35"/>'
        f'<ellipse cx="-62" cy="-22" rx="34" ry="12" fill="{h}" opacity=".55"/>'
    )


def kadai(b: str, h: str) -> str:
    return (
        f'<path d="M-130 -30 Q-120 90 0 92 Q120 90 130 -30Z" fill="{b}"/>'
        f'<ellipse cx="0" cy="-30" rx="130" ry="30" fill="{b}"/>'
        f'<ellipse cx="0" cy="-30" rx="112" ry="21" fill="{h}" opacity=".3"/>'
        f'<path d="M-130 -26q-42 0-42 22t42 18M130 -26q42 0 42 22t-42 18" fill="none" '
        f'stroke="{b}" stroke-width="12" stroke-linecap="round"/>'
        f'<path d="M-96 6q16 52 70 64" stroke="{h}" stroke-width="10" fill="none" '
        f'stroke-linecap="round" opacity=".5"/>'
    )


def tawa(b: str, h: str) -> str:
    return (
        f'<rect x="110" y="-12" width="120" height="16" rx="8" fill="{WOOD}" '
        'transform="rotate(-6)"/>'
        f'<ellipse cx="-20" cy="0" rx="138" ry="48" fill="{b}"/>'
        f'<ellipse cx="-60" cy="-14" rx="52" ry="12" fill="{h}" opacity=".45"/>'
    )


def cooker(b: str, h: str) -> str:
    return (
        f'<rect x="-100" y="-70" width="200" height="150" rx="16" fill="{b}"/>'
        f'<ellipse cx="0" cy="80" rx="100" ry="22" fill="{b}"/>'
        f'<ellipse cx="0" cy="-70" rx="100" ry="24" fill="{h}" opacity=".55"/>'
        f'<rect x="-14" y="-118" width="28" height="44" rx="6" fill="{HANDLE}"/>'
        f'<rect x="-10" y="-96" width="190" height="20" rx="10" fill="{HANDLE}" '
        'transform="rotate(-4)"/>'
        f'<rect x="-82" y="-50" width="22" height="110" rx="10" fill="{h}" opacity=".45"/>'
    )


def lid(b: str, h: str) -> str:
    return (
        f'<path d="M-140 30 Q-120 -90 0 -92 Q120 -90 140 30Z" fill="{b}" opacity=".9"/>'
        f'<ellipse cx="0" cy="30" rx="140" ry="22" fill="{b}"/>'
        f'<ellipse cx="0" cy="-100" rx="28" ry="12" fill="{HANDLE}"/>'
        f'<path d="M-92 -10 Q-80 -62 -30 -74" stroke="{h}" stroke-width="12" fill="none" '
        f'stroke-linecap="round" opacity=".7"/>'
    )


def knife(b: str, h: str) -> str:
    return (
        '<g transform="rotate(-32)">'
        f'<path d="M-170 -2 L60 -26 L60 12 L-150 14 Q-176 10 -170 -2Z" fill="{b}"/>'
        f'<path d="M-150 -1 L56 -20" stroke="{h}" stroke-width="5" opacity=".7"/>'
        f'<rect x="60" y="-24" width="120" height="34" rx="12" fill="{HANDLE}"/>'
        '<circle cx="92" cy="-7" r="4" fill="#d8d2c8"/>'
        '<circle cx="140" cy="-7" r="4" fill="#d8d2c8"/>'
        "</g>"
    )


def board(b: str, h: str) -> str:
    return (
        f'<path d="M-150 -40 L110 -80 Q150 -84 156 -48 L170 40 Q172 70 140 76 L-120 110 '
        f'Q-160 114 -164 80 L-178 -6 Q-182 -36 -150 -40Z" fill="{b}"/>'
        f'<path d="M-140 -10 L130 -50M-136 22 L136 -16M-130 56 L144 18" stroke="{h}" '
        f'stroke-width="3" opacity=".5"/>'
        '<circle cx="128" cy="-40" r="12" fill="#00000033"/>'
    )


def utensil(b: str, h: str, head: str) -> str:
    heads = {
        "ladle": f'<ellipse cx="-120" cy="10" rx="54" ry="40" fill="{b}"/>'
        f'<ellipse cx="-132" cy="0" rx="22" ry="12" fill="{h}" opacity=".55"/>',
        "turner": f'<rect x="-178" y="-34" width="96" height="72" rx="10" fill="{b}"/>'
        '<path d="M-160 -16h60M-160 2h60M-160 20h60" stroke="#00000030" stroke-width="5"/>',
        "spatula": f'<path d="M-180 -26 Q-130 -40 -88 -14 L-88 14 Q-130 40 -180 26Z" fill="{b}"/>',
    }
    return (
        '<g transform="rotate(-24)">'
        f'<rect x="-92" y="-9" width="260" height="18" rx="9" fill="{b}"/>'
        f'<rect x="-80" y="-6" width="230" height="5" rx="2" fill="{h}" opacity=".6"/>'
        f"{heads[head]}</g>"
    )


def mug(b: str, h: str) -> str:
    return (
        f'<path d="M86 -40 q74 0 74 52 t-74 52" fill="none" stroke="{b}" stroke-width="22"/>'
        f'<rect x="-96" y="-90" width="192" height="180" rx="22" fill="{b}"/>'
        f'<ellipse cx="0" cy="-90" rx="96" ry="22" fill="{h}" opacity=".5"/>'
        f'<rect x="-74" y="-60" width="24" height="120" rx="12" fill="{h}" opacity=".4"/>'
    )


def coffee_bag(b: str, h: str) -> str:
    return (
        f'<path d="M-100 -120 L100 -120 L118 120 L-118 120Z" fill="{b}"/>'
        f'<path d="M-100 -120 L100 -120 L104 -96 L-104 -96Z" fill="{h}" opacity=".6"/>'
        '<rect x="-66" y="-30" width="132" height="92" rx="4" fill="#f4ecdf"/>'
        f'<path d="M-44 -6h88M-44 14h60M-44 34h74" stroke="{b}" stroke-width="6" '
        'stroke-linecap="round"/>'
    )


def south_filter(b: str, h: str) -> str:
    return (
        f'<rect x="-80" y="-30" width="160" height="130" rx="12" fill="{b}"/>'
        f'<rect x="-66" y="-120" width="132" height="96" rx="10" fill="{b}"/>'
        f'<ellipse cx="0" cy="-120" rx="66" ry="16" fill="{h}" opacity=".6"/>'
        f'<ellipse cx="0" cy="-138" rx="16" ry="8" fill="{HANDLE}"/>'
        f'<rect x="-56" y="-100" width="18" height="70" rx="9" fill="{h}" opacity=".4"/>'
        f'<rect x="-66" y="0" width="20" height="86" rx="10" fill="{h}" opacity=".4"/>'
    )


def dripper(b: str, h: str) -> str:
    return (
        f'<path d="M-110 -80 L110 -80 L46 60 L-46 60Z" fill="{b}"/>'
        f'<ellipse cx="0" cy="-80" rx="110" ry="24" fill="{h}" opacity=".55"/>'
        f'<rect x="-70" y="60" width="140" height="18" rx="9" fill="{b}"/>'
        f'<path d="M110 -60 q50 10 30 50" fill="none" stroke="{b}" stroke-width="16" '
        'stroke-linecap="round"/>'
    )


def kettle(b: str, h: str) -> str:
    return (
        f'<path d="M-110 70 Q-130 -60 0 -70 Q130 -60 110 70Z" fill="{b}"/>'
        f'<path d="M100 10 L170 -50 L176 -40 L118 40Z" fill="{b}"/>'
        f'<path d="M-70 -64 Q0 -170 70 -64" fill="none" stroke="{HANDLE}" stroke-width="16" '
        'stroke-linecap="round"/>'
        f'<ellipse cx="0" cy="70" rx="110" ry="18" fill="{b}"/>'
        f'<path d="M-76 30 Q-90 -30 -30 -50" stroke="{h}" stroke-width="12" fill="none" '
        f'stroke-linecap="round" opacity=".6"/>'
    )


def chai_pan(b: str, h: str) -> str:
    return (
        f'<rect x="96" y="-24" width="112" height="20" rx="10" fill="{HANDLE}" '
        'transform="rotate(-8)"/>'
        f'<path d="M-110 -40 L110 -40 L100 60 Q0 76 -100 60Z" fill="{b}"/>'
        f'<ellipse cx="0" cy="-40" rx="110" ry="24" fill="{h}" opacity=".5"/>'
        f'<path d="M-110 -40 l-24 -10" stroke="{b}" stroke-width="10" stroke-linecap="round"/>'
    )


def masala_box(b: str, h: str) -> str:
    wells = "".join(
        f'<circle cx="{x}" cy="{y}" r="26" fill="{c}"/>'
        for (x, y), c in zip(
            [(0, 0), (-62, -30), (62, -30), (-62, 34), (62, 34), (0, -64), (0, 66)],
            ["#c43512", "#e3a72f", "#6b3d1e", "#8a8f2c", "#d77a2a", "#5a2c1c", "#b8862f"],
            strict=True,
        )
    )
    return (
        f'<ellipse cx="0" cy="18" rx="150" ry="118" fill="{b}"/>'
        f'<ellipse cx="0" cy="0" rx="150" ry="118" fill="{h}"/>'
        f'<g transform="scale(1 .78)">{wells}</g>'
    )


def jar(b: str, h: str) -> str:
    return (
        f'<rect x="-86" y="-80" width="172" height="180" rx="26" fill="{b}" opacity=".92"/>'
        f'<rect x="-92" y="-118" width="184" height="44" rx="10" fill="{HANDLE}"/>'
        f'<rect x="-64" y="-56" width="22" height="130" rx="11" fill="{h}" opacity=".55"/>'
    )


def tiffin(b: str, h: str) -> str:
    tiers = "".join(
        f'<rect x="-90" y="{y}" width="180" height="62" rx="10" fill="{b}"/>'
        f'<rect x="-90" y="{y}" width="180" height="8" rx="4" fill="{h}" opacity=".7"/>'
        for y in (-60, 6, 72)
    )
    return (
        f'<path d="M-70 -66 Q0 -170 70 -66" fill="none" stroke="{b}" stroke-width="12"/>'
        f'{tiers}<rect x="-74" y="-50" width="16" height="170" rx="8" fill="{h}" opacity=".4"/>'
    )


def plate(b: str, h: str, thali: bool) -> str:
    katoris = (
        "".join(
            f'<ellipse cx="{x}" cy="{y}" rx="34" ry="16" fill="{h}" stroke="{b}" stroke-width="4"/>'
            for x, y in ((-64, -14), (0, -30), (64, -14))
        )
        if thali
        else ""
    )
    return (
        f'<ellipse cx="0" cy="8" rx="160" ry="70" fill="{b}"/>'
        f'<ellipse cx="0" cy="0" rx="160" ry="70" fill="{h}"/>'
        f'<ellipse cx="0" cy="4" rx="118" ry="48" fill="{b}" opacity=".25"/>{katoris}'
    )


def bowl(b: str, h: str) -> str:
    return (
        f'<path d="M-130 -20 Q-118 96 0 98 Q118 96 130 -20Z" fill="{b}"/>'
        f'<ellipse cx="0" cy="-20" rx="130" ry="34" fill="{h}" opacity=".65"/>'
        f'<ellipse cx="0" cy="-16" rx="108" ry="22" fill="{b}" opacity=".35"/>'
        f'<path d="M-92 16 Q-80 64 -30 78" stroke="{h}" stroke-width="10" fill="none" '
        f'stroke-linecap="round" opacity=".5"/>'
    )


# how far below the object's centre its shadow falls, per silhouette (flat things sit low)
FOOT: dict[str, int] = {
    "pan": 62,
    "tawa": 52,
    "plate": 74,
    "utensil": 74,
    "knife": 86,
    "board": 112,
    "masala_box": 132,
    "coffee_bag": 124,
    "kadai": 96,
    "bowl": 100,
    "dripper": 82,
}


def foot(category: str, stem: str) -> int:
    name = getattr(shape(category, stem), "__name__", "")
    leaf = category.split(" > ")[-1]
    if leaf in ("Spatulas & Ladles", "Dinner Plates"):
        name = "utensil" if leaf.startswith("Spatulas") else "plate"
    return FOOT.get(name, 104)


def shape(category: str, stem: str) -> Shape:
    """Silhouette for a category leaf and product stem ("Masala Box" vs "Storage Jar")."""
    leaf = category.split(" > ")[-1]
    s = stem.lower()
    if leaf == "Spatulas & Ladles":
        head = "ladle" if "ladle" in s else "turner" if "turner" in s else "spatula"
        return lambda b, h: utensil(b, h, head)
    if leaf == "Dinner Plates":
        return lambda b, h: plate(b, h, "thali" in s)
    table: dict[str, Shape] = {
        "Frying Pans": pan,
        "Kadai & Woks": kadai,
        "Tawa": tawa,
        "Pressure Cookers": cooker,
        "Lids": lid,
        "Knives": knife,
        "Chopping Boards": board,
        "Mugs": mug,
        "Coffee": coffee_bag,
        "Coffee Filters": dripper if "pour" in s else south_filter,
        "Tea Kettles": chai_pan if "chai" in s else kettle,
        "Jars & Containers": masala_box if "masala" in s else jar,
        "Lunch Boxes": tiffin,
        "Bowls": bowl,
    }
    return table.get(leaf, jar)


def material_of(title: str) -> str:
    for m in sorted(INKS, key=len, reverse=True):  # "Tri-ply Steel" before "Steel"
        if m in title:
            return m
    return "Steel"


def plate_svg(product_id: str, title: str, category: str, stem: str) -> str:
    """The product's plate: same id, same picture, every time."""
    r = Rand(product_id)
    body, high = INKS[material_of(title)]
    ground = r.pick(GROUNDS_DEEP if _light(body) else GROUNDS_LIGHT)
    pattern = r.pick(["plain", "halftone", "ruled"])
    scale = 0.82 + 0.12 * r.u()
    tilt = -6 + 12 * r.u()
    figure = shape(category, stem)(body, high)
    shadow_y = CY + foot(category, stem) * scale
    ground_marks = {
        "plain": "",
        "halftone": '<rect width="400" height="500" fill="url(#d)" opacity=".5"/>',
        "ruled": '<path d="'
        + "".join(f"M0 {y}h400" for y in range(30, 500, 30))
        + '" stroke="#00000014"/>',
    }[pattern]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
        '<defs><pattern id="d" width="8" height="8" patternUnits="userSpaceOnUse">'
        '<circle cx="2" cy="2" r="1.3" fill="#00000026"/></pattern>'
        '<pattern id="s" width="6" height="6" patternUnits="userSpaceOnUse">'
        '<circle cx="3" cy="3" r="1.8" fill="#3a2a1e"/></pattern></defs>'
        f'<rect width="{W}" height="{H}" fill="{ground}"/>{ground_marks}'
        f'<ellipse cx="{CX + 18}" cy="{shadow_y:.0f}" rx="{140 * scale:.0f}" ry="20" '
        'fill="url(#s)" opacity=".26"/>'
        f'<g transform="translate({CX} {CY}) rotate({tilt:.1f}) scale({scale:.3f})">'
        f'<g transform="translate(1.5 1)" opacity=".22" style="mix-blend-mode:multiply">'
        f"{figure.replace(body, SECOND_INK)}</g>{figure}</g></svg>"
    )
