# ────────────────────────────────────────────────────────────────────────────────────
# BioNOVA Design System — Centralized styling, theme, and component definitions
# ────────────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Optional

from utils import escape_html


# ── PALETTE: Scientific, restrained, WCAG-accessible colors ──
@dataclass
class ColorPalette:
    """Primary color system. All values WCAG-AA compliant."""

    bg_primary: str = "#F6F8FB"
    bg_secondary: str = "#FFFFFF"
    bg_tertiary: str = "#F0F4F8"
    bg_dark: str = "#0A0F1E"
    bg_dark_secondary: str = "#111827"

    text_primary: str = "#14213D"
    text_secondary: str = "#53657A"
    text_light: str = "#F9FAFB"

    primary: str = "#1769AA"
    primary_dark: str = "#0D3B66"
    primary_light: str = "#3B82F6"

    secondary: str = "#0F8B8D"
    secondary_light: str = "#14B8A6"

    accent: str = "#6C63FF"
    accent_light: str = "#A78BFA"

    success: str = "#2E8B57"
    success_light: str = "#10B981"

    warning: str = "#D99000"
    warning_light: str = "#F59E0B"

    danger: str = "#C73E3A"
    danger_light: str = "#EF4444"

    border: str = "#E5E7EB"
    border_dark: str = "#2D3A52"

    muted: str = "#9CA3AF"


light_palette = ColorPalette()


# ── TYPOGRAPHY ──
@dataclass
class Typography:
    """Type hierarchy. Modern professional sans-serif (Inter preferred)."""

    font_family: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    size_xs: int = 12
    size_sm: int = 13
    size_base: int = 14
    size_md: int = 16
    size_lg: int = 18
    size_xl: int = 24
    size_2xl: int = 28
    size_3xl: int = 32
    size_4xl: int = 36
    weight_normal: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700
    line_height_tight: float = 1.25
    line_height_normal: float = 1.6
    line_height_relaxed: float = 1.8


typography = Typography()


# ── SPACING ──
@dataclass
class Spacing:
    """Modular spacing scale."""

    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32
    xxxl: int = 48


spacing = Spacing()


# ── COMPONENT TOKENS ──
@dataclass
class ComponentTokens:
    """Reusable component styling constants."""

    border_width: str = "1px"
    border_radius_sm: str = "6px"
    border_radius_md: str = "8px"
    border_radius_lg: str = "12px"
    shadow_sm: str = "0 1px 2px rgba(15, 23, 42, 0.06)"
    shadow_md: str = "0 8px 24px rgba(15, 23, 42, 0.08)"
    shadow_lg: str = "0 18px 38px rgba(15, 23, 42, 0.12)"
    transition_fast: str = "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)"
    transition_normal: str = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
    button_height_sm: str = "32px"
    button_height_md: str = "40px"
    button_height_lg: str = "48px"
    input_height: str = "40px"
    input_padding: str = "10px 14px"


components = ComponentTokens()


def _theme_css(dark: bool = False) -> str:
    p = light_palette
    t = typography
    s = spacing
    c = components

    if dark:
        bg_primary = p.bg_dark
        bg_secondary = p.bg_dark_secondary
        bg_tertiary = "#1F2937"
        text_primary = p.text_light
        text_secondary = "#AAB6C8"
        primary = p.primary_light
        secondary = p.secondary_light
        accent = p.accent_light
        success = p.success
        warning = p.warning_light
        danger = p.danger_light
        border = p.border_dark
        muted = "#6B7280"
        brand_gradient = "linear-gradient(135deg, #60A5FA, #A78BFA)"
        card_shadow = "0 16px 32px rgba(0, 0, 0, 0.22)"
    else:
        bg_primary = p.bg_primary
        bg_secondary = p.bg_secondary
        bg_tertiary = p.bg_tertiary
        text_primary = p.text_primary
        text_secondary = p.text_secondary
        primary = p.primary
        secondary = p.secondary
        accent = p.accent
        success = p.success
        warning = p.warning
        danger = p.danger
        border = p.border
        muted = p.muted
        brand_gradient = "linear-gradient(135deg, #1769AA, #6C63FF)"
        card_shadow = c.shadow_md

    return f"""
    <style>
        :root {{
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-tertiary: {bg_tertiary};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --primary: {primary};
            --primary-dark: {p.primary_dark};
            --secondary: {secondary};
            --accent: {accent};
            --success: {success};
            --warning: {warning};
            --danger: {danger};
            --border: {border};
            --muted: {muted};
            --font-family: {t.font_family};
            --radius-sm: {c.border_radius_sm};
            --radius-md: {c.border_radius_md};
            --radius-lg: {c.border_radius_lg};
            --shadow-sm: {c.shadow_sm};
            --shadow-md: {card_shadow};
            --shadow-lg: {c.shadow_lg};
        }}

        * {{ box-sizing: border-box; }}

        body, .stApp {{
            font-family: var(--font-family);
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: {t.size_base}px;
            line-height: {t.line_height_normal};
        }}

        .block-container {{
            padding-top: {s.lg}px;
            padding-bottom: {s.xxl}px;
            max-width: 1600px;
        }}

        h1, h2, h3, h4 {{ color: var(--text-primary); }}
        p, li, label {{ color: var(--text-primary); }}

        .stMarkdown p {{ line-height: {t.line_height_normal}; }}

        .stButton > button {{
            font-family: var(--font-family);
            font-weight: {t.weight_semibold};
            border-radius: {c.border_radius_md};
            border: 1px solid transparent;
            min-height: {c.button_height_md};
            transition: {c.transition_fast};
            box-shadow: none;
        }}

        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
        }}

        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(23, 105, 170, 0.18);
        }}

        .stButton > button[kind="secondary"] {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            border-color: var(--border);
        }}

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"],
        .stNumberInput input {{
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: {c.border_radius_md} !important;
        }}

        .stTextArea textarea {{ min-height: 120px; }}

        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(23, 105, 170, 0.12) !important;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: {s.lg}px;
            padding-bottom: {s.lg}px;
        }}

        section[data-testid="stSidebar"] .stRadio > div {{ gap: {s.sm}px; }}
        section[data-testid="stSidebar"] .stRadio label {{ width: 100%; }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: {c.border_radius_md};
            padding: 10px 12px;
            transition: {c.transition_fast};
            color: var(--text-primary);
            margin-bottom: 6px;
        }}

        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
            background: var(--bg-tertiary);
            border-color: var(--border);
        }}

        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {{
            background: linear-gradient(135deg, rgba(23, 105, 170, 0.10), rgba(108, 99, 255, 0.08));
            border-color: rgba(23, 105, 170, 0.25);
            box-shadow: inset 0 0 0 1px rgba(23, 105, 170, 0.08);
        }}

        .sidebar-brand {{
            padding: 18px 18px 16px 18px;
            border: 1px solid var(--border);
            border-radius: {c.border_radius_lg};
            background: linear-gradient(180deg, var(--bg-secondary), var(--bg-tertiary));
            margin-bottom: 12px;
            box-shadow: var(--shadow-sm);
        }}

        .sidebar-brand-title {{
            font-size: {t.size_xl}px;
            font-weight: {t.weight_bold};
            background: {brand_gradient};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 6px;
        }}

        .sidebar-brand-subtitle {{
            font-size: {t.size_sm}px;
            color: var(--text-secondary);
            line-height: {t.line_height_normal};
        }}

        .page-header {{
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
            border: 1px solid var(--border);
            border-radius: {c.border_radius_lg};
            padding: 22px 24px;
            margin-bottom: 20px;
            box-shadow: var(--shadow-sm);
        }}

        .page-header-title {{
            font-size: {t.size_2xl}px;
            font-weight: {t.weight_bold};
            color: var(--text-primary);
            letter-spacing: 0.01em;
            margin-bottom: 4px;
        }}

        .page-header-subtitle {{
            color: var(--text-secondary);
            font-size: {t.size_sm}px;
            max-width: 900px;
        }}

        .page-header-corpus {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-top: 12px;
            background: rgba(23, 105, 170, 0.08);
            color: var(--primary);
            border: 1px solid rgba(23, 105, 170, 0.14);
            border-radius: 999px;
            padding: 5px 12px;
            font-size: {t.size_xs}px;
            font-weight: {t.weight_semibold};
        }}

        .metric-card, .quick-card, .evidence-card, .gap-card, .hypothesis-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: {c.border_radius_lg};
            box-shadow: var(--shadow-sm);
        }}

        .metric-card {{ padding: 16px 18px; }}
        .metric-card-label {{ color: var(--text-secondary); font-size: {t.size_xs}px; text-transform: uppercase; letter-spacing: 0.08em; }}
        .metric-card-value {{ color: var(--text-primary); font-size: {t.size_xl}px; font-weight: {t.weight_bold}; margin-top: 6px; }}

        .quick-card {{
            padding: 16px 18px;
            min-height: 148px;
            transition: {c.transition_fast};
        }}

        .quick-card:hover,
        .evidence-card:hover,
        .gap-card:hover,
        .hypothesis-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        .quick-card-kicker {{
            font-size: {t.size_xs}px;
            color: var(--primary);
            font-weight: {t.weight_semibold};
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .quick-card-title {{
            font-size: {t.size_lg}px;
            font-weight: {t.weight_semibold};
            color: var(--text-primary);
            margin-bottom: 8px;
        }}

        .quick-card-text {{ color: var(--text-secondary); font-size: {t.size_sm}px; }}

        .recent-analysis-card {{
            padding: 16px 18px;
            background: linear-gradient(135deg, rgba(23, 105, 170, 0.08), rgba(15, 139, 141, 0.06));
            border: 1px solid rgba(23, 105, 170, 0.14);
            border-radius: {c.border_radius_lg};
            margin-top: 16px;
        }}

        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            font-size: {t.size_sm}px;
            font-weight: {t.weight_semibold};
            margin: 8px 0 14px 0;
        }}

        .status-indicator.ready {{ color: var(--primary); }}
        .status-indicator.running {{ color: var(--warning); }}
        .status-indicator.complete {{ color: var(--success); }}

        .pipeline-progress {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 18px 0;
        }}

        .pipeline-step {{
            padding: 12px 14px;
            border-radius: {c.border_radius_md};
            border: 1px solid var(--border);
            background: var(--bg-secondary);
        }}

        .pipeline-step-label {{
            font-size: {t.size_xs}px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}

        .pipeline-step-text {{ font-size: {t.size_sm}px; font-weight: {t.weight_semibold}; }}
        .pipeline-step.complete {{ border-color: rgba(46, 139, 87, 0.28); background: rgba(46, 139, 87, 0.08); }}
        .pipeline-step.active {{ border-color: rgba(217, 144, 0, 0.28); background: rgba(217, 144, 0, 0.08); }}
        .pipeline-step.pending {{ opacity: 0.75; }}

        .evidence-card, .gap-card, .hypothesis-card {{ padding: 18px 20px; margin-bottom: 12px; }}
        .evidence-card-header, .gap-card-header, .hypothesis-card-header {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }}

        .evidence-card-title, .gap-card-title, .hypothesis-card-title {{
            font-size: {t.size_lg}px;
            font-weight: {t.weight_semibold};
            color: var(--text-primary);
        }}

        .evidence-card-text, .gap-card-text, .hypothesis-card-text {{
            color: var(--text-secondary);
            font-size: {t.size_sm}px;
            line-height: {t.line_height_normal};
        }}

        .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }}

        .soft-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: {t.size_xs}px;
            font-weight: {t.weight_semibold};
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .gap-card {{ border-left: 4px solid var(--accent); }}
        .hypothesis-card {{ border-left: 4px solid var(--primary); }}

        .score-block {{ margin: 6px 0; }}
        .score-block .score-label {{ font-size: {t.size_xs}px; color: var(--text-secondary); }}
        .score-block .score-bar {{ font-family: monospace; color: var(--primary); letter-spacing: 1px; }}

        .empty-state {{
            text-align: center;
            padding: 48px 24px;
            background: var(--bg-tertiary);
            border-radius: {c.border_radius_lg};
            border: 1px dashed var(--border);
        }}

        .stAlert, [data-testid="metric-container"], .stDataFrame {{
            border-radius: {c.border_radius_md};
        }}

        .stExpander {{ border: 1px solid var(--border); border-radius: {c.border_radius_md}; overflow: hidden; }}
        .stTabs [data-baseweb="tab"] {{ color: var(--text-secondary); font-weight: {t.weight_medium}; }}
        .stTabs [aria-selected="true"] {{ color: var(--primary); border-bottom-color: var(--primary); }}

        .sidebar-corpus-summary {{
            margin-top: 16px;
            padding: 14px;
            border-radius: {c.border_radius_md};
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            font-size: {t.size_sm}px;
            color: var(--text-secondary);
        }}

        .sidebar-divider {{ margin: 18px 0 12px 0; border-top: 1px solid var(--border); }}

        a {{ color: var(--primary); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
    """


def get_light_theme_css() -> str:
    """Generate complete light-mode CSS."""
    return _theme_css(dark=False)


def get_dark_theme_css() -> str:
    """Generate complete dark-mode CSS."""
    return _theme_css(dark=True)


def get_page_header_html(title: str, subtitle: str, corpus_info: Optional[str] = None) -> str:
    corpus_html = ""
    if corpus_info:
        corpus_html = (
            f'<div class="page-header-corpus">🧪 {escape_html(corpus_info)}</div>'
        )
    return f"""
    <div class="page-header">
        <div class="page-header-title">{escape_html(title)}</div>
        <div class="page-header-subtitle">{escape_html(subtitle)}</div>
        {corpus_html}
    </div>
    """


def get_score_bar_html(label: str, score: float, show_level: bool = True) -> str:
    score = max(0.0, min(1.0, float(score or 0.0)))
    filled = round(score * 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    level = "HIGH" if score >= 0.7 else "MODERATE" if score >= 0.4 else "LOW"
    color = "#2E8B57" if score >= 0.7 else "#D99000" if score >= 0.4 else "#C73E3A"
    level_html = f'<span style="font-size:11px;color:{color};font-weight:600;">{level}</span>' if show_level else ""
    return f'''<div style="margin:6px 0;">
        <span style="font-size:12px;color:#53657A;">{escape_html(label)}</span><br>
        <span style="font-family:monospace;color:#1769AA;letter-spacing:1px;">{bar}</span>
        <span style="font-size:13px;color:#14213D;margin-left:8px;">{score:.2f}</span>
        {level_html}
    </div>'''


def get_empty_state_html(icon: str, title: str, message: str, action: str = "") -> str:
    action_html = f'<div style="margin-top:12px;font-size:13px;color:#1769AA;">{escape_html(action)}</div>' if action else ""
    return f'''<div style="text-align:center;padding:48px 24px;background:#F0F4F8;
               border-radius:12px;border:1px dashed #E5E7EB;">
        <div style="font-size:32px;margin-bottom:12px;opacity:0.6;">{escape_html(icon)}</div>
        <div style="font-size:18px;font-weight:600;color:#14213D;margin-bottom:8px;">{escape_html(title)}</div>
        <div style="font-size:13px;color:#53657A;max-width:400px;margin:0 auto;">{escape_html(message)}</div>
        {action_html}
    </div>'''


def get_polarity_badge_html(polarity: str) -> str:
    polarity = (polarity or "uncertain").lower()
    config = {
        "positive": ("#2E8B57", "rgba(46,139,87,0.1)", "POSITIVE"),
        "negative": ("#C73E3A", "rgba(199,62,58,0.1)", "NEGATIVE"),
        "uncertain": ("#D99000", "rgba(217,144,0,0.1)", "UNCERTAIN"),
        "speculative": ("#6C63FF", "rgba(108,99,255,0.1)", "SPECULATIVE"),
    }.get(polarity, ("#53657A", "rgba(83,101,122,0.1)", polarity.upper()))
    color, bg, label = config
    return f'<span style="background:{bg};color:{color};border:1px solid {color};border-radius:999px;padding:2px 10px;font-size:11px;font-weight:600;">{escape_html(label)}</span>'


def get_evidence_card_html(subject, predicate, obj, sentence, pmid, year, confidence, polarity) -> str:
    conf_str = f"{float(confidence):.2f}" if confidence is not None else "N/A"
    year_str = str(year) if year else "N/A"
    pmid_str = str(pmid) if pmid else "N/A"
    polarity_html = get_polarity_badge_html(str(polarity or "uncertain"))
    return f'''<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;padding:14px 18px;margin:8px 0;transition:box-shadow 0.15s;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <span style="font-weight:600;color:#14213D;">{escape_html(str(subject or 'N/A'))}</span>
                <span style="color:#0F8B8D;font-size:12px;background:rgba(15,139,141,0.1);padding:2px 8px;border-radius:4px;">{escape_html(str(predicate or 'related_to'))}</span>
                <span style="font-weight:600;color:#14213D;">{escape_html(str(obj or 'N/A'))}</span>
            </div>
            {polarity_html}
        </div>
        <div style="font-size:13px;color:#53657A;margin-bottom:10px;line-height:1.6;">{escape_html(str(sentence or 'No evidence sentence available.'))}</div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
            <span style="font-size:11px;background:rgba(23,105,170,0.1);color:#1769AA;padding:2px 8px;border-radius:999px;">PMID {escape_html(pmid_str)}</span>
            <span style="font-size:11px;color:#53657A;">{escape_html(year_str)}</span>
            <span style="font-size:11px;color:#53657A;">Confidence: {escape_html(conf_str)}</span>
        </div>
    </div>'''
