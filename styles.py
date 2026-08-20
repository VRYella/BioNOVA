# ────────────────────────────────────────────────────────────────────────────────────
# BioNOVA Design System — Centralized styling, theme, and component definitions
# ────────────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Literal

# ── PALETTE: Scientific, restrained, WCAG-accessible colors ──
@dataclass
class ColorPalette:
    """Primary color system. All values WCAG-AA compliant."""
    bg_primary: str = "#F6F8FB"  # Very light neutral/blue-gray
    bg_secondary: str = "#FFFFFF"  # Surface
    bg_tertiary: str = "#F0F4F8"  # Surface alternative
    bg_dark: str = "#0A0F1E"  # Dark mode primary
    bg_dark_secondary: str = "#111827"  # Dark surface
    
    text_primary: str = "#14213D"  # Deep navy
    text_secondary: str = "#53657A"  # Muted blue-gray
    text_light: str = "#F9FAFB"  # Light for dark mode
    
    primary: str = "#1769AA"  # Deep scientific blue
    primary_dark: str = "#0D3B66"  # Darker scientific blue
    primary_light: str = "#3B82F6"  # Lighter scientific blue
    
    secondary: str = "#0F8B8D"  # Teal
    secondary_light: str = "#14B8A6"  # Light teal
    
    accent: str = "#6C63FF"  # Violet for AI/discovery
    accent_light: str = "#A78BFA"  # Light violet
    
    success: str = "#2E8B57"  # Scientific green
    success_light: str = "#10B981"  # Light green
    
    warning: str = "#D99000"  # Amber
    warning_light: str = "#F59E0B"  # Light amber
    
    danger: str = "#C73E3A"  # Red
    danger_light: str = "#EF4444"  # Light red
    
    border: str = "#E5E7EB"  # Light gray border
    border_dark: str = "#2D3A52"  # Dark mode border
    
    muted: str = "#9CA3AF"  # Muted gray
    
light_palette = ColorPalette()

# ── TYPOGRAPHY ──
@dataclass
class Typography:
    """Type hierarchy. Modern professional sans-serif (Inter preferred)."""
    font_family: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    
    # Sizes (px)
    size_xs: int = 12
    size_sm: int = 13
    size_base: int = 14
    size_md: int = 16
    size_lg: int = 18
    size_xl: int = 24
    size_2xl: int = 28
    size_3xl: int = 32
    size_4xl: int = 36
    
    # Weights
    weight_normal: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700
    
    # Line heights
    line_height_tight: float = 1.4
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
    # Borders
    border_width: str = "1px"
    border_radius_sm: str = "6px"
    border_radius_md: str = "8px"
    border_radius_lg: str = "12px"
    
    # Shadows (subtle)
    shadow_sm: str = "0 1px 2px rgba(0, 0, 0, 0.05)"
    shadow_md: str = "0 4px 6px rgba(0, 0, 0, 0.07)"
    shadow_lg: str = "0 10px 15px rgba(0, 0, 0, 0.1)"
    
    # Transitions
    transition_fast: str = "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)"
    transition_normal: str = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
    
    # Button sizes
    button_height_sm: str = "32px"
    button_height_md: str = "40px"
    button_height_lg: str = "48px"
    
    # Input sizes
    input_height: str = "40px"
    input_padding: str = "10px 14px"

components = ComponentTokens()

def get_light_theme_css() -> str:
    """Generate complete light-mode CSS."""
    p = light_palette
    t = typography
    s = spacing
    c = components
    
    return f"""
    <style>
        :root {{
            --bg-primary: {p.bg_primary};
            --bg-secondary: {p.bg_secondary};
            --bg-tertiary: {p.bg_tertiary};
            --text-primary: {p.text_primary};
            --text-secondary: {p.text_secondary};
            --primary: {p.primary};
            --primary-dark: {p.primary_dark};
            --secondary: {p.secondary};
            --accent: {p.accent};
            --success: {p.success_light};
            --warning: {p.warning_light};
            --danger: {p.danger_light};
            --border: {p.border};
            --muted: {p.muted};
            --font-family: {t.font_family};
            --size-base: {t.size_base}px;
            --radius-md: {c.border_radius_md};
            --shadow-md: {c.shadow_md};
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body, .stApp {{
            font-family: var(--font-family);
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-size: var(--size-base);
            line-height: {t.line_height_normal};
        }}
        
        .block-container {{ padding-top: {s.lg}px; padding-bottom: {s.xl}px; max-width: 1600px; }}
        
        /* Headings */
        h1, .h1 {{
            font-size: {t.size_3xl}px;
            font-weight: {t.weight_bold};
            line-height: {t.line_height_tight};
            color: var(--text-primary);
            margin-bottom: {s.lg}px;
        }}
        
        h2, .h2 {{
            font-size: {t.size_2xl}px;
            font-weight: {t.weight_bold};
            line-height: {t.line_height_tight};
            color: var(--text-primary);
            margin-bottom: {s.md}px;
        }}
        
        h3, .h3 {{
            font-size: {t.size_lg}px;
            font-weight: {t.weight_semibold};
            line-height: {t.line_height_normal};
            color: var(--text-primary);
            margin-bottom: {s.sm}px;
        }}
        
        /* Buttons */
        .stButton > button {{
            font-family: var(--font-family);
            font-weight: {t.weight_semibold};
            border-radius: {c.border_radius_md};
            border: none;
            height: {c.button_height_md};
            transition: {c.transition_fast};
            text-transform: none;
            letter-spacing: 0;
        }}
        
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            box-shadow: 0 2px 8px rgba(23, 105, 170, 0.2);
        }}
        
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(23, 105, 170, 0.3);
        }}
        
        .stButton > button[kind="secondary"] {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }}
        
        /* Inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div > input {{
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: {c.border_radius_md};
            font-family: var(--font-family);
            font-size: {t.size_base}px;
            padding: {c.input_padding};
            transition: {c.transition_fast};
        }}
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(23, 105, 170, 0.1);
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border);
        }}
        
        section[data-testid="stSidebar"] .stRadio > div {{
            gap: {s.sm}px;
        }}
        
        section[data-testid="stSidebar"] .stRadio label {{
            font-weight: {t.weight_medium};
            color: var(--text-primary);
        }}
        
        /* Cards & Containers */
        .metric-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: {c.border_radius_lg};
            padding: {s.lg}px;
            box-shadow: {c.shadow_sm};
            transition: {c.transition_fast};
        }}
        
        .metric-card:hover {{
            box-shadow: {c.shadow_md};
            transform: translateY(-2px);
        }}
        
        .score-indicator {{
            display: inline-flex;
            align-items: center;
            gap: {s.sm}px;
            padding: {s.xs}px {s.md}px;
            background: var(--bg-tertiary);
            border-radius: 999px;
            font-size: {t.size_sm}px;
            font-weight: {t.weight_semibold};
            color: var(--text-primary);
            border: 1px solid var(--border);
        }}
        
        .score-high {{ color: var(--success); border-color: var(--success); }}
        .score-medium {{ color: var(--warning); border-color: var(--warning); }}
        .score-low {{ color: var(--danger); border-color: var(--danger); }}
        
        /* Evidence & Pills */
        .evidence-pill {{
            display: inline-block;
            background: rgba(23, 105, 170, 0.1);
            color: var(--primary);
            border-radius: 999px;
            padding: {s.xs}px {s.md}px;
            font-size: {t.size_xs}px;
            font-weight: {t.weight_medium};
            margin: {s.xs}px {s.xs}px 0 0;
            border: 1px solid rgba(23, 105, 170, 0.2);
        }}
        
        .polarity-positive {{ background: rgba(46, 139, 87, 0.1); color: var(--success); border-color: var(--success); }}
        .polarity-negative {{ background: rgba(199, 62, 58, 0.1); color: var(--danger); border-color: var(--danger); }}
        .polarity-uncertain {{ background: rgba(217, 144, 0, 0.1); color: var(--warning); border-color: var(--warning); }}
        
        /* Divider */
        .stDivider {{ border-color: var(--border); opacity: 0.6; }}
        
        /* Data table */
        .stDataFrame {{
            border: 1px solid var(--border);
            border-radius: {c.border_radius_md};
            overflow: hidden;
        }}
        
        /* Empty state */
        .empty-state {{
            text-align: center;
            padding: {s.xxxl}px {s.lg}px;
            background: var(--bg-tertiary);
            border-radius: {c.border_radius_lg};
            border: 1px dashed var(--border);
        }}
        
        .empty-state-icon {{
            font-size: {t.size_2xl}px;
            margin-bottom: {s.md}px;
            opacity: 0.6;
        }}
        
        .empty-state-title {{
            font-size: {t.size_lg}px;
            font-weight: {t.weight_semibold};
            color: var(--text-primary);
            margin-bottom: {s.sm}px;
        }}
        
        .empty-state-text {{
            font-size: {t.size_sm}px;
            color: var(--text-secondary);
            max-width: 400px;
            margin: 0 auto;
            line-height: {t.line_height_relaxed};
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: {c.border_radius_md};
            font-weight: {t.weight_medium};
        }}
        
        .streamlit-expanderHeader:hover {{
            background: #FFFBF0;
        }}
        
        /* Success/Warning/Error messages */
        .stAlert {{
            border-radius: {c.border_radius_md};
            border-left: 4px solid var(--primary);
        }}
        
        .stSuccess {{ border-left-color: var(--success); }}
        .stWarning {{ border-left-color: var(--warning); }}
        .stError {{ border-left-color: var(--danger); }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab"] {{
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            font-weight: {t.weight_medium};
            transition: {c.transition_fast};
        }}
        
        .stTabs [aria-selected="true"] {{
            border-bottom-color: var(--primary);
            color: var(--primary);
        }}
        
        /* Slider */
        .stSlider > div > div > div > div {{
            color: var(--primary);
        }}
        
        /* Metric */
        [data-testid="metric-container"] {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: {c.border_radius_lg};
            padding: {s.md}px;
        }}
        
        /* Global text utilities */
        .text-muted {{ color: var(--text-secondary); }}
        .text-accent {{ color: var(--accent); }}
        .font-mono {{ font-family: 'Monaco', 'Courier New', monospace; }}
        
        /* Layout utilities */
        .flex-center {{ display: flex; align-items: center; justify-content: center; }}
        .gap-sm {{ gap: {s.sm}px; }}
        .gap-md {{ gap: {s.md}px; }}
        .gap-lg {{ gap: {s.lg}px; }}
    </style>
    """

def get_dark_theme_css() -> str:
    """Generate complete dark-mode CSS."""
    p = light_palette
    t = typography
    s = spacing
    c = components
    
    return f"""
    <style>
        :root {{
            --bg-primary: {p.bg_dark};
            --bg-secondary: {p.bg_dark_secondary};
            --bg-tertiary: #1F2937;
            --text-primary: {p.text_light};
            --text-secondary: #9CA3AF;
            --primary: {p.primary_light};
            --primary-dark: {p.primary_dark};
            --secondary: {p.secondary_light};
            --accent: {p.accent_light};
            --success: {p.success_light};
            --warning: {p.warning_light};
            --danger: {p.danger_light};
            --border: {p.border_dark};
            --muted: #6B7280;
        }}
        
        body, .stApp {{
            font-family: {t.font_family};
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-size: {t.size_base}px;
            line-height: {t.line_height_normal};
        }}
        
        .block-container {{ padding-top: {s.lg}px; padding-bottom: {s.xl}px; }}
        
        /* Headings */
        h1, h2, h3 {{ color: var(--text-primary); }}
        
        /* Inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {{
            background-color: {p.bg_dark_secondary};
            color: var(--text-light);
            border-color: {p.border_dark};
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {p.bg_dark_secondary};
            border-right-color: {p.border_dark};
        }}
        
        /* Cards */
        .metric-card {{
            background: {p.bg_dark_secondary};
            border-color: {p.border_dark};
        }}
        
        .stAlert {{
            background-color: rgba(0, 0, 0, 0.3);
            color: var(--text-light);
        }}
    </style>
    """
