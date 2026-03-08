"""
HTML component templates for the Churn Prediction Streamlit App.
Keep all HTML strings here to maintain separation from app logic.
"""


def brand_logo() -> str:
    return """
    <div class="brand">
        <div class="dot"></div>
        ChurnGuard AI
    </div>
    """


def api_status_badge(is_online: bool) -> str:
    status = "online" if is_online else "offline"
    label = "API ONLINE" if is_online else "API OFFLINE"
    return f"""
    <div class="api-badge {status}">
        <div class="dot"></div>
        {label}
    </div>
    """


def stat_card(label: str, value: str, icon: str) -> str:
    return f"""
    <div class="stat-card">
        <div>
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        <div class="icon">{icon}</div>
    </div>
    """


def prediction_result_card(will_churn: bool, probability: float, risk_level: str) -> str:
    churn_text = "Will Churn: Yes" if will_churn else "Will Churn: No"
    risk_class = risk_level.lower()
    risk_color = {
        "high": "#FF4444",
        "medium": "#FF9900",
        "low": "#00FF94"
    }.get(risk_class, "#FFFFFF")

    risk_icon = {
        "high": "⚠️",
        "medium": "⚡",
        "low": "✅"
    }.get(risk_class, "")

    return f"""
    <div class="result-card {risk_class}">
        <div class="result-title">Prediction Result</div>
        <div class="result-main">{churn_text}</div>
        <div class="result-row">
            Probability: <span>{probability:.0%}</span>
        </div>
        <div class="result-row">
            Risk Level: <span style="color: {risk_color};">
                {risk_level.upper()} {risk_icon}
            </span>
        </div>
    </div>
    """


def section_box_start(title: str) -> str:
    return f"""
    <div class="section-box">
        <div class="section-title">{title}</div>
    """


def section_box_end() -> str:
    return "</div>"


def page_header(title: str, subtitle: str = "") -> str:
    sub = f'<p style="color:#666680; font-size:14px; margin-top:-12px;">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin-bottom: 28px;">
        <h1 style="margin-bottom: 4px;">{title}</h1>
        {sub}
    </div>
    """


def churn_badge(will_churn: bool, probability: float) -> str:
    if will_churn:
        color = "#FF4444"
        bg = "#FF444420"
        label = f"High ({probability:.0%})"
    elif probability > 0.4:
        color = "#FF9900"
        bg = "#FF990020"
        label = f"Medium ({probability:.0%})"
    else:
        color = "#00FF94"
        bg = "#00FF9420"
        label = f"Low ({probability:.0%})"

    return f"""
    <span style="
        background: {bg};
        color: {color};
        border: 1px solid {color}40;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    ">{label}</span>
    """


def empty_state(message: str, icon: str = "📭") -> str:
    return f"""
    <div style="
        text-align: center;
        padding: 60px 20px;
        color: #444460;
        background: #0D0D1A;
        border: 1px dashed #1A1A2E;
        border-radius: 10px;
        margin-top: 16px;
    ">
        <div style="font-size: 40px; margin-bottom: 12px;">{icon}</div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 13px;">
            {message}
        </div>
    </div>
    """
