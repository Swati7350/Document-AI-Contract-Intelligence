import streamlit as st
import streamlit.components.v1 as components
import json


def render() -> None:
    result = st.session_state.get("ocr_result")
    if not result:
        st.warning("No OCR results yet. Please upload and process a document first.")
        if st.button("← Go to Upload"):
            st.session_state.page = "upload"; st.rerun()
        return

    st.markdown(f"""
<div class="sec-header">
  <div class="sec-title">🔍 OCR & Layout Results</div>
  <div class="sec-sub">Engine: <strong>{result['engine'].upper()}</strong> &nbsp;·&nbsp;
  Confidence: <strong>{result['avg_confidence']*100:.1f}%</strong> &nbsp;·&nbsp;
  Pages: <strong>{result['layout']['pages']}</strong> &nbsp;·&nbsp;
  Time: <strong>{result['processing_time_ms']}ms</strong></div>
</div>""", unsafe_allow_html=True)

    # KPI row
    lay = result["layout"]
    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-val blue">{len(result['text_blocks'])}</div><div class="kpi-lbl">Text Blocks</div></div>
  <div class="kpi"><div class="kpi-val blue">{len(result['tables'])}</div><div class="kpi-lbl">Tables Detected</div></div>
  <div class="kpi"><div class="kpi-val blue">{len(lay['stamps_seals'])}</div><div class="kpi-lbl">Stamps / Seals</div></div>
  <div class="kpi"><div class="kpi-val blue">{lay['signatures_count']}</div><div class="kpi-lbl">Signatures</div></div>
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Blocks", "📊 Tables", "🗺️ Layout Map", "🔲 Bounding Boxes"])

    with tab1:
        for blk in result["text_blocks"]:
            badge_map = {"heading":"badge-blue","paragraph":"badge-green","signature":"badge-purple","table":"badge-orange"}
            badge_cls = badge_map.get(blk["type"], "badge-blue")
            conf_pct  = int(blk["confidence"] * 100)
            st.markdown(f"""
<div class="card" style="margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span class="badge {badge_cls}">{blk['type'].title()}</span>
    <span style="font-size:.72rem;color:#94a3b8;margin-left:auto;">Confidence: {conf_pct}%</span>
  </div>
  <div style="font-size:.875rem;color:#0f172a;line-height:1.65;">{blk['text']}</div>
  <div class="prog-wrap"><div class="prog-fill" style="width:{conf_pct}%"></div></div>
</div>""", unsafe_allow_html=True)

    with tab2:
        for tbl in result["tables"]:
            st.markdown(f'<div style="font-weight:700;color:#0f172a;margin-bottom:10px;">📊 {tbl["title"]}</div>', unsafe_allow_html=True)
            header_html = "".join(f"<th>{h}</th>" for h in tbl["headers"])
            rows_html   = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                for row in tbl["rows"]
            )
            st.markdown(f'<table class="tbl"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        layout_items = [
            ("Pages detected",         str(lay["pages"])),
            ("Column layout",          f"{lay['columns_detected']}-column"),
            ("Headers / Footers",      "✓ Detected" if lay["headers_footers"] else "✗ Not found"),
            ("Tables",                 str(lay["tables_count"])),
            ("Signatures",             str(lay["signatures_count"])),
        ]
        for label, val in layout_items:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9;font-size:.875rem;"><span style="color:#64748b;">{label}</span><span style="font-weight:600;color:#0f172a;">{val}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if lay["stamps_seals"]:
            st.markdown('<div style="font-weight:600;color:#0f172a;margin:14px 0 8px;">🔏 Stamps & Seals</div>', unsafe_allow_html=True)
            for s in lay["stamps_seals"]:
                st.markdown(f'<div class="badge badge-orange" style="margin-bottom:6px;">🔏 {s}</div>', unsafe_allow_html=True)

    with tab4:
        boxes = lay.get("bounding_boxes", [])
        # Render an SVG document mock with overlaid bounding boxes
        box_svgs = ""
        for b in boxes:
            x = b["x"] * 560; y = b["y"] * 720
            w = b["w"] * 560; h = b["h"] * 720
            color = b["color"]
            box_svgs += f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" fill="{color}22" stroke="{color}" stroke-width="2" rx="4"/>'
            box_svgs += f'<text x="{x+4:.0f}" y="{y-5:.0f}" font-size="11" fill="{color}" font-weight="bold">{b["label"]}</text>'

        svg_html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f8fafc;">
<svg width="100%" viewBox="0 0 560 720" style="border:1px solid #e2e8f0;border-radius:12px;background:white;">
  <rect width="560" height="720" fill="white"/>
  <!-- Page lines -->
  {''.join(f'<line x1="44" y1="{70+i*22}" x2="516" y2="{70+i*22}" stroke="#f1f5f9" stroke-width="1"/>' for i in range(28))}
  {box_svgs}
  <text x="280" y="750" text-anchor="middle" font-size="11" fill="#94a3b8">Document page — bounding box overlay</text>
</svg>
</body></html>"""
        components.html(svg_html, height=480, scrolling=False)

        # Legend
        legend_html = " &nbsp;".join(f'<span class="badge" style="background:{b["color"]}22;color:{b["color"]};border:1px solid {b["color"]}55;">{b["label"]}</span>' for b in boxes)
        st.markdown(f'<div style="margin-top:10px;">{legend_html}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧩  Run Structured Extraction →", type="primary", use_container_width=True):
            st.session_state.page = "extraction"; st.rerun()
    with c2:
        if st.button("💬  Go to Contract RAG →", type="secondary", use_container_width=True):
            st.session_state.page = "rag"; st.rerun()
