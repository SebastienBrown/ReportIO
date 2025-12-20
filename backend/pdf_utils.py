from fpdf import FPDF
from datetime import datetime
import re
import io

class ReportPDF(FPDF):
    # Professional Color Palette
    PRIMARY_COLOR = (79, 70, 229)    # Indigo 600
    SECONDARY_COLOR = (51, 65, 85)   # Slate 700
    TEXT_COLOR = (15, 23, 42)        # Slate 900
    MUTED_COLOR = (100, 116, 139)    # Slate 500
    LINK_COLOR = (37, 99, 235)       # Blue 600
    BORDER_COLOR = (226, 232, 240)   # Slate 200

    def __init__(self):
        super().__init__()
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "B", 8)
        self.set_text_color(*self.MUTED_COLOR)
        self.cell(self.epw, 6, "Research Report | ReportIO", border="B", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*self.MUTED_COLOR)
        self.cell(self.epw, 8, f"Page {self.page_no()}", align="C")
        self.set_x(20)
        self.cell(0, 8, "ReportIO Intelligence Service", align="L")

    def draw_section_divider(self):
        self.ln(2)
        curr_y = self.get_y()
        self.set_draw_color(*self.PRIMARY_COLOR)
        self.set_line_width(0.4)
        self.line(20, curr_y, 190, curr_y)
        self.ln(4)

    def section_header(self, label):
        self.ln(3)
        self.set_font("helvetica", "B", 13)
        self.set_text_color(*self.PRIMARY_COLOR)
        self.cell(self.epw, 8, label.upper(), ln=1)
        self.draw_section_divider()

    def item_title(self, number, text):
        # Standard color for titles
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*self.TEXT_COLOR) 
        title_text = f"{number}. {text}"
        # Use markdown for titles (bold/italic supported)
        self.multi_cell(self.epw, 6, clean_text(title_text, mode="markdown"), markdown=True)
        self.ln(0.5)

    def write_styled_content(self, text):
        # Helper to write HTML content for blue link support
        html_text = clean_text(text, mode="html")
        self.write_html(html_text)
        self.ln(2)

    def separator(self):
        self.ln(2)
        self.set_draw_color(*self.BORDER_COLOR)
        self.set_line_width(0.1)
        curr_y = self.get_y()
        self.line(30, curr_y, 180, curr_y)
        self.ln(3)

def clean_text(text, mode="markdown"):
    """
    mode can be: 
    - "markdown": for fpdf2's multi_cell(markdown=True)
    - "html": for fpdf2's write_html() -> supports blue links
    - "plain": for stripped output
    """
    if not text:
        return ""
    
    # Standard Unicode Substitutions
    substitutions = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", 
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", 
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', 
        "\u2022": "-", "\u2023": "-", "\u2024": "-", 
        "\u2026": "...",
        "\u00a0": " ", 
    }
    for char, replacement in substitutions.items():
        text = text.replace(char, replacement)

    text = text.encode("latin-1", "replace").decode("latin-1")
    text = text.replace("?", " ")

    if mode == "html":
        # Convert Markdown to HTML for blue link support
        # Bold **x** -> <b>x</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Italic *x* -> <i>x</i>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        # Citation Links [[1]](url) or [1](url) -> <font color="#2563eb"><a href="url">[1]</a></font>
        # We use [[\1]] style to ensure brackets are shown if using markdown, but in HTML we just write them.
        text = re.sub(r'\[+([^\[\]]+)\]+\(([^\s\)]+)\)', 
                      r'<font color="#2563eb"><a href="\2">[\1]</a></font>', text)
        # Basic paragraph break
        text = text.replace("\n", "<br>")
    elif mode == "markdown":
        # Ensure brackets are preserved for markdown rendered output [[1]]
        text = re.sub(r'\[+([^\[\]]+)\]+\(([^\s\)]+)\)', r'[[\1]](\2)', text)
    else:
        # Plain text
        text = re.sub(r'\[+([^\[\]]+)\]+\([^\)]+\)', r'\1', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Strip Headers
    text = re.sub(r'#{1,6}\s', '', text)
    
    return text.strip()

def generate_pdf_from_data(query, answer, snippets, videos):
    pdf = ReportPDF()
    self_epw = pdf.epw
    
    # --- HERO SECTION ---
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(*pdf.PRIMARY_COLOR)
    pdf.cell(self_epw, 12, "Intelligence Report", ln=1)
    
    pdf.set_font("helvetica", "B", 8)
    pdf.set_text_color(*pdf.MUTED_COLOR)
    timestamp = datetime.now().strftime('%B %d, %Y | %H:%M')
    ref_id = hex(int(datetime.now().timestamp()))[2:].upper()
    pdf.cell(self_epw, 5, f"REFERENCE: {ref_id}  |  ISSUED: {timestamp}", ln=1)
    
    pdf.ln(1)
    pdf.set_draw_color(*pdf.PRIMARY_COLOR)
    pdf.set_line_width(1.0)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    # --- QUERY ---
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(*pdf.MUTED_COLOR)
    pdf.cell(self_epw, 5, "PRIMARY RESEARCH OBJECTIVE", ln=1)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*pdf.TEXT_COLOR)
    pdf.multi_cell(self_epw, 6, clean_text(query, mode="markdown"), markdown=True)
    pdf.ln(6)

    # --- ANSWER / SUMMARY ---
    if answer:
        pdf.section_header("Executive Summary")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(*pdf.SECONDARY_COLOR)
        # Use HTML path for blue bracketed links
        pdf.write_styled_content(answer)
        pdf.ln(4)

    # --- SOURCES ---
    if snippets:
        pdf.section_header("Supporting Evidence & Sources")
        for i, s in enumerate(snippets):
            pdf.item_title(i + 1, s.get("title", "Untitled Source"))
            
            if s.get("snippet"):
                pdf.set_font("helvetica", "", 9)
                pdf.set_text_color(*pdf.SECONDARY_COLOR)
                pdf.write_styled_content(s["snippet"])
            
            if s.get("url"):
                pdf.ln(0.5)
                pdf.set_font("helvetica", "I", 8)
                pdf.set_text_color(*pdf.LINK_COLOR)
                pdf.multi_cell(self_epw, 4.5, clean_text(s["url"], mode="plain"))
            
            if i < len(snippets) - 1:
                pdf.separator()

    # --- VIDEOS ---
    if videos:
        if pdf.get_y() > 220:
            pdf.add_page()
            
        pdf.section_header("Visual Insights & Media")
        for i, v in enumerate(videos):
            pdf.item_title(i + 1, v.get("title", "Reference Video"))
            
            if v.get("summary"):
                pdf.set_font("helvetica", "I", 9)
                pdf.set_text_color(*pdf.SECONDARY_COLOR)
                pdf.write_styled_content(v["summary"])
            
            video_url = f"https://www.youtube.com/watch?v={v['video_id']}" if v.get("video_id") else None
            if video_url:
                pdf.ln(0.5)
                pdf.set_font("helvetica", "", 8)
                pdf.set_text_color(*pdf.LINK_COLOR)
                pdf.multi_cell(self_epw, 4.5, video_url)
            
            if v.get("moments"):
                pdf.ln(1)
                pdf.set_font("helvetica", "B", 8)
                pdf.set_text_color(*pdf.MUTED_COLOR)
                pdf.cell(self_epw, 4, "KEY SEGMENTS:", ln=1)
                
                pdf.set_font("helvetica", "", 8.5)
                pdf.set_text_color(*pdf.SECONDARY_COLOR)
                for m in v["moments"]:
                    time_m, time_s = divmod(int(m.get("start", 0)), 60)
                    ts = f"{time_m}:{time_s:02d}"
                    # Use plain for segments to avoid complex nesting in lists
                    pdf.multi_cell(self_epw, 4.5, f"  [{ts}] {clean_text(m.get('summary', ''), mode='plain')}")
            
            if i < len(videos) - 1:
                pdf.separator()

    return pdf.output()
