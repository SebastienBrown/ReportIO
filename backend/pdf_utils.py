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
        self.set_margins(20, 20, 20) # Tightened margins
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()

    def header(self):
        if self.page_no() == 1:
            return
        # Subsequent pages header: Compact line
        self.set_font("helvetica", "B", 8)
        self.set_text_color(*self.MUTED_COLOR)
        self.cell(self.epw, 6, "Research Report | ReportIO", border="B", align="R")
        self.ln(8) # Reduced from 12

    def footer(self):
        self.set_y(-12) # Closer to page edge
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*self.MUTED_COLOR)
        self.cell(self.epw, 8, f"Page {self.page_no()}", align="C")
        self.set_x(20)
        self.cell(0, 8, "ReportIO Intelligence Service", align="L")

    def draw_section_divider(self):
        self.ln(2) # Reduced
        curr_y = self.get_y()
        self.set_draw_color(*self.PRIMARY_COLOR)
        self.set_line_width(0.4)
        self.line(20, curr_y, 190, curr_y)
        self.ln(4) # Reduced

    def section_header(self, label):
        self.ln(3) # Reduced
        self.set_font("helvetica", "B", 13) # Compact but bold
        self.set_text_color(*self.PRIMARY_COLOR)
        self.cell(self.epw, 8, label.upper(), ln=1)
        self.draw_section_divider()

    def item_title(self, number, text):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*self.TEXT_COLOR)
        title_text = f"{number}. {text}"
        self.multi_cell(self.epw, 6, title_text) # Tighter height
        self.ln(0.5)

    def separator(self):
        self.ln(2) # Reduced
        self.set_draw_color(*self.BORDER_COLOR)
        self.set_line_width(0.1)
        curr_y = self.get_y()
        self.line(30, curr_y, 180, curr_y)
        self.ln(3) # Reduced

def clean_text(text):
    if not text:
        return ""
    
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

    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'#{1,6}\s', '', text)
    return text.strip()

def generate_pdf_from_data(query, answer, snippets, videos):
    pdf = ReportPDF()
    self_epw = pdf.epw
    
    # --- HERO SECTION ---
    pdf.set_font("helvetica", "B", 20) # Slightly more modular
    pdf.set_text_color(*pdf.PRIMARY_COLOR)
    pdf.cell(self_epw, 12, "Intelligence Report", ln=1)
    
    pdf.set_font("helvetica", "B", 8) # BOLD timestamp/ref
    pdf.set_text_color(*pdf.MUTED_COLOR)
    timestamp = datetime.now().strftime('%B %d, %Y | %H:%M')
    ref_id = hex(int(datetime.now().timestamp()))[2:].upper()
    pdf.cell(self_epw, 5, f"REFERENCE: {ref_id}  |  ISSUED: {timestamp}", ln=1)
    
    pdf.ln(1)
    pdf.set_draw_color(*pdf.PRIMARY_COLOR)
    pdf.set_line_width(1.0)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6) # Reduced

    # --- QUERY ---
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(*pdf.MUTED_COLOR)
    pdf.cell(self_epw, 5, "PRIMARY RESEARCH OBJECTIVE", ln=1)
    
    pdf.set_font("helvetica", "B", 12) # Stronger weight for query
    pdf.set_text_color(*pdf.TEXT_COLOR)
    pdf.multi_cell(self_epw, 6, clean_text(query))
    pdf.ln(6) # Reduced

    # --- ANSWER / SUMMARY ---
    if answer:
        pdf.section_header("Executive Summary")
        pdf.set_font("helvetica", "", 10) # Slightly smaller
        pdf.set_text_color(*pdf.SECONDARY_COLOR)
        pdf.multi_cell(self_epw, 5.5, clean_text(answer)) # Tighter rows
        pdf.ln(4)

    # --- SOURCES ---
    if snippets:
        pdf.section_header("Supporting Evidence & Sources")
        for i, s in enumerate(snippets):
            pdf.item_title(i + 1, clean_text(s.get("title", "Untitled Source")))
            
            if s.get("snippet"):
                pdf.set_font("helvetica", "", 9)
                pdf.set_text_color(*pdf.SECONDARY_COLOR)
                pdf.multi_cell(self_epw, 5, clean_text(s["snippet"])) # Tighter
            
            if s.get("url"):
                pdf.ln(0.5)
                pdf.set_font("helvetica", "I", 8)
                pdf.set_text_color(*pdf.LINK_COLOR)
                pdf.multi_cell(self_epw, 4.5, clean_text(s["url"]))
            
            if i < len(snippets) - 1:
                pdf.separator()

    # --- VIDEOS ---
    if videos:
        # Check if we should move to next page
        if pdf.get_y() > 220:
            pdf.add_page()
            
        pdf.section_header("Visual Insights & Media")
        for i, v in enumerate(videos):
            pdf.item_title(i + 1, clean_text(v.get("title", "Reference Video")))
            
            if v.get("summary"):
                pdf.set_font("helvetica", "I", 9)
                pdf.set_text_color(*pdf.SECONDARY_COLOR)
                pdf.multi_cell(self_epw, 5, clean_text(v["summary"]))
            
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
                    pdf.multi_cell(self_epw, 4.5, f"  [{ts}] {clean_text(m.get('summary', ''))}")
            
            if i < len(videos) - 1:
                pdf.separator()

    return pdf.output()
