from flask import send_file

from charts import get_chart_paths

def export_chart_image():
    import glob, os
    chart_files = sorted(glob.glob('static/chart_*.png'), key=os.path.getmtime, reverse=True)
    if chart_files:
        return send_file(chart_files[0], as_attachment=True)
    return "No chart found", 404

def export_data_csv(df):
    if df is not None:
        out_path = "static/data_export.csv"
        df.to_csv(out_path, index=False)
        return send_file(out_path, as_attachment=True)
    return "No data loaded", 404

def export_report_pdf(df):
    # Use fpdf to generate a PDF with charts and analysis summary
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'AI Analytics Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Rows: {len(df)}", 0, 1)
    chart_files = sorted(get_chart_paths(), reverse=True)
    for chart in chart_files[:2]:  # Include first two charts
        pdf.image(chart, w=150)
        pdf.ln(10)
    out_pdf = "static/report.pdf"
    pdf.output(out_pdf)
    return send_file(out_pdf, as_attachment=True)
