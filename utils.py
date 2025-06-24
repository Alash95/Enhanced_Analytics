import re

import pandas as pd

def clean_md(text):
    return re.sub(r"^[\*\s:\-]+|[\*\s:\-]+$", "", text).strip()

def get_relevant_columns(df):
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric, non_numeric

def parse_ai_answer_for_charts(answer):
    # Returns list of chart specs dicts: {'x':..., 'y':..., 'type':..., 'series':...}
    specs = []
    # Find all "Recommended Chart" blocks and parse x/y/type/series
    blocks = re.findall(
        r'(?i)Recommended Chart[^\n\r]*[\n\r\s\-:]*'
        r'(?:\*\*)?(?P<chart_type>.+?)(?:\*\*)?(?:\:|[\n\r])'
        r'[\n\r\s\-]*- (?:\*\*)?X-axis(?:\*\*)?:\s*(?P<x_axis>.+)'
        r'[\n\r\s\-]*- (?:\*\*)?Y-axis(?:\*\*)?:\s*(?P<y_axis>.+)'
        r'(?:[\n\r\s\-]*- (?:\*\*)?Series(?:\*\*)?:\s*(?P<series>.+))?'
        r'(?:[\n\r\s\-]*- (?:\*\*)?Chart Type(?:\*\*)?:\s*(?P<plot_type>.+))?',
        answer
    )
    for block in blocks:
        chart_type_phrase, x_axis, y_axis, series, plot_type = block
        type_detected = 'bar'
        ctp = (chart_type_phrase or '').lower()
        pt = (plot_type or '').lower()
        if 'horizontal' in pt or 'horizontal' in ctp:
            type_detected = 'horizontal_bar'
        elif 'grouped' in pt or 'grouped' in ctp:
            type_detected = 'grouped_bar'
        elif 'line' in pt or 'line' in ctp:
            type_detected = 'line'
        elif 'pie' in pt or 'pie' in ctp:
            type_detected = 'pie'
        elif 'donut' in pt or 'donut' in ctp:
            type_detected = 'donut'
        # Multi-line: look for multi-line/multi-series language
        elif series:
            if 'line' in pt or 'line' in ctp:
                type_detected = 'multi_line'
            else:
                type_detected = 'grouped_bar'
        specs.append({
            'x': clean_md(x_axis),
            'y': clean_md(y_axis),
            'type': type_detected,
            'series': clean_md(series) if series else None,
        })
    return specs

def parse_markdown_tables(answer):
    # Simple extraction of markdown tables and render as HTML
    table_blocks = re.findall(r'((?:\|[^\n]+\|\n)+)', answer)
    html_tables = []
    for tb in table_blocks:
        # Convert markdown table to DataFrame
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(tb.replace('|', ',')), sep=',')
            html_tables.append(df.to_html(classes='table table-bordered', index=False, border=0))
        except Exception:
            continue
    return html_tables
