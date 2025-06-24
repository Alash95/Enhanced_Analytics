import os
import re
import uuid
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import openai
import markdown2
from charts import generate_chart_from_ai

app = Flask(__name__)
app.secret_key = "supersecret"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

DATASTORE = {}

##############################################
# Data Cleaning Functions                   #
##############################################

def analyze_data_quality(df):
    """Analyze data quality and return cleaning suggestions"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicates': df.duplicated().sum(),
        'data_types': {},
        'potential_issues': [],
        'cleaning_suggestions': []
    }
    
    # Check missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        quality_report['missing_values'][col] = {
            'count': missing_count,
            'percentage': round(missing_pct, 2)
        }
        
        if missing_pct > 50:
            quality_report['potential_issues'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
            quality_report['cleaning_suggestions'].append(f"Consider dropping column '{col}' due to high missing values")
        elif missing_pct > 10:
            quality_report['cleaning_suggestions'].append(f"Handle missing values in column '{col}' ({missing_pct:.1f}% missing)")
    
    # Check data types
    for col in df.columns:
        dtype = str(df[col].dtype)
        quality_report['data_types'][col] = dtype
        
        # Check for potential date columns
        if dtype == 'object':
            sample_values = df[col].dropna().head(10).astype(str)
            if any(re.match(r'\d{4}-\d{2}-\d{2}', str(val)) for val in sample_values):
                quality_report['cleaning_suggestions'].append(f"Column '{col}' might be a date - consider converting to datetime")
    
    # Check for duplicates
    if quality_report['duplicates'] > 0:
        quality_report['cleaning_suggestions'].append(f"Remove {quality_report['duplicates']} duplicate rows")
    
    return quality_report

def clean_dataset(df, cleaning_options):
    """Apply cleaning operations based on user selections"""
    cleaned_df = df.copy()
    cleaning_log = []
    
    try:
        # Remove duplicates
        if cleaning_options.get('remove_duplicates', False):
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                cleaning_log.append(f"Removed {removed_rows} duplicate rows")
        
        # Handle missing values
        missing_strategy = cleaning_options.get('missing_strategy', 'keep')
        if missing_strategy == 'drop_rows':
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna()
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                cleaning_log.append(f"Dropped {removed_rows} rows with missing values")
        
        elif missing_strategy == 'fill_numeric':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    fill_value = cleaned_df[col].median()
                    cleaned_df[col].fillna(fill_value, inplace=True)
                    cleaning_log.append(f"Filled missing values in '{col}' with median ({fill_value})")
        
        # Remove columns with high missing values
        high_missing_threshold = cleaning_options.get('high_missing_threshold', 80)
        cols_to_drop = []
        for col in cleaned_df.columns:
            missing_pct = (cleaned_df[col].isnull().sum() / len(cleaned_df)) * 100
            if missing_pct > high_missing_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop and cleaning_options.get('drop_high_missing', False):
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            cleaning_log.append(f"Dropped columns with >{high_missing_threshold}% missing values: {', '.join(cols_to_drop)}")
        
        # Convert date columns
        date_columns = cleaning_options.get('date_columns', [])
        for col in date_columns:
            if col in cleaned_df.columns:
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    cleaning_log.append(f"Converted '{col}' to datetime")
                except:
                    cleaning_log.append(f"Failed to convert '{col}' to datetime")
        
        # Standardize column names
        if cleaning_options.get('standardize_columns', False):
            old_columns = cleaned_df.columns.tolist()
            cleaned_df.columns = [col.lower().strip().replace(' ', '_') for col in cleaned_df.columns]
            cleaning_log.append("Standardized column names (lowercase, underscores)")
        
        return cleaned_df, cleaning_log
    
    except Exception as e:
        return df, [f"Error during cleaning: {str(e)}"]

##############################################
# Q&A Functions                             #
##############################################

def generate_data_context(df):
    """Generate context about the dataset for the AI"""
    context = f"""
Dataset Overview:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Columns: {', '.join(df.columns)}

Column Details:
"""
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        context += f"- {col}: {dtype}, {null_count} nulls, {unique_count} unique values"
        
        if dtype in ['int64', 'float64']:
            context += f", range: {df[col].min():.2f} to {df[col].max():.2f}"
        elif dtype == 'object' and unique_count < 20:
            sample_values = df[col].value_counts().head(5).index.tolist()
            context += f", top values: {', '.join(map(str, sample_values))}"
        
        context += "\n"
    
    # Add sample data
    context += f"\nSample data (first 5 rows):\n{df.head().to_string()}"
    
    return context

def answer_data_question(question, df):
    """Use AI to answer questions about the data"""
    data_context = generate_data_context(df)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst assistant. Answer questions about the provided dataset. "
                "Be specific and provide actionable insights. When relevant, suggest specific analyses, "
                "visualizations, or KPIs. Use exact column names from the dataset. "
                "If calculations are needed, provide the specific aggregation functions like sum(), avg(), count(), etc."
            )
        },
        {
            "role": "user",
            "content": f"Dataset Context:\n{data_context}\n\nQuestion: {question}"
        }
    ]
    
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

##############################################
# Existing Functions (Updated)             #
##############################################

def extract_kpis_from_answer(answer, df=None):
    """Extract KPIs using multiple patterns for any LLM output style."""
    kpis = []
    seen = set()
    patterns = [
        r"\*\*([^\*]+)\*\*:? [`']?(sum|avg|count|min|max)\(([a-zA-Z0-9_\/]+)\)[`']?",
        r"([A-Za-z ]+)\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*aggregation:\s*([a-z]+)\s*\)",
        r"([A-Za-z ]+): (sum|avg|count|min|max)\(([a-zA-Z0-9_\/]+)\)",
        r"(sum|avg|count|min|max)\(([a-zA-Z0-9_\/]+)\)"
    ]
    
    if df is not None:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
    
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        for m in matches:
            if len(m) == 3:
                label, agg, col = m
            elif len(m) == 2:
                agg, col = m
                label = f"{agg.capitalize()} {col.capitalize()}"
            else:
                continue
            
            key = (label.strip().lower(), agg.strip().lower(), col.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            
            col = col.lower().strip()
            agg = agg.lower().strip()
            label = label.strip()
            value, isnum = None, False
            
            if df is not None and col in df.columns:
                try:
                    if agg == 'sum':
                        value = round(df[col].sum(), 2)
                        isnum = True
                    elif agg == 'avg':
                        value = round(df[col].mean(), 2)
                        isnum = True
                    elif agg == 'count':
                        value = int(df[col].nunique())
                        isnum = True
                    elif agg == 'min':
                        value = round(df[col].min(), 2)
                        isnum = True
                    elif agg == 'max':
                        value = round(df[col].max(), 2)
                        isnum = True
                except Exception:
                    value = f"{agg}({col})"
            else:
                value = f"{agg}({col})"
            
            if label.lower() not in ['x-axis', 'y-axis', 'values', 'labels']:
                kpis.append({'label': label, 'value': value, 'isnum': isnum})
    
    return kpis

def parse_ai_answer_for_charts(answer):
    """Parse AI answer for chart specifications"""
    charts = []
    patterns = [
        r"Chart Type\*\*: ([a-zA-Z ]+)[\s\S]*?X[- ]?Axis\*\*: ?[`']?([a-zA-Z0-9_]+)[`']?[\s\S]*?Y[- ]?Axis\*\*: ?[`']?(sum|avg|count|min|max)\(([a-zA-Z0-9_]+)\)[`']?",
        r"(Bar Chart|Line Chart|Pie Chart|Scatter Plot|Histogram)[\s\S]*?X[- ]?Axis: ?[`']?([a-zA-Z0-9_]+)[`']?[\s\S]*?Y[- ]?Axis: ?[`']?(sum|avg|count|min|max)\(([a-zA-Z0-9_]+)\)[`']?",
        r"X[- ]?Axis: ?[`']?([a-zA-Z0-9_]+)[`']?[\s\S]*?Y[- ]?Axis: ?[`']?(sum|avg|count|min|max)\(([a-zA-Z0-9_]+)\)[`']?"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        for m in matches:
            if len(m) == 4:
                chart_type, x, agg, y = m
            elif len(m) == 3:
                x, agg, y = m
                chart_type = "bar chart"
            else:
                continue
            
            charts.append({
                'chart_type': chart_type.strip().lower(),
                'x': x.strip().lower(),
                'agg': agg.strip().lower(),
                'y': y.strip().lower(),
            })
    
    return charts

def parse_markdown_tables(answer):
    """Parse markdown tables from AI answer"""
    tables, table_lines, in_table = [], [], False
    for line in answer.splitlines():
        if "|" in line:
            table_lines.append(line)
            in_table = True
        elif in_table:
            tables.append("\n".join(table_lines))
            table_lines, in_table = [], False
    
    if table_lines:
        tables.append("\n".join(table_lines))
    
    html_tables = []
    for t in tables:
        rows = [r.strip() for r in t.splitlines() if "|" in r]
        if not rows:
            continue
        
        html = "<table class='table table-striped table-bordered'>"
        for i, row in enumerate(rows):
            cols = [c.strip() for c in row.split("|") if c.strip()]
            if not cols:
                continue
            
            if i == 0:
                html += "<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>"
            else:
                html += "<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>"
        
        html += "</tbody></table>"
        html_tables.append(html)
    
    return html_tables

#############################
#         FLASK ROUTES      #
#############################

@app.route('/', methods=['GET', 'POST'])
def index():
    table, error, filename = None, None, None
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith(('.csv', '.xlsx')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
                df.columns = [c.lower() for c in df.columns]
                DATASTORE['df'] = df
                DATASTORE['original_df'] = df.copy()  # Keep original for reference
                filename = file.filename
                table = df.head(20).to_html(classes='table table-striped', index=False, border=0)
            except Exception as e:
                error = f"Failed to read file: {str(e)}"
        else:
            error = "Only CSV or Excel files allowed."
    elif 'df' in DATASTORE:
        df = DATASTORE['df']
        table = df.head(20).to_html(classes='table table-striped', index=False, border=0)
    
    return render_template('index.html', table=table, error=error, filename=filename)

@app.route('/data-quality')
def data_quality():
    """Show data quality analysis"""
    df = DATASTORE.get('df')
    if df is None:
        return render_template('data_quality.html', error="No data loaded. Please upload a file first.")
    
    quality_report = analyze_data_quality(df)
    return render_template('data_quality.html', quality_report=quality_report)

@app.route('/clean-data', methods=['POST'])
def clean_data():
    """Apply data cleaning operations"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'})
    
    cleaning_options = request.json
    cleaned_df, cleaning_log = clean_dataset(df, cleaning_options)
    
    # Update datastore with cleaned data
    DATASTORE['df'] = cleaned_df
    DATASTORE['cleaning_log'] = cleaning_log
    
    return jsonify({
        'success': True,
        'cleaning_log': cleaning_log,
        'new_shape': cleaned_df.shape
    })

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """Answer questions about the data using AI"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'})
    
    question = request.json.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'})
    
    answer = answer_data_question(question, df)
    
    # Extract any KPIs and charts from the answer
    kpis = extract_kpis_from_answer(answer, df)
    chart_specs = parse_ai_answer_for_charts(answer)
    
    chart_urls = []
    for spec in chart_specs:
        url = generate_chart_from_ai(spec, df)
        if url:
            chart_urls.append(url)
    
    return jsonify({
        'answer': answer,
        'answer_html': markdown2.markdown(answer),
        'kpis': kpis,
        'chart_urls': chart_urls
    })

@app.route('/dashboard')
def dashboard():
    """Generate and display dashboard"""
    df = DATASTORE.get('df')
    if df is None:
        return render_template('dashboard.html', 
                             error="No data loaded. Please upload a file first.",
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])
    
    sample = df.head(20).to_csv(index=False)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a dashboard designer. Analyze the user's data sample and generate a dashboard plan:\n"
                "- List 3 to 5 KPIs (with exact column names and aggregation: sum, avg, count, etc)\n"
                "- Suggest 2 to 3 most important charts (type, x, y columns)\n"
                "- If a table is useful, describe it\n"
                "Reply using Markdown with clear sections: KPIs, Charts, Tables, and a summary."
            )
        },
        {
            "role": "user",
            "content": f"CSV data sample:\n{sample}\n\nGenerate an executive dashboard."
        }
    ]
    
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.2,
            max_tokens=1200
        )
        
        dashboard_plan = response.choices[0].message.content.strip()
    except Exception as e:
        return render_template('dashboard.html', 
                             error=f"Error contacting Azure OpenAI: {str(e)}",
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])
    
    # Extract KPIs
    kpis = extract_kpis_from_answer(dashboard_plan, df)
    
    # Extract and generate charts
    chart_specs = parse_ai_answer_for_charts(dashboard_plan)
    chart_urls = []
    for spec in chart_specs:
        url = generate_chart_from_ai(spec, df)
        if url:
            chart_urls.append(url)
    
    # Extract tables
    html_tables = parse_markdown_tables(dashboard_plan)
    
    # Render summary as markdown
    dashboard_plan_html = markdown2.markdown(dashboard_plan)
    
    return render_template('dashboard.html',
                         dashboard_plan_html=dashboard_plan_html,
                         kpis=kpis,
                         chart_urls=chart_urls,
                         tables=html_tables,
                         error=None)

@app.route('/reset-data')
def reset_data():
    """Reset to original data"""
    if 'original_df' in DATASTORE:
        DATASTORE['df'] = DATASTORE['original_df'].copy()
        return jsonify({'success': True, 'message': 'Data reset to original state'})
    return jsonify({'error': 'No original data found'})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(host="0.0.0.0", port=port)
