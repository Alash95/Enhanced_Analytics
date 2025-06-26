import os
import re
import uuid
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import openai
import markdown2

app = Flask(__name__)
app.secret_key = "supersecret"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

DATASTORE = {}

##############################################
# Chart Generation Import                    #
##############################################

try:
    from charts import generate_chart_from_ai, create_default_charts, debug_column_matching
except ImportError:
    print("Charts module not found. Chart generation will be disabled.")
    def generate_chart_from_ai(spec, df):
        return None
    def create_default_charts(df):
        return []
    def debug_column_matching(df):
        print("Debug function not available")

##############################################
# Enhanced Column Utilities                 #
##############################################

def normalize_column_name(col_name):
    """Normalize column names for consistent matching"""
    if not col_name:
        return ""
    normalized = str(col_name).strip().lower()
    normalized = re.sub(r'[\s\-/]+', '_', normalized)
    normalized = re.sub(r'[^\w_]', '', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    normalized = normalized.strip('_')
    return normalized

def clean_column_names_enhanced(df, store_mapping=True):
    """Enhanced column name cleaning with better mapping storage"""
    original_columns = df.columns.tolist()
    
    # Clean column names - keep spaces for readability but ensure consistency
    cleaned_columns = []
    for col in original_columns:
        cleaned = str(col).strip().lower()
        cleaned_columns.append(cleaned)
    
    # Apply cleaned names
    df.columns = cleaned_columns
    
    # Store comprehensive mapping in DATASTORE
    if store_mapping:
        DATASTORE['column_mapping'] = dict(zip(original_columns, cleaned_columns))
        DATASTORE['reverse_column_mapping'] = dict(zip(cleaned_columns, original_columns))
        DATASTORE['normalized_mapping'] = {
            normalize_column_name(orig): cleaned 
            for orig, cleaned in zip(original_columns, cleaned_columns)
        }
    
    return df, dict(zip(original_columns, cleaned_columns))

def validate_chart_spec_in_app(spec, df):
    """Validate chart specification against dataframe columns"""
    if not spec or not isinstance(spec, dict):
        return None
    
    # Import the enhanced column matching from charts module
    try:
        from charts import find_matching_column
        
        x_col = find_matching_column(spec.get('x', ''), df.columns, debug=True)
        y_col = find_matching_column(spec.get('y', ''), df.columns, debug=True)
        
        if not x_col or not y_col:
            print(f"Chart validation failed: x='{spec.get('x')}' -> {x_col}, y='{spec.get('y')}' -> {y_col}")
            return None
        
        # Ensure Y column is numeric
        if y_col not in df.select_dtypes(include=[np.number]).columns:
            print(f"Y column '{y_col}' is not numeric")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                y_col = numeric_cols[0]
                print(f"Using numeric column '{y_col}' instead")
            else:
                return None
        
        return {
            'chart_type': spec.get('chart_type', 'bar chart'),
            'x': x_col,
            'y': y_col,
            'agg': spec.get('agg', 'sum')
        }
    
    except ImportError:
        print("Enhanced column matching not available")
        return spec

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
    """Generate enhanced context about the dataset for the AI"""
    context = f"""
Dataset Overview:
- Total rows: {len(df):,}
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
            try:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                total_val = df[col].sum()
                context += f", range: {min_val:.2f} to {max_val:.2f}, mean: {mean_val:.2f}, total: {total_val:,.2f}"
            except:
                context += f", numeric column"
        elif dtype == 'object' and unique_count < 50:
            value_counts = df[col].value_counts().head(10)
            context += f", top values: {dict(value_counts)}"
        
        context += "\n"
    
    # Add sample calculations for common aggregations
    context += f"\nSample Calculations (first 10 rows):\n"
    
    # Show actual sample data with calculations
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        sample_groupby = df.groupby(categorical_cols[0])[numeric_cols[0]].agg(['sum', 'mean', 'count']).head()
        context += f"Sample aggregation by {categorical_cols[0]}:\n{sample_groupby.to_string()}\n"
    
    # Add first 5 rows of actual data
    context += f"\nActual data sample (first 5 rows):\n{df.head().to_string()}\n"
    
    return context

def perform_actual_calculation(question, df):
    """Perform actual calculations based on the question and provide real results"""
    question_lower = question.lower()
    results = {}
    
    try:
        # Detect what kind of calculation is needed
        if 'total' in question_lower and 'segment' in question_lower:
            if 'segment' in df.columns and 'sales' in df.columns:
                segment_totals = df.groupby('segment')['sales'].sum().sort_values(ascending=False)
                results['segment_sales'] = segment_totals.to_dict()
        
        elif 'total' in question_lower and 'region' in question_lower:
            if 'region' in df.columns and 'sales' in df.columns:
                region_totals = df.groupby('region')['sales'].sum().sort_values(ascending=False)
                results['region_sales'] = region_totals.to_dict()
        
        elif 'average' in question_lower or 'avg' in question_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col.lower() in question_lower:
                    results[f'avg_{col}'] = df[col].mean()
        
        elif 'top' in question_lower:
            # Find top performers
            if 'sales' in df.columns:
                categorical_cols = df.select_dtypes(include=['object']).columns
                for cat_col in categorical_cols:
                    if cat_col.lower() in question_lower:
                        top_values = df.groupby(cat_col)['sales'].sum().sort_values(ascending=False).head(10)
                        results[f'top_{cat_col}'] = top_values.to_dict()
        
        return results
    except Exception as e:
        print(f"Calculation error: {e}")
        return {}

def answer_data_question(question, df):
    """Use AI to answer questions about the data with actual calculations"""
    data_context = generate_data_context(df)
    
    # Perform actual calculations
    calculations = perform_actual_calculation(question, df)
    calculation_text = ""
    if calculations:
        calculation_text = f"\nPre-calculated results for your question:\n{calculations}\n"
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst assistant. Answer questions about the provided dataset by providing actual numerical results and insights. "
                "ALWAYS give specific numbers and calculated values, not instructions on how to calculate. "
                "When you see pre-calculated results, use those exact numbers in your response. "
                "Provide clear, actionable insights with real data from the dataset. "
                "Format your response professionally with clear sections and bullet points. "
                "Focus on answering the question directly with specific numerical results."
            )
        },
        {
            "role": "user",
            "content": f"Dataset Context:\n{data_context}\n{calculation_text}\nQuestion: {question}\n\nProvide the actual calculated answer with specific numbers, insights, and recommendations."
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
            temperature=0.1,  # Lower temperature for more factual responses
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

##############################################
# KPI and Chart Parsing Functions          #
##############################################

def extract_kpis_from_answer(answer, df=None):
    """Extract KPIs using multiple patterns for any LLM output style."""
    if not answer:
        return []
        
    kpis = []
    seen = set()
    patterns = [
        r"\*\*([^\*]+)\*\*:? [`']?(sum|avg|count|min|max)\(([a-zA-Z0-9_\s\/]+)\)[`']?",
        r"([A-Za-z ]+)\s*\(\s*([a-zA-Z0-9_\s]+)\s*,\s*aggregation:\s*([a-z]+)\s*\)",
        r"([A-Za-z ]+): (sum|avg|count|min|max)\(([a-zA-Z0-9_\s\/]+)\)",
        r"(sum|avg|count|min|max)\(([a-zA-Z0-9_\s\/]+)\)"
    ]
    
    if df is not None:
        df_work = df.copy()
        # Create a mapping for column lookup
        col_lookup = {col.lower().strip(): col for col in df_work.columns}
    
    try:
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for m in matches:
                try:
                    if len(m) == 3:
                        label, agg, col = m
                    elif len(m) == 2:
                        agg, col = m
                        label = f"{agg.capitalize()} {col.capitalize()}"
                    else:
                        continue
                    
                    # Create unique key using original label to avoid duplicates
                    key = (label.strip().lower(), agg.strip().lower(), col.strip().lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    col = col.lower().strip()
                    agg = agg.lower().strip()
                    label = label.strip()
                    value, isnum = None, False
                    
                    if df is not None:
                        # Find matching column
                        actual_col = None
                        if col in col_lookup:
                            actual_col = col_lookup[col]
                        else:
                            # Try fuzzy matching
                            for df_col in df_work.columns:
                                if col in df_col.lower() or df_col.lower() in col:
                                    actual_col = df_col
                                    break
                        
                        if actual_col and actual_col in df_work.columns:
                            try:
                                if agg == 'sum':
                                    value = round(df_work[actual_col].sum(), 2)
                                    isnum = True
                                elif agg == 'avg':
                                    value = round(df_work[actual_col].mean(), 2)
                                    isnum = True
                                elif agg == 'count':
                                    value = int(df_work[actual_col].nunique()) if df_work[actual_col].dtype == 'object' else len(df_work)
                                    isnum = True
                                elif agg == 'min':
                                    value = round(df_work[actual_col].min(), 2)
                                    isnum = True
                                elif agg == 'max':
                                    value = round(df_work[actual_col].max(), 2)
                                    isnum = True
                            except Exception:
                                value = f"{agg}({col})"
                        else:
                            value = f"{agg}({col})"
                    else:
                        value = f"{agg}({col})"
                    
                    # Skip axis-related labels and ensure unique KPIs
                    if label.lower() not in ['x-axis', 'y-axis', 'values', 'labels']:
                        # Additional check to avoid duplicate KPIs with same calculation
                        is_duplicate = False
                        for existing_kpi in kpis:
                            if (existing_kpi['label'].lower() == label.lower() or 
                                (existing_kpi['value'] == value and existing_kpi['isnum'] == isnum)):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            kpis.append({'label': label, 'value': value, 'isnum': isnum})
                
                except Exception as e:
                    print(f"Error processing KPI match {m}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error extracting KPIs: {e}")
    
    return kpis

def parse_ai_answer_for_charts(answer):
    """Parse AI answer for chart specifications with improved pattern matching"""
    if not answer:
        return []
    
    charts = []
    patterns = [
        # Pattern 1: **Chart Type**: ... **X-Axis**: ... **Y-Axis**: ...
        r'\*\*Chart Type\*\*:\s*([^*\n]+).*?\*\*X[- ]?Axis\*\*:\s*([a-zA-Z0-9_\s]+).*?\*\*Y[- ]?Axis\*\*:\s*(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)',
        # Pattern 2: Chart Type: ... X-Axis: ... Y-Axis: ...
        r'(Bar Chart|Line Chart|Pie Chart|Horizontal Bar Chart|Scatter Plot|Histogram).*?X[- ]?Axis:\s*([a-zA-Z0-9_\s]+).*?Y[- ]?Axis:\s*(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)',
        # Pattern 3: X-Axis: ... Y-Axis: ... (default to bar chart)
        r'X[- ]?Axis:\s*([a-zA-Z0-9_\s]+).*?Y[- ]?Axis:\s*(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)',
        # Pattern 4: Simple aggregation pattern
        r'(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)\s+by\s+([a-zA-Z0-9_\s]+)'
    ]
    
    try:
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, answer, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                try:
                    if i == 0 and len(match) == 4:  # Pattern 1
                        chart_type, x, agg, y = match
                    elif i == 1 and len(match) == 4:  # Pattern 2
                        chart_type, x, agg, y = match
                    elif i == 2 and len(match) == 3:  # Pattern 3
                        x, agg, y = match
                        chart_type = "bar chart"
                    elif i == 3 and len(match) == 3:  # Pattern 4
                        agg, y, x = match
                        chart_type = "bar chart"
                    else:
                        continue
                    
                    # Clean and validate the extracted values
                    chart_type = chart_type.strip().lower()
                    x = x.strip().lower()
                    y = y.strip().lower()
                    agg = agg.strip().lower()
                    
                    # Skip invalid combinations
                    if x == 'avg' or y == 'avg' or x == agg or y == agg:
                        continue
                    
                    # Skip if column names are too generic or invalid
                    if x in ['column', 'field', 'data'] or y in ['column', 'field', 'data']:
                        continue
                    
                    charts.append({
                        'chart_type': chart_type,
                        'x': x,
                        'agg': agg,
                        'y': y,
                    })
                    
                except Exception as e:
                    print(f"Error processing match {match}: {e}")
                    continue
        
        # Remove duplicates while preserving order
        unique_charts = []
        seen = set()
        for chart in charts:
            chart_key = (chart['x'], chart['y'], chart['agg'])
            if chart_key not in seen:
                seen.add(chart_key)
                unique_charts.append(chart)
        
        return unique_charts
    
    except Exception as e:
        print(f"Error parsing chart specifications: {e}")
        return []

def parse_markdown_tables(answer):
    """Parse markdown tables from AI answer"""
    if not answer:
        return []
        
    tables = []
    try:
        table_lines = []
        in_table = False
        
        for line in answer.splitlines():
            if "|" in line:
                table_lines.append(line)
                in_table = True
            elif in_table:
                tables.append("\n".join(table_lines))
                table_lines = []
                in_table = False
        
        if table_lines:
            tables.append("\n".join(table_lines))
        
        html_tables = []
        for t in tables:
            try:
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
            except Exception as e:
                print(f"Error processing table: {e}")
                continue
        
        return html_tables
    
    except Exception as e:
        print(f"Error parsing markdown tables: {e}")
        return []

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
                
                # Store original data first
                DATASTORE['original_df'] = df.copy()
                DATASTORE['original_columns'] = df.columns.tolist()
                
                # Enhanced column cleaning with better mapping
                df, column_mapping = clean_column_names_enhanced(df)
                
                # Store cleaned data
                DATASTORE['df'] = df
                
                filename = file.filename
                table = df.head(20).to_html(classes='table table-striped', index=False, border=0)
                
                # Log column cleaning with more detail
                print(f"Column names cleaned: {column_mapping}")
                print(f"DataFrame shape: {df.shape}")
                print(f"Column data types: {df.dtypes.to_dict()}")
                
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
    return render_template('data_quality.html', quality_report=quality_report, error=None)

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
    if chart_specs:
        for spec in chart_specs:
            # Validate spec before generating chart
            validated_spec = validate_chart_spec_in_app(spec, df)
            if validated_spec:
                url = generate_chart_from_ai(validated_spec, df)
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
    """Generate and display dashboard with improved chart generation"""
    df = DATASTORE.get('df')
    if df is None:
        return render_template('dashboard.html', 
                             error="No data loaded. Please upload a file first.",
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])
    
    # Enhanced debugging information
    print(f"=== DASHBOARD GENERATION DEBUG ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    # Run column matching debug
    debug_column_matching(df)
    
    # Create sample data for AI analysis (first 20 rows)
    sample = df.head(20).to_csv(index=False)
    
    # Enhanced prompt with explicit column information
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert business intelligence analyst. Create a comprehensive executive dashboard plan.\n\n"
                "IMPORTANT FORMATTING REQUIREMENTS:\n"
                "- For KPIs, use this EXACT format: **KPI Name**: aggregation(column_name)\n"
                "- For charts, use this EXACT format:\n"
                "  **Chart Type**: Bar Chart (or Pie Chart, Line Chart, Horizontal Bar Chart)\n"
                "  **X-Axis**: column_name\n"
                "  **Y-Axis**: aggregation(column_name)\n\n"
                "Example:\n"
                "**Total Revenue**: sum(amount)\n"
                "**Chart Type**: Bar Chart\n"
                "**X-Axis**: branch\n"
                "**Y-Axis**: sum(amount)\n\n"
                f"AVAILABLE COLUMNS: {', '.join(df.columns)}\n"
                f"NUMERIC COLUMNS: {', '.join(df.select_dtypes(include=[np.number]).columns)}\n"
                f"CATEGORICAL COLUMNS: {', '.join(df.select_dtypes(include=['object', 'category']).columns)}\n\n"
                "RULES:\n"
                "- ONLY use exact column names shown above\n"
                "- Y-Axis must use numeric columns (quantity, time, amount)\n"
                "- Use aggregations: sum, avg, count, min, max\n"
                "- Create 2-3 charts maximum\n"
                "- Focus on business insights\n"
            )
        },
        {
            "role": "user",
            "content": f"Dataset sample:\n{sample}\n\nCreate a dashboard plan using the EXACT formatting requirements above."
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
        
        dashboard_plan = response.choices[0].message.content.strip()
        print(f"Dashboard plan generated: {dashboard_plan[:200]}...")
        
    except Exception as e:
        error_msg = f"Error contacting Azure OpenAI: {str(e)}"
        print(error_msg)
        return render_template('dashboard.html', 
                             error=error_msg,
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])
    
    # Extract KPIs with improved handling
    kpis = extract_kpis_from_answer(dashboard_plan, df)
    print(f"Extracted KPIs: {len(kpis)} - {[kpi['label'] for kpi in kpis]}")
    
    # Extract and validate chart specifications
    chart_specs = parse_ai_answer_for_charts(dashboard_plan)
    print(f"Raw chart specifications: {chart_specs}")
    
    # Generate charts with enhanced validation
    chart_urls = []
    successful_charts = 0
    
    for i, spec in enumerate(chart_specs):
        try:
            print(f"\n--- Generating Chart {i+1} ---")
            print(f"Original spec: {spec}")
            
            # Validate and fix specification
            validated_spec = validate_chart_spec_in_app(spec, df)
            if validated_spec:
                print(f"Validated spec: {validated_spec}")
                url = generate_chart_from_ai(validated_spec, df)
                if url:
                    chart_urls.append(url)
                    successful_charts += 1
                    print(f"Chart {i+1} generated successfully: {url}")
                else:
                    print(f"Chart {i+1} generation failed")
            else:
                print(f"Chart {i+1} validation failed")
        except Exception as e:
            print(f"Error generating chart {i+1}: {e}")
            continue
    
    print(f"Generated {successful_charts} out of {len(chart_specs)} charts from AI specs")
    
    # If no charts were generated, create default charts
    if not chart_urls:
        try:
            print("No charts generated from AI specs, creating default charts...")
            default_charts = create_default_charts(df)
            if default_charts:
                chart_urls.extend(default_charts)
                print(f"Created {len(default_charts)} default charts")
            else:
                print("Failed to create default charts")
        except Exception as e:
            print(f"Error creating default charts: {e}")
    
    # Extract markdown tables
    html_tables = parse_markdown_tables(dashboard_plan)
    
    # Convert dashboard plan to HTML
    dashboard_plan_html = markdown2.markdown(dashboard_plan)
    
    print(f"Final dashboard: {len(kpis)} KPIs, {len(chart_urls)} charts, {len(html_tables)} tables")
    print("=== END DASHBOARD DEBUG ===")
    
    return render_template('dashboard.html',
                         dashboard_plan_html=dashboard_plan_html,
                         kpis=kpis,
                         chart_urls=chart_urls,
                         tables=html_tables,
                         error=None)

@app.route('/debug-charts')
def debug_charts():
    """Debug route to test chart generation"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'})
    
    # Test chart specs that should work with your pizza data
    test_specs = [
        {
            'chart_type': 'bar chart',
            'x': 'branch',
            'y': 'amount',
            'agg': 'sum'
        },
        {
            'chart_type': 'pie chart',
            'x': 'pizza sold',
            'y': 'quantity',
            'agg': 'sum'
        },
        {
            'chart_type': 'bar chart',
            'x': 'branch',
            'y': 'price',
            'agg': 'avg'
        },
        {
            'chart_type': 'horizontal bar chart',
            'x': 'time range',
            'y': 'amount',
            'agg': 'sum'
        }
    ]
    
    results = []
    for i, spec in enumerate(test_specs):
        try:
            print(f"\n=== Testing Chart Spec {i+1} ===")
            print(f"Original: {spec}")
            
            # Test validation
            validated_spec = validate_chart_spec_in_app(spec, df)
            print(f"Validated: {validated_spec}")
            
            if validated_spec:
                url = generate_chart_from_ai(validated_spec, df)
                results.append({
                    'spec': spec,
                    'validated_spec': validated_spec,
                    'url': url,
                    'success': url is not None
                })
            else:
                results.append({
                    'spec': spec,
                    'validated_spec': None,
                    'url': None,
                    'success': False,
                    'error': 'Validation failed'
                })
        except Exception as e:
            results.append({
                'spec': spec,
                'validated_spec': None,
                'url': None,
                'success': False,
                'error': str(e)
            })
    
    return jsonify({
        'data_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        },
        'column_mappings': DATASTORE.get('column_mapping', {}),
        'test_results': results
    })

@app.route('/convert-data-types', methods=['POST'])
def convert_data_types():
    """Convert data types for specified columns"""
    try:
        df = DATASTORE.get('df')
        if df is None:
            return jsonify({'error': 'No data loaded'})
        
        data_type_changes = request.get_json().get('data_type_changes', {})
        if not data_type_changes:
            return jsonify({'error': 'No data type changes provided'})
        
        conversion_log = []
        converted_df = df.copy()
        
        for column, new_type in data_type_changes.items():
            if column not in converted_df.columns:
                conversion_log.append(f"Column '{column}' not found - skipped")
                continue
            
            try:
                original_type = str(converted_df[column].dtype)
                
                if new_type == 'datetime':
                    converted_df[column] = pd.to_datetime(converted_df[column], errors='coerce')
                    conversion_log.append(f"'{column}': {original_type} → datetime")
                
                elif new_type == 'category':
                    converted_df[column] = converted_df[column].astype('category')
                    conversion_log.append(f"'{column}': {original_type} → category")
                
                elif new_type == 'int':
                    # Handle NaN values before converting to int
                    if converted_df[column].isnull().any():
                        converted_df[column] = converted_df[column].fillna(0)
                    converted_df[column] = pd.to_numeric(converted_df[column], errors='coerce').astype('Int64')
                    conversion_log.append(f"'{column}': {original_type} → integer")
                
                elif new_type == 'float':
                    converted_df[column] = pd.to_numeric(converted_df[column], errors='coerce')
                    conversion_log.append(f"'{column}': {original_type} → float")
                
                elif new_type == 'string':
                    converted_df[column] = converted_df[column].astype('string')
                    conversion_log.append(f"'{column}': {original_type} → string")
                
                elif new_type == 'boolean':
                    # Convert to boolean with common mappings
                    bool_map = {'true': True, 'false': False, '1': True, '0': False, 
                               'yes': True, 'no': False, 'y': True, 'n': False}
                    converted_df[column] = converted_df[column].astype(str).str.lower().map(bool_map)
                    converted_df[column] = converted_df[column].astype('boolean')
                    conversion_log.append(f"'{column}': {original_type} → boolean")
                
            except Exception as e:
                conversion_log.append(f"Failed to convert '{column}' to {new_type}: {str(e)}")
        
        # Update the datastore
        DATASTORE['df'] = converted_df
        
        return jsonify({
            'success': True,
            'conversion_log': conversion_log
        })
    
    except Exception as e:
        print(f"Error in convert_data_types: {e}")
        return jsonify({'error': f'Data type conversion failed: {str(e)}'})

@app.route('/suggest-data-types', methods=['POST'])
def suggest_data_types():
    """AI-powered data type suggestions"""
    try:
        df = DATASTORE.get('df')
        if df is None:
            return jsonify({'error': 'No data loaded'})
        
        suggestions = {}
        
        for column in df.columns:
            current_type = str(df[column].dtype)
            sample_values = df[column].dropna().head(10).astype(str).tolist()
            
            # Auto-detect potential data types
            if current_type == 'object':
                # Check for dates
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                ]
                
                if any(re.match(pattern, str(val)) for pattern in date_patterns for val in sample_values):
                    suggestions[column] = 'datetime'
                
                # Check for booleans
                elif all(str(val).lower() in ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n'] 
                        for val in sample_values if str(val).lower() != 'nan'):
                    suggestions[column] = 'boolean'
                
                # Check for categories (if unique values are less than 50% of total)
                elif df[column].nunique() / len(df) < 0.5 and df[column].nunique() < 50:
                    suggestions[column] = 'category'
                
                # Check for numeric values stored as strings
                elif all(str(val).replace('.', '').replace('-', '').isdigit() 
                        for val in sample_values if str(val).lower() != 'nan'):
                    if any('.' in str(val) for val in sample_values):
                        suggestions[column] = 'float'
                    else:
                        suggestions[column] = 'int'
            
            # Check for potential date columns that are already numeric (timestamps)
            elif current_type in ['int64', 'float64']:
                # Check if values look like timestamps
                if df[column].min() > 1000000000 and df[column].max() < 9999999999:  # Unix timestamp range
                    suggestions[column] = 'datetime'
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    
    except Exception as e:
        print(f"Error in suggest_data_types: {e}")
        return jsonify({'error': f'Data type suggestion failed: {str(e)}'})

@app.route('/reset-data')
def reset_data():
    """Reset to original data"""
    if 'original_df' in DATASTORE:
        # Reset to original and re-clean column names
        original_df = DATASTORE['original_df'].copy()
        df, column_mapping = clean_column_names_enhanced(original_df)
        DATASTORE['df'] = df
        return jsonify({'success': True, 'message': 'Data reset to original state'})
    return jsonify({'error': 'No original data found'})

if __name__ == '__main__':
    app.run(debug=True)
                