import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from uuid import uuid4
import warnings
import re
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def normalize_column_name(col_name):
    """Normalize column names for consistent matching"""
    if not col_name:
        return ""
    # Convert to string, strip, lowercase, replace spaces with underscores
    normalized = str(col_name).strip().lower()
    # Replace various separators with underscores
    normalized = re.sub(r'[\s\-/]+', '_', normalized)
    # Remove special characters except underscores
    normalized = re.sub(r'[^\w_]', '', normalized)
    # Remove multiple consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    return normalized

def find_matching_column(target_col, available_cols, debug=True):
    """Enhanced column matching with multiple strategies"""
    if not target_col:
        return None
    
    target_norm = normalize_column_name(target_col)
    
    if debug:
        print(f"Looking for column: '{target_col}' (normalized: '{target_norm}')")
        print(f"Available columns: {list(available_cols)}")
    
    # Strategy 1: Exact match (case-insensitive)
    for col in available_cols:
        if col.lower() == target_col.lower():
            if debug:
                print(f"Exact match found: '{col}'")
            return col
    
    # Strategy 2: Normalized match
    for col in available_cols:
        if normalize_column_name(col) == target_norm:
            if debug:
                print(f"Normalized match found: '{col}' -> '{normalize_column_name(col)}'")
            return col
    
    # Strategy 3: Partial match (contains)
    for col in available_cols:
        col_norm = normalize_column_name(col)
        if target_norm in col_norm or col_norm in target_norm:
            if debug:
                print(f"Partial match found: '{col}'")
            return col
    
    # Strategy 4: Keyword matching for common variations
    keyword_mappings = {
        'pizza_sold': ['pizza', 'sold', 'product'],
        'pizza': ['pizza', 'sold', 'product'],
        'sold': ['pizza', 'sold', 'product'],
        'product': ['pizza', 'sold', 'product'],
        'amount': ['amount', 'total', 'value', 'revenue', 'sales'],
        'sales': ['amount', 'total', 'value', 'revenue', 'sales'],
        'revenue': ['amount', 'total', 'value', 'revenue', 'sales'],
        'total': ['amount', 'total', 'value', 'revenue', 'sales'],
        'value': ['amount', 'total', 'value', 'revenue', 'sales'],
        'price': ['price', 'cost'],
        'cost': ['price', 'cost'],
        'quantity': ['quantity', 'qty', 'count'],
        'qty': ['quantity', 'qty', 'count'],
        'count': ['quantity', 'qty', 'count'],
        'date': ['date', 'time', 'day'],
        'time': ['date', 'time', 'day'],
        'day': ['date', 'time', 'day'],
        'branch': ['branch', 'location', 'store'],
        'location': ['branch', 'location', 'store'],
        'store': ['branch', 'location', 'store'],
    }
    
    if target_norm in keyword_mappings:
        keywords = keyword_mappings[target_norm]
        for col in available_cols:
            col_norm = normalize_column_name(col)
            for keyword in keywords:
                if keyword in col_norm:
                    if debug:
                        print(f"Keyword match found: '{col}' (keyword: '{keyword}')")
                    return col
    
    # Strategy 5: Smart fallbacks for common issues
    smart_fallbacks = {
        'sum': ['amount', 'total', 'value', 'sales', 'revenue'],  # 'sum' is not a column
        'avg': ['price', 'cost', 'amount'],
        'count': ['quantity', 'qty'],
    }
    
    if target_norm in smart_fallbacks:
        fallback_targets = smart_fallbacks[target_norm]
        for fallback in fallback_targets:
            match = find_matching_column(fallback, available_cols, debug=False)
            if match:
                if debug:
                    print(f"Smart fallback found: '{target_col}' -> '{match}'")
                return match
    
    if debug:
        print(f"No match found for '{target_col}'")
    return None

def validate_chart_spec(spec, df):
    """Validate and fix chart specification"""
    if not spec or not isinstance(spec, dict):
        return None
    
    # Find actual column names
    x_col = find_matching_column(spec.get('x', ''), df.columns)
    y_col = find_matching_column(spec.get('y', ''), df.columns)
    
    if not x_col:
        print(f"Cannot find X column for '{spec.get('x', '')}'")
        return None
    
    if not y_col:
        print(f"Cannot find Y column for '{spec.get('y', '')}'")
        return None
    
    # Ensure Y column is numeric for aggregations
    if y_col not in df.select_dtypes(include=[np.number]).columns:
        print(f"Y column '{y_col}' is not numeric (dtype: {df[y_col].dtype})")
        # Try to find a numeric alternative
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            y_col = numeric_cols[0]
            print(f"Using numeric column '{y_col}' instead")
        else:
            print("No numeric columns available")
            return None
    
    return {
        'chart_type': spec.get('chart_type', 'bar chart'),
        'x': x_col,
        'y': y_col,
        'agg': spec.get('agg', 'sum')
    }

def generate_chart_from_ai(spec, df):
    """Generate charts from AI specifications with improved validation"""
    try:
        print(f"Generating chart with spec: {spec}")
        
        # Validate and fix the specification
        validated_spec = validate_chart_spec(spec, df)
        if not validated_spec:
            print(f"Chart generation failed for spec: {spec}")
            return None
        
        chart_type = validated_spec['chart_type'].lower()
        x_col = validated_spec['x']
        y_col = validated_spec['y']
        agg = validated_spec['agg'].lower()
        
        print(f"Using columns: x='{x_col}', y='{y_col}', agg='{agg}'")
        
        # Prepare data based on aggregation
        try:
            df_copy = df.copy()
            
            # Handle different data types for grouping
            if df_copy[x_col].dtype == 'object':
                sample_val = df_copy[x_col].dropna().iloc[0] if not df_copy[x_col].dropna().empty else None
                if hasattr(sample_val, 'strftime'):
                    df_copy[x_col] = df_copy[x_col].astype(str)
            
            # Perform aggregation
            if agg == 'sum':
                data = df_copy.groupby(x_col, observed=True)[y_col].sum().reset_index()
            elif agg in ['avg', 'mean']:
                data = df_copy.groupby(x_col, observed=True)[y_col].mean().reset_index()
            elif agg == 'count':
                data = df_copy.groupby(x_col, observed=True)[y_col].count().reset_index()
            elif agg == 'min':
                data = df_copy.groupby(x_col, observed=True)[y_col].min().reset_index()
            elif agg == 'max':
                data = df_copy.groupby(x_col, observed=True)[y_col].max().reset_index()
            else:
                data = df_copy.groupby(x_col, observed=True)[y_col].sum().reset_index()
            
            # Sort and limit data for better visualization
            data = data.sort_values(y_col, ascending=False).head(15)
            
            if data.empty:
                print("No data after aggregation")
                return None
                
        except Exception as e:
            print(f"Error during data aggregation: {e}")
            return None
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        try:
            if 'bar' in chart_type or 'horizontal' in chart_type:
                if 'horizontal' in chart_type:
                    bars = plt.barh(range(len(data)), data[y_col], 
                                   color='#667eea', alpha=0.8, edgecolor='white', linewidth=0.5)
                    plt.yticks(range(len(data)), data[x_col])
                    plt.xlabel(f'{agg.title()} of {y_col.title()}', fontsize=12, fontweight='bold')
                    plt.ylabel(x_col.title(), fontsize=12, fontweight='bold')
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                               f'{width:,.0f}', ha='left', va='center', fontsize=10)
                else:
                    bars = plt.bar(range(len(data)), data[y_col], 
                                  color='#667eea', alpha=0.8, edgecolor='white', linewidth=0.5)
                    plt.xticks(range(len(data)), data[x_col], rotation=45, ha='right')
                    plt.ylabel(f'{agg.title()} of {y_col.title()}', fontsize=12, fontweight='bold')
                    plt.xlabel(x_col.title(), fontsize=12, fontweight='bold')
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:,.0f}', ha='center', va='bottom', fontsize=10)
            
            elif 'line' in chart_type:
                plt.plot(range(len(data)), data[y_col], marker='o', 
                        linewidth=3, markersize=8, color='#667eea')
                plt.xticks(range(len(data)), data[x_col], rotation=45, ha='right')
                plt.ylabel(f'{agg.title()} of {y_col.title()}', fontsize=12, fontweight='bold')
                plt.xlabel(x_col.title(), fontsize=12, fontweight='bold')
            
            elif 'pie' in chart_type:
                # Use top 8 for pie chart clarity
                plot_data = data.head(8)
                colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
                plt.pie(plot_data[y_col], labels=plot_data[x_col], 
                       autopct='%1.1f%%', startangle=90, colors=colors)
                plt.axis('equal')
            
            else:
                # Default to bar chart
                bars = plt.bar(range(len(data)), data[y_col], 
                              color='#667eea', alpha=0.8)
                plt.xticks(range(len(data)), data[x_col], rotation=45, ha='right')
                plt.ylabel(f'{agg.title()} of {y_col.title()}', fontsize=12, fontweight='bold')
                plt.xlabel(x_col.title(), fontsize=12, fontweight='bold')
            
            # Styling
            if 'pie' not in chart_type:
                plt.grid(True, alpha=0.3, axis='y')
            
            plt.title(f'{agg.title()} of {y_col.title()} by {x_col.title()}', 
                     fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Ensure static directory exists
            os.makedirs('static', exist_ok=True)
            
            # Save chart
            filename = f'chart_{uuid4().hex[:8]}.png'
            filepath = os.path.join('static', filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Chart saved successfully: {filepath}")
            return f'/static/{filename}'
        
        except Exception as e:
            print(f"Error creating plot: {e}")
            plt.close()
            return None
    
    except Exception as e:
        print(f"Chart generation error: {e}")
        try:
            plt.close()
        except:
            pass
        return None

def create_default_charts(df):
    """Create default charts when AI parsing fails"""
    try:
        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Creating default charts. Numeric cols: {numeric_cols}, Categorical cols: {categorical_cols}")
        
        if not numeric_cols:
            print("No numeric columns found for charting")
            return []
        
        if not categorical_cols:
            print("No categorical columns found for charting")
            return []
        
        # Chart 1: First categorical vs first numeric (sum)
        spec1 = {
            'chart_type': 'bar chart',
            'x': categorical_cols[0],
            'agg': 'sum',
            'y': numeric_cols[0]
        }
        url1 = generate_chart_from_ai(spec1, df)
        if url1:
            charts.append(url1)
        
        # Chart 2: Pie chart with same data
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            spec2 = {
                'chart_type': 'pie chart',
                'x': categorical_cols[0],
                'agg': 'sum',
                'y': numeric_cols[0]
            }
            url2 = generate_chart_from_ai(spec2, df)
            if url2:
                charts.append(url2)
        
        # Chart 3: Second numeric column if available
        if len(numeric_cols) > 1:
            spec3 = {
                'chart_type': 'bar chart',
                'x': categorical_cols[0],
                'agg': 'avg',
                'y': numeric_cols[1]
            }
            url3 = generate_chart_from_ai(spec3, df)
            if url3:
                charts.append(url3)
        
        return charts
    
    except Exception as e:
        print(f"Error creating default charts: {e}")
        return []

def debug_column_matching(df):
    """Debug function to test column matching"""
    print("=== COLUMN MATCHING DEBUG ===")
    print(f"Available columns: {list(df.columns)}")
    
    test_cases = ['pizza_sold', 'pizza sold', 'sum', 'amount', 'quantity']
    for test in test_cases:
        match = find_matching_column(test, df.columns)
        print(f"'{test}' -> '{match}'")
    
    print("=== END DEBUG ===")