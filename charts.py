import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid

def generate_chart_from_ai(spec, df):
    chart_type = spec.get('chart_type', '').lower()
    x = spec.get('x')
    y = spec.get('y')
    agg = spec.get('agg', 'sum')

    try:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if chart_type in ['scatter', 'scatter plot']:
            # Just plot raw data
            if x in df.columns and y in df.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.scatter(df[x], df[y], alpha=0.7)
                ax.set_xlabel(x.title())
                ax.set_ylabel(y.title())
                ax.set_title(f"{y.title()} vs {x.title()}")
            else:
                raise Exception(f"Columns {x}, {y} not found in DataFrame.")
        elif chart_type in ['histogram', 'hist']:
            if x in df.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                df[x].plot(kind='hist', bins=20, ax=ax)
                ax.set_xlabel(x.title())
                ax.set_title(f"Histogram of {x.title()}")
            else:
                raise Exception(f"Column {x} not found for histogram.")
        else:
            # GROUP DATA for bar/line/pie, etc.
            if x not in df.columns or y not in df.columns:
                raise Exception(f"Columns {x} or {y} not found in DataFrame.")
            if agg == 'sum':
                grouped = df.groupby(x)[y].sum()
            elif agg == 'avg':
                grouped = df.groupby(x)[y].mean()
            elif agg == 'count':
                grouped = df.groupby(x)[y].count()
            elif agg == 'min':
                grouped = df.groupby(x)[y].min()
            elif agg == 'max':
                grouped = df.groupby(x)[y].max()
            else:
                grouped = df.groupby(x)[y].sum()

            fig, ax = plt.subplots(figsize=(6, 3.5))
            if 'line' in chart_type:
                grouped.plot(kind='line', ax=ax)
            elif 'bar' in chart_type and 'horizontal' not in chart_type:
                grouped.plot(kind='bar', ax=ax)
            elif 'horizontal' in chart_type or 'barh' in chart_type:
                grouped.plot(kind='barh', ax=ax)
            elif 'pie' in chart_type:
                grouped.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_ylabel('')
            else:
                grouped.plot(kind='bar', ax=ax)
            ax.set_xlabel(x.title())
            ax.set_ylabel(y.title())
            ax.set_title(f"{agg.title()} {y.title()} by {x.title()}")

        plt.tight_layout()
        img_id = str(uuid.uuid4())
        img_path = f'static/chart_{img_id}.png'
        plt.savefig(img_path)
        plt.close(fig)
        return '/' + img_path
    except Exception as e:
        print("Chart error:", e)
        return None
