import re

def extract_kpis_from_answer(answer):
    # Find lines like "Total Sales: 1,200,000"
    kpis = []
    for line in answer.splitlines():
        m = re.match(r'([A-Za-z\s]+):\s*([\d\.,]+)', line)
        if m:
            label = m.group(1).strip()
            value = m.group(2).strip()
            kpis.append({'label': label, 'value': value})
    return kpis
