"""Replay the supplied QA workbook through the current FastAPI streaming entry point."""
import concurrent.futures
import contextlib
import json
import logging
import sys
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT.parent / "outputs" / "acot-chat-review"
OUTPUT.mkdir(parents=True, exist_ok=True)
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def rows(archive, number):
    tree = ET.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
    result = []
    for row in tree.findall("s:sheetData/s:row", NS):
        cells = {}
        for cell in row.findall("s:c", NS):
            column = ''.join(c for c in cell.attrib['r'] if c.isalpha())
            cells[column] = ''.join(cell.itertext()) if cell.attrib.get('t') == 'inlineStr' else cell.findtext('s:v', '', NS)
        result.append(cells)
    return result


def main():
    with zipfile.ZipFile(sys.argv[1]) as archive:
        cases = rows(archive, 2)[1:]
        executions = {row['A']: row.get('D', '') for row in rows(archive, 15)[1:]}
        calculations = {row['A']: row.get('C', '') for row in rows(archive, 7)[1:]}
    cases = [row for row in cases if row.get('A', '').startswith('AI-')]
    assert len(cases) == 100
    groups = defaultdict(list)
    for row in cases:
        groups[executions.get(row['A']) or row['A']].append(row)
    prefix = uuid.uuid4().hex
    completed = 0
    lock = threading.Lock()
    with (OUTPUT / 'service.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        from fastapi.testclient import TestClient
        from api import app
        with TestClient(app) as client:
            def run_group(group):
                nonlocal completed
                session = prefix + '-' + group[0]['A']
                for row in group:
                    started = time.perf_counter()
                    result = {
                        'id': row['A'], 'category': row.get('B'), 'question': row['D'],
                        'expected': row['E'], 'previous_issue': row.get('F'),
                        'previous_status': row.get('G'), 'verification': row.get('R'),
                        'calculation_reference': calculations.get(row['A']), 'session': session,
                    }
                    try:
                        response = client.post('/api/ask/stream', json={'question': row['D'], 'session_id': session})
                        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
                        result['http_status'] = response.status_code
                        result['errors'] = [event for event in events if event.get('type') == 'error']
                        result['response'] = next((event.get('response') for event in events if event.get('type') == 'done'), None)
                        result['deltas'] = ''.join(event.get('text', '') for event in events if event.get('type') == 'delta')
                    except Exception as exc:
                        result['errors'] = [type(exc).__name__]
                    result['seconds'] = round(time.perf_counter() - started, 2)
                    (OUTPUT / (row['A'] + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
                    with lock:
                        completed += 1
                        print(f"{completed}/100 {row['A']} done={bool(result.get('response'))} seconds={result['seconds']}", file=sys.__stdout__, flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(run_group, groups.values()))
    print('All 100 requests recorded.', flush=True)


if __name__ == '__main__':
    main()
