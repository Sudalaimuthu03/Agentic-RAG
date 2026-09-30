"""Deterministic V4 release gate. No network and no model loading."""
from __future__ import annotations
import ast, hashlib, os, sys, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS={'.venv','venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','build','dist'}
FORBIDDEN_SUFFIXES={'.pyc','.pyo','.log','.zip'}
STALE={'rag/agent/resolver.py','rag/agent/target.py','rag/agent/relevance.py','rag/agent/precheck.py','rag/agent/guardrails.py','rag/agent/pagination.py','rag/agent/compose.py','rag/agent/state.py'}

def files():
    return [p for p in ROOT.rglob('*') if p.is_file() and not any(x in FORBIDDEN_DIRS for x in p.parts)]

def check_tree():
    bad=[]
    for p in files():
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix in FORBIDDEN_SUFFIXES: bad.append(rel)
        if rel in STALE: bad.append(rel)
    if any(p.name=='agent' and p.is_dir() for p in ROOT.rglob('agent/agent')): bad.append('duplicate agent/agent tree')
    return bad

def check_syntax():
    errors=[]
    for p in files():
        if p.suffix=='.py':
            try: ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
            except SyntaxError as e: errors.append(f'{p}: {e}')
    return errors

def check_imports():
    errors=[]
    pyfiles={p.with_suffix('').relative_to(ROOT).as_posix().replace('/','.') for p in files() if p.suffix=='.py'}
    pyfiles={x[:-9] if x.endswith('.__init__') else x for x in pyfiles}
    for p in files():
        if p.suffix!='.py': continue
        try: tree=ast.parse(p.read_text(encoding='utf-8'))
        except SyntaxError: continue
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):
                names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom) and n.level==0:
                names=[n.module or '']
            else: continue
            for name in names:
                top=name.split('.')[0]
                if top=='rag' and name.startswith('rag.'):
                    target=name
                    if target not in pyfiles and not any(x.startswith(target+'.') for x in pyfiles):
                        errors.append(f'{p.relative_to(ROOT)}: unresolved internal import {name}')
    return errors

def main():
    bad=check_tree(); syn=check_syntax(); imp=check_imports()
    print('V4 TREE:', 'PASS' if not bad else 'FAIL')
    if bad: print(*bad,sep='\n')
    print('V4 SYNTAX:', 'PASS' if not syn else 'FAIL')
    if syn: print(*syn,sep='\n')
    print('V4 INTERNAL IMPORTS:', 'PASS' if not imp else 'FAIL')
    if imp: print(*imp,sep='\n')
    ok=not (bad or syn or imp)
    print('V4 STATIC GATE:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1
if __name__=='__main__': sys.exit(main())
