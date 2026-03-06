#!/usr/bin/env python3
"""
map_query.py — query a 1bcoder map file.

Two modes:
  find   — filter file blocks by filename and/or child-line content
  trace  — follow call chain backwards from a defined identifier (BFS)

Usage:
    python map_query.py find [tokens ...] [--map FILE]
    python map_query.py trace <identifier>  [--map FILE]

Filter token syntax for find:
    term       filename contains term
    !term      exclude if filename contains term
    \\term     include if any child line contains term
    \\!term    include but hide child lines containing term
    \\\\!term  exclude entire block if any child contains term

    (no tokens → print full map)

Examples:
    python map_query.py find register
    python map_query.py find \\register !mock
    python map_query.py find auth \\UserService \\!deprecated
    python map_query.py trace insertEmail
    python map_query.py trace register --map path/to/map.txt
"""

import re
import os
import sys
import argparse

DEFAULT_MAP = os.path.join('.1bcoder', 'map.txt')


# ── parse ───────────────────────────────────────────────────────────────────────

def parse_map(map_path: str) -> tuple:
    """Parse map.txt → (defines_map, links_map).

    defines_map : { rel_file → { name → lineno } }
    links_map   : { caller_rel → { target_rel → { name → kind } } }
    """
    with open(map_path, encoding='utf-8') as f:
        content = f.read()

    defines_map  = {}
    links_map    = {}
    current_file = None

    for line in content.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if not line.startswith(' '):
            current_file = line.strip()
            defines_map.setdefault(current_file, {})
            links_map.setdefault(current_file, {})
        elif current_file and 'defines :' in line:
            items_str = line.split('defines :', 1)[1].strip()
            for item in items_str.split(', '):
                m = re.match(r'(\w[\w-]*)\(ln:(\d+)\)', item.strip())
                if m:
                    defines_map[current_file][m.group(1)] = int(m.group(2))
        elif current_file and 'links  →' in line:
            m = re.match(r'\s+links\s+→\s+(\S+)\s+\((.+)\)', line)
            if m:
                target   = m.group(1)
                refs_str = m.group(2)
                refs = {}
                for item in refs_str.split(', '):
                    if ':' in item:
                        kind, name = item.split(':', 1)
                        refs[name.strip()] = kind.strip()
                links_map[current_file].setdefault(target, {}).update(refs)

    return defines_map, links_map


# ── find ────────────────────────────────────────────────────────────────────────

def find_map(map_path: str, query: str) -> tuple:
    """Search map.txt with filter syntax.

    Returns (hits, rendered_string).
    hits  — list of matching block strings (empty list means full map returned).
    rendered_string — the text to display / inject.
    """
    with open(map_path, encoding='utf-8') as f:
        content = f.read()

    tokens       = query.split()
    pos_file     = []   # term     — filename must contain
    neg_file     = []   # !term    — filename must NOT contain
    pos_child    = []   # \term    — any child line must contain
    filter_child = []   # \!term   — hide child lines containing
    neg_child    = []   # \\!term  — exclude block if any child contains

    for t in tokens:
        if t.startswith('\\\\!') and len(t) > 3:
            neg_child.append(t[3:].lower())
        elif t.startswith('\\!') and len(t) > 2:
            filter_child.append(t[2:].lower())
        elif t.startswith('\\') and len(t) > 1:
            pos_child.append(t[1:].lower())
        elif t.startswith('!') and len(t) > 1:
            neg_file.append(t[1:].lower())
        else:
            pos_file.append(t.lower())

    # no criteria → return full map
    if not any([pos_file, neg_file, pos_child, filter_child, neg_child]):
        return [], content

    blocks = re.split(r'\n(?=\S)', content)

    def process_block(block):
        lines       = block.split('\n')
        fname       = lines[0].lower()
        child_lines = [l for l in lines[1:] if l.strip()]

        if pos_file and not all(t in fname for t in pos_file):
            return None
        if any(t in fname for t in neg_file):
            return None
        if pos_child:
            children_text = '\n'.join(child_lines).lower()
            if not all(t in children_text for t in pos_child):
                return None
        if neg_child:
            children_text = '\n'.join(child_lines).lower()
            if any(t in children_text for t in neg_child):
                return None
        if filter_child:
            child_lines = [l for l in child_lines
                           if not any(t in l.lower() for t in filter_child)]

        return lines[0] + ('\n' + '\n'.join(child_lines) if child_lines else '')

    hits = [r for b in blocks
            if not b.startswith('#')
            for r in [process_block(b)] if r is not None]

    return hits, '\n'.join(hits)


# ── trace ───────────────────────────────────────────────────────────────────────

def trace_map(map_path: str, identifier: str) -> str:
    """BFS backwards through the call graph from a defined identifier.

    Returns a rendered string, or None if the identifier is not found in defines.
    """
    defines_map, links_map = parse_map(map_path)

    # locate the defining file
    start_file = None
    start_ln   = None
    for frel, defs in defines_map.items():
        if identifier in defs:
            start_file = frel
            start_ln   = defs[identifier]
            break

    if not start_file:
        return None

    # reverse index: target_file → [(caller_file, name, kind)]
    incoming = {}
    for caller, targets in links_map.items():
        for target, refs in targets.items():
            for name, kind in refs.items():
                incoming.setdefault(target, []).append((caller, name, kind))

    MAX_DEPTH = 8
    lines_out = [
        f'trace: {identifier}',
        f'{start_file}  [defines {identifier}(ln:{start_ln})]',
    ]
    visited = {start_file}
    queue   = [(start_file, 1)]

    while queue:
        current, depth = queue.pop(0)
        if depth > MAX_DEPTH:
            break
        indent  = '  ' * depth
        callers = incoming.get(current, [])
        for caller, name, kind in sorted(callers, key=lambda x: x[0]):
            lines_out.append(f'{indent}← {kind}:{name}  {caller}')
            if caller not in visited:
                visited.add(caller)
                queue.append((caller, depth + 1))

    return '\n'.join(lines_out)


# ── CLI entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Query a 1bcoder map file (find or trace identifiers).'
    )
    parser.add_argument('--map', metavar='FILE', default=DEFAULT_MAP,
                        help=f'Map file to query (default: {DEFAULT_MAP})')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_find = sub.add_parser('find', help='Filter map blocks by filename/content')
    p_find.add_argument('query', nargs='*',
                        help='Filter tokens (term, !term, \\term, \\!term, \\\\!term)')

    p_trace = sub.add_parser('trace', help='Follow call chain backwards from an identifier')
    p_trace.add_argument('identifier', help='Identifier name to trace')

    args = parser.parse_args()

    if not os.path.exists(args.map):
        print(f'error: map file not found: {args.map}', file=sys.stderr)
        print('hint:  run map_index.py first to build the map', file=sys.stderr)
        sys.exit(1)

    if args.cmd == 'find':
        query = ' '.join(args.query)
        hits, result = find_map(args.map, query)
        if not query:
            print(result)
        elif hits:
            print(result)
            print(f'\n[map] {len(hits)} match(es)', file=sys.stderr)
        else:
            print(f'[map] no matches for: {query}', file=sys.stderr)
            sys.exit(1)

    elif args.cmd == 'trace':
        result = trace_map(args.map, args.identifier)
        if result is None:
            print(f"[map] '{args.identifier}' not found in any defines", file=sys.stderr)
            print(f"hint:  try: python map_query.py find \\{args.identifier}", file=sys.stderr)
            sys.exit(1)
        print(result)


if __name__ == '__main__':
    main()
