#!/usr/bin/env python3
"""One coherent pass: bind every inline gx_read(A, N) in the chain driver's
phase-2 region and the generator's three function literals to reusable
loop-scope variables (g_A_N), emitting the assignments BEFORE the enclosing
statement (paren-aware grouping). Fixes the registry-string leak: inline
gx_read temporaries register (refs=1) and their release never fires."""
import re

def bind_reads(text):
    out = []
    lines = text.split('\n')
    i = 0
    n = 0
    while i < len(lines):
        line = lines[i]
        if 'gx_read(' not in line:
            out.append(line); i += 1; continue
        stripped = line.strip()
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]* = gx_read\(', stripped):
            out.append(line); i += 1; continue  # already bound
        # accumulate the statement until parens balance
        stmt = [line]
        depth = line.count('(') - line.count(')')
        j = i
        while depth > 0 and j + 1 < len(lines):
            j += 1
            stmt.append(lines[j])
            depth += lines[j].count('(') - lines[j].count(')')
        block = '\n'.join(stmt)
        calls = list(re.finditer(r'gx_read\(([A-Za-z_][A-Za-z0-9_]*), (\d+)\)', block))
        if not calls:
            out.append(line); i += 1; continue
        indent = stmt[0][:len(stmt[0]) - len(stmt[0].lstrip())]
        pre = []
        newblock = block
        for m in reversed(calls):
            var = 'g_' + m.group(1) + '_' + m.group(2)
            pre.append(indent + var + ' = gx_read(' + m.group(1) + ', ' + m.group(2) + ')')
            newblock = newblock[:m.start()] + var + newblock[m.end():]
        out.extend(pre)
        out.extend(newblock.split('\n'))
        n += len(calls)
        i = j + 1
    return '\n'.join(out), n

# 1. the snippet: phase-2 region only (after the R = 1 downgrade line)
f = '/home/rick/Projects/VybForge/native/glm53/_chain_main_snippet.txt'
s = open(f).read()
i = s.find('\n    R = 1\n')
head, tail = s[:i], s[i:]
tail, n1 = bind_reads(tail)
s = head + tail
open(f, 'w').write(s)
print('snippet: bound', n1)

# 2. the generator's three function literals
g = '/home/rick/Projects/VybForge/native/glm53/build_decode_chain.py'
s = open(g).read()
total = 0
for fn in ('moe_tail_step', 'mtp_build_row', 'mtp_layer_step'):
    m = re.search(r'^' + fn + r'\(', s, re.M)
    a = m.start()
    b = s.find('\n}', a) + 2
    block, n = bind_reads(s[a:b])
    s = s[:a] + block + s[b:]
    total += n
open(g, 'w').write(s)
print('generator: bound', total)
