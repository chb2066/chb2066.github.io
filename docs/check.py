# -*- coding: utf-8 -*-
"""docs/review-format.md 의 서식 규칙을 검사한다.

    python docs/check.py              # src/content 전체
    python docs/check.py reviews/swav.md ...   # 특정 파일만

규칙을 어긴 곳을 파일:줄 로 찍고, 하나라도 있으면 exit 1.
"""
from __future__ import print_function
import io, os, re, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT = os.path.join(ROOT, 'src', 'content')

EM, EN = u'—', u'–'
LABEL = re.compile(r'^\*\*[^*\n]+\*\*$')
BOLD = re.compile(r'\*\*([^*\n]+)\*\*')
LEAD = re.compile(r'^(\s*(?:[-*]|\d+\.)?\s*)\*\*([^*\n]+)\*\*(.*)$')
ITAL = re.compile(u'(^|[\\s(\\[{"“\\-])\\*(?!\\*)([^*\\n]+?)\\*(?!\\*)')
IMG = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
HEAD_OK = re.compile(r'^\[[^\]]+\]\(https?://[^)]+\)$')
PLAIN_SKIP = ('-', '*', '|', '#', '>', '!', '```', '$$')
NARRATIVE = re.compile(u'(다|는가|인가|까)$')


def is_plain(t):
    s = t.strip()
    if not s or s.startswith(PLAIN_SKIP):
        return False
    return not re.match(r'^\d+\.\s', s)


def is_label(txt):
    t = txt.strip()
    if len(t) > 24 or re.search(r'[.!?]$', t):
        return False
    return not (re.search(u'(다|음|함|임|됨)$', t) and len(t) > 8)


def check(path):
    try:
        rel = os.path.relpath(path, ROOT).replace(os.sep, '/')
    except ValueError:          # 다른 드라이브에 있는 파일
        rel = path.replace(os.sep, '/')
    raw = io.open(path, encoding='utf-8').read()
    lines = raw.split('\n')
    bad = []

    def err(i, msg):
        bad.append('%s:%d  %s' % (rel, i, msg))

    if raw.count('```') % 2:
        err(0, '코드 펜스가 안 닫혔다')

    # ** 짝은 파일 단위로 센다. 볼드가 두 줄에 걸칠 수 있어 줄 단위로는 오탐이 난다.
    outside = re.sub(r'```.*?```', '', raw, flags=re.S)
    if outside.count('**') % 2:
        err(0, '** 짝이 안 맞는다 (파일 전체 기준)')

    parts = raw.split('---\n')
    body = parts[2] if len(parts) > 2 else raw
    first = ([l for l in body.split('\n') if l.strip()] or [''])[0]
    is_review = '/reviews/' in rel
    if is_review and not HEAD_OK.match(first.strip()):
        err(0, '머리글이 링크 한 줄이 아니다: %s' % first.strip()[:60])

    fence = False
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith('```'):
            fence = not fence
            continue
        if fence:
            continue

        if EM in ln or EN in ln:
            err(i, 'em/en dash. "-" 를 쓸 것')
        if re.match(r'^## 정리\s*$', ln.strip()):
            err(i, '정리 절은 쓰지 않는다')

        stripped = ITAL.sub(lambda m: m.group(1) + m.group(2), ln)
        if stripped != ln:
            err(i, '기울임(*...*)')

        for m in IMG.finditer(ln):
            src = m.group(1)
            if src.startswith('/'):
                if not os.path.exists(os.path.join(ROOT, 'public', src.lstrip('/'))):
                    err(i, '이미지 파일이 없다: %s' % src)
                if re.search(r'/[ft]\d+\.[a-z]+$', src):
                    err(i, '논문에서 잘라 넣은 그림은 쓰지 않는다: %s' % src)

        s = ln.strip()
        if BOLD.search(ln) and not LABEL.match(s):
            if s.startswith('|'):
                err(i, '표 안 볼드')
            else:
                m = LEAD.match(ln)
                ok = False
                if m and is_label(m.group(2)):
                    ok = bool(re.match(r'^\s*$|^\s*[-:]\s|^:', ln[m.end():]))
                if not ok:
                    err(i, '소제목이 아닌 볼드')

        if LABEL.match(s):
            inner = s.strip('*').strip()
            if NARRATIVE.search(inner):
                err(i, '서사 전환 제목. 명사 라벨로 바꿀 것: %s' % inner)
            if i < len(lines) and is_plain(lines[i]):
                err(i, '볼드 소제목 밑에 빈 줄이 없다. 한 문단으로 뭉친다')

    return bad


def main():
    # 윈도우 콘솔이 cp949 라 이모지가 섞이면 출력이 죽는다
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if args:
        files = [a if os.path.isabs(a) else os.path.join(CONTENT, a) for a in args]
    else:
        files = [f for f in glob.glob(os.path.join(CONTENT, '**', '*.md'), recursive=True)
                 if '_TEMPLATE' not in f]
    bad = []
    for f in sorted(files):
        bad.extend(check(f))
    for b in bad:
        print(b)
    print('\n%d개 파일 검사, 문제 %d건' % (len(files), len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
