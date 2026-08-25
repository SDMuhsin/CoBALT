#!/usr/bin/env bash
# Render the PRISM architecture diagram and produce the images a review needs.
#
#   bash scripts/render_prism_figure.sh
#
# A diagram cannot be reviewed from its source. Overlap, clipping and a
# misdirected arrowhead are properties of the rendered page, so this script
# produces, into the scratch render directory:
#
# Everything lands in llmdocs/figure_build/, outside the deliverable, so the pdf
# sits next to the .tex it came from:
#
#   prism_architecture_standalone.pdf   the figure on its own, ready to view
#   full.png     the whole figure at 300 dpi, for reading it as a whole
#   z1..z6.png   six overlapping zoom tiles, each magnified, laid out as a
#                3 wide by 2 tall grid: z1 z2 z3 across the top, z4 z5 z6
#                across the bottom. Small collisions are only visible here.
#   inpaper.png  the figure set in the paper's own class at \linewidth, the
#                only way an overflow of the text block shows up
#
# The tiles are cut in LaTeX with trim and clip rather than by cropping a
# bitmap, so the region each one covers is exact and the mapping from tile to
# part of the figure can be relied on in a review.
#
# It also prints the figure's width against the column width, which is the
# check that catches a picture whose bounding box quietly exceeds the text
# block, and every LaTeX warning that bears on layout.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIGDIR="$REPO/llmdocs/paper/v3_neuro/figures"
OUT="${PRISM_RENDER_DIR:-$REPO/llmdocs/figure_build}"
mkdir -p "$OUT"
cd "$FIGDIR" || exit 1

echo "== standalone =="
pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$OUT" \
    prism_architecture_standalone.tex >"$OUT/standalone.log" 2>&1
if [ $? -ne 0 ]; then
    echo "FAILED. Errors:"
    grep -A6 -E "^! " "$OUT/standalone.log" | head -60
    exit 1
fi
grep -E "Overfull|Underfull|Missing|undefined" "$OUT/standalone.log" | head -20

# --- size against the column, and the paper's own float mechanism
cat >"$OUT/inpaper.tex" <<'EOF'
\documentclass[preprint,12pt]{elsarticle}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{graphicx}
\input{prism_diagram_data}
\newsavebox{\figbox}
\begin{document}
\sbox{\figbox}{\input{prism_architecture}}
\typeout{FIGSIZE width=\the\wd\figbox height=\the\ht\figbox column=\the\linewidth}
\begin{figure}[p]
\centering
\usebox{\figbox}
\caption{Placeholder.}
\end{figure}
\null\newpage\null
\end{document}
EOF
echo
echo "== size and in-paper placement =="
TEXINPUTS="$FIGDIR:" pdflatex -interaction=nonstopmode -output-directory "$OUT" \
    "$OUT/inpaper.tex" >"$OUT/inpaper.log" 2>&1
grep -E "FIGSIZE" "$OUT/inpaper.log"
grep -E "Overfull|Float too large" "$OUT/inpaper.log" | head -10
echo "(no Overfull or Float-too-large line above means the PICTURE fits)"

# The check above sets the picture with a one word placeholder caption, so it
# says nothing about the real caption, and the real caption is what overruns.
# A float taller than the text block raises no Overfull box: it silently prints
# over the page number, which is invisible in every image this script produces.
# It shipped that way once, by 1.8pt. The paper's own log is the only authority.
PAPERLOG="$FIGDIR/../main.log"
echo
echo "== the real caption, read from the paper's last build =="
if [ ! -f "$PAPERLOG" ]; then
    echo "no main.log yet; build the paper to check the caption"
elif grep -qE "Float too large" "$PAPERLOG"; then
    grep -E "Float too large" "$PAPERLOG" | head -3
    echo "THE CAPTION DOES NOT FIT. Shorten it, or it prints over the page number."
    echo "Caption room is the text block less the picture: 548pt less $(grep -oE 'height=[0-9.]+pt' "$OUT/inpaper.log" | head -1 | tr -d 'height=')."
else
    echo "figure and caption fit the page"
fi
echo "(rebuild the paper after editing the caption, or this reads a stale log)"

# --- the zoom sheet: exact regions, cut with trim and clip
env -u PYTHONPATH /usr/bin/python3 - "$OUT" <<'PY'
import re, subprocess, sys, os
out = sys.argv[1]
log = open(os.path.join(out, 'inpaper.log')).read()
m = re.search(r'FIGSIZE width=([\d.]+)pt\s*height=([\d.]+)pt', log)
w, h = float(m.group(1)), float(m.group(2))
# the standalone pdf carries a 2 mm border on every side
b = 5.69
W, H = w + 2 * b, h + 2 * b
cols, rows, ov = 3, 2, 0.12
tw, th = W / cols * (1 + ov), H / rows * (1 + ov)
tex = [r'\documentclass{article}',
       r'\usepackage[papersize={%.2fbp,%.2fbp},margin=0bp]{geometry}' % (tw * 2.4, th * 2.4),
       r'\usepackage{graphicx}\pagestyle{empty}', r'\begin{document}']
for r in range(rows):
    for c in range(cols):
        L = max(0.0, c * W / cols - W * ov / (2 * cols))
        B = max(0.0, (rows - 1 - r) * H / rows - H * ov / (2 * rows))
        R = max(0.0, W - L - tw)
        T = max(0.0, H - B - th)
        tex.append(r'\noindent\includegraphics[trim=%.2fbp %.2fbp %.2fbp %.2fbp,clip,scale=2.4]{%s}\newpage'
                   % (L, B, R, T, os.path.join(out, 'prism_architecture_standalone.pdf')))
tex.append(r'\end{document}')
open(os.path.join(out, 'zoom.tex'), 'w').write('\n'.join(tex))
subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', out,
                os.path.join(out, 'zoom.tex')], stdout=subprocess.DEVNULL, check=False)
for i in range(1, rows * cols + 1):
    subprocess.run(['pdftoppm', '-png', '-r', '150', '-singlefile', '-f', str(i), '-l', str(i),
                    os.path.join(out, 'zoom.pdf'), os.path.join(out, 'z%d' % i)], check=False)
print('figure %.1f x %.1f pt; 6 zoom tiles at 2.4x, %.0f%% overlap' % (w, h, ov * 100))
PY

# --- the text budget, checked against the pdf and not by eye.
# The figure was built under a ban on text, enforced here by refusing a pdf
# that carried a font resource. The ban is now a budget: a small number of
# symbols is allowed and anything past it fails the build. The reason for
# checking the pdf rather than the source is unchanged. A stray label three
# millimetres tall is easy to miss and impossible to argue with once printed,
# and a label added without a decision is exactly what a budget catches.
#
# Raise PRISM_GLYPH_BUDGET only together with a decision to add text, never to
# make a build pass.
#
# Raised from 72 to 260 on 2026-08-02, with the author's decision to give every
# block a short title and every matrix its symbol. The figure now shows about
# 150 glyphs, so the budget is still roughly the loose 1.7x of the real count it
# was before, and it still fires on a label added without a decision.
env -u PYTHONPATH PRISM_GLYPH_BUDGET="${PRISM_GLYPH_BUDGET:-260}" \
    /usr/bin/python3 - "$OUT" <<'BUDGET'
import os, re, sys, zlib
pdf = os.path.join(sys.argv[1], 'prism_architecture_standalone.pdf')
d = open(pdf, 'rb').read()
budget = int(os.environ['PRISM_GLYPH_BUDGET'])
glyphs, runs = 0, 0
for m in re.finditer(b'stream\r?\n', d):
    a = m.end()
    b = d.find(b'endstream', a)
    if b < 0:
        continue
    try:
        raw = zlib.decompress(d[a:b])
    except Exception:
        continue
    runs += len(re.findall(b'\\bBT\\b', raw))
    # every text-showing operation: a string with Tj, or an array with TJ
    shown = [m.group(1) for m in re.finditer(rb'\[(.*?)\]\s*TJ', raw, re.S)]
    shown += [m.group(0) for m in re.finditer(rb'\((?:\\.|[^\\()])*\)\s*Tj', raw)]
    shown += [m.group(0) for m in re.finditer(rb'<[0-9A-Fa-f\s]*>\s*Tj', raw)]
    for chunk in shown:
        # one glyph per character of a literal string, escapes counted once,
        # and one per byte pair of a hex string. Kerning numbers are not text.
        for lit in re.findall(rb'\((?:\\.|[^\\()])*\)', chunk):
            glyphs += len(re.sub(rb'\\[0-7]{1,3}|\\.', b'x', lit[1:-1]))
        for hx in re.findall(rb'<([0-9A-Fa-f\s]*)>', chunk):
            glyphs += len(re.sub(rb'\s', b'', hx)) // 2
print('text budget: %d glyphs in %d runs, budget %d' % (glyphs, runs, budget))
if glyphs > budget:
    print('TEXT BUDGET EXCEEDED: %d glyphs against a budget of %d'
          % (glyphs, budget))
    sys.exit(1)
BUDGET

cd "$OUT" || exit 1
pdftoppm -png -r 300 -singlefile prism_architecture_standalone.pdf full
# a greyscale proof: the journal may print without colour, and four separate
# paths in this figure are distinguished partly by hue
pdftoppm -png -gray -r 300 -singlefile prism_architecture_standalone.pdf full_grey
pdftoppm -png -r 150 -f 1 -l 3 inpaper.pdf inpaper   # the float may land on a later page
ls "$OUT"/full.png "$OUT"/z?.png 2>/dev/null | wc -l | xargs echo "images written:"
