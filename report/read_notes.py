import pymupdf
import sys

doc = pymupdf.open('report/main.pdf')
lines = []
lines.append(f'Total pages: {len(doc)}')
found = 0
for page_num in range(len(doc)):
    page = doc[page_num]
    annots = list(page.annots())
    if annots:
        lines.append(f'=== Page {page_num + 1} Annotations ({len(annots)}) ===')
        for a in annots:
            found += 1
            info = a.info
            lines.append(f'Type: {a.type}, Rect: {a.rect}')
            lines.append(f'Content: {info.get("content", "")}')
            lines.append(f'Subject: {info.get("subject", "")}')
            lines.append(f'Title: {info.get("title", "")}')
            try:
                text = page.get_text("text", clip=a.rect)
                if text.strip():
                    lines.append(f'Selected Text: {text.strip()}')
            except Exception:
                pass
            lines.append('---')

if found == 0:
    lines.append('No annotations found in main.pdf.')

with open('report/extracted_notes.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Wrote {found} annotations to report/extracted_notes.txt')

