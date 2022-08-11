from difflib import SequenceMatcher


def getTwoTextEdits(src_text, m1_text):
    r = SequenceMatcher(None, src_text, m1_text)
    diffs = r.get_opcodes()
    m1_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "equal" in tag:
            continue
        m1_edits.append((tag, src_text[i1:i2], m1_text[j1:j2], i1, i2, j1, j2))
    return m1_edits