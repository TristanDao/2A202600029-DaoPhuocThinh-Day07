import re

def split_vi_en(text: str) -> tuple[str, str]:
    """
    Split a bilingual text into English and Vietnamese parts.
    Heuristic: English is at the top, Vietnamese follows. 
    Finds the first line with Vietnamese characters and treats it as the start of the VI section.
    """
    # Full regex for Vietnamese accented characters
    vi_chars = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)

    lines = text.split("\n")
    
    # Find the first line that definitely contains Vietnamese
    pivot_index = -1
    for i, line in enumerate(lines):
        if vi_chars.search(line):
            pivot_index = i
            break
    
    if pivot_index == -1:
        # No Vietnamese found, assume everything is English
        return text.strip(), ""
    
    # Backtrack to find the start of the block containing the pivot line
    # (Optional: find the first non-empty line after a gap before the pivot)
    # For now, let's keep it simple: split from the first empty line preceding the pivot
    start_vi = pivot_index
    while start_vi > 0 and lines[start_vi-1].strip() != "":
        start_vi -= 1
        
    en_part = "\n".join(lines[:start_vi]).strip()
    vi_part = "\n".join(lines[start_vi:]).strip()
    
    return en_part, vi_part
