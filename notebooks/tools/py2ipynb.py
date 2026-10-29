"""把「# ===== 第 N 段｜標題 =====」分段的 .py 轉成 Kaggle／Colab 可直接開的 .ipynb
用法：python notebooks/tools/py2ipynb.py 輸入.py 輸出.ipynb
- 開頭的註解區塊 → 第一格 Markdown（筆記本說明）
- 每一段 → 一格 Markdown（大標題）＋ 一格程式碼
- 程式碼裡以「# !pip」開頭的行會自動取消註解
"""
import json, re, sys

src = open(sys.argv[1], encoding="utf-8").read()
parts = re.split(r"^# ===== (.+?) =====\s*$", src, flags=re.M)
header, segs = parts[0], list(zip(parts[1::2], parts[2::2]))

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    text = re.sub(r"^# (!pip .*)$", r"\1", text.strip("\n"), flags=re.M)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}

intro = "  \n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in header.strip().splitlines()
                  if not set(l.strip()) <= set("#= "))
cells = [md(intro.strip())]
for title, body in segs:
    cells.append(md(f"## {title}"))
    cells.append(code(body))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{sys.argv[2]}：{len(segs)} 段")
