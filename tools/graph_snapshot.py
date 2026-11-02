# ===== 進度圖快照工具 =====
# 用途：把 graphify 產生的 graph.html 變成「不需連網也能顯示」的版本，並存到 docs/graph/
# 為什麼：graph.html 預設從 unpkg 網站載入繪圖套件 vis-network，
#        Claude 的預覽視窗與部分網路環境會擋外部網站 → 看不到圖。
# 用法（在 repo 根目錄）：python tools/graph_snapshot.py
#        需要 npm（用來下載 vis-network 9.1.6）

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

VIS_VERSION = "9.1.6"
SRC_HTML = Path("graphify-out/graph.html")
SRC_REPORT = Path("graphify-out/GRAPH_REPORT.md")
DST_DIR = Path("docs/graph")


# ----- 第 1 段｜下載 vis-network（從 npm 套件庫，不從 CDN）-----
def fetch_vis_js() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["npm", "pack", f"vis-network@{VIS_VERSION}", "-q"],
                       cwd=tmp, check=True, capture_output=True)
        tgz = next(Path(tmp).glob("vis-network-*.tgz"))
        subprocess.run(["tar", "xzf", tgz.name], cwd=tmp, check=True)
        js = (Path(tmp) / "package/standalone/umd/vis-network.min.js").read_text(encoding="utf-8")
    # 避免 JS 裡的 </script> 字樣提早結束 <script> 標籤
    return js.replace("</script>", "<\\/script>")


# ----- 第 2 段｜把外部 <script src=unpkg...> 換成內嵌程式碼 -----
def inline_vis(html: str, js: str) -> str:
    pattern = r'<script src="https://unpkg\.com/vis-network[^>]*></script>'
    new_html, n = re.subn(pattern, lambda _: f"<script>{js}</script>", html, count=1, flags=re.S)
    if n == 0 and "vis.Network" in html and "unpkg.com/vis-network" not in html:
        return html  # 已經內嵌過
    if n != 1:
        raise SystemExit("找不到 vis-network 的 <script> 標籤；graphify 版本可能改了 HTML 格式")
    return new_html.replace("<title>graphify - graphify-out/graph.html</title>",
                            "<title>電網秘書 專案進度圖</title>")


# ----- 第 3 段｜存快照 -----
def main() -> None:
    if not SRC_HTML.exists():
        raise SystemExit("沒有 graphify-out/graph.html，請先執行 /graphify . 或 /graphify . --update")
    html = inline_vis(SRC_HTML.read_text(encoding="utf-8"), fetch_vis_js())
    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / "graph.html").write_text(html, encoding="utf-8")
    shutil.copy(SRC_REPORT, DST_DIR / "GRAPH_REPORT.md")
    print(f"已存快照：{DST_DIR}/graph.html（{len(html)//1024} KB）、{DST_DIR}/GRAPH_REPORT.md")


if __name__ == "__main__":
    main()
