#!/usr/bin/env bash
# 修正上一版封存脚本的 glob bug：文件名是 `run_manifest.json.SUPERSEDED_...`，
# 而 `*_SUPERSEDED_*` 要求 SUPERSEDED **前面紧贴下划线** ⇒ 匹配不到。
# 用 `*SUPERSEDED*` 重做，把 SYSU 的封存残留移到同级 `_ARCHIVE/`。
#
# 只做改名/移动，**绝不删除**（项目纪律：产物只改名封存）。
set -u
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8
SY="$CKPT_ROOT/CTRL_R8/SYSU-CD-256"
AR="$CKPT_ROOT/CTRL_R8/_ARCHIVE"
mkdir -p "$AR"

echo "=== 修 glob + 移走封存残留 $(date '+%F %T') ==="
echo "  处理前: [$(ls -A "$SY" 2>/dev/null | tr '\n' ' ')]"
for f in "$SY"/*; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in
    *SUPERSEDED*)
      mv "$f" "$AR/SYSU-CD-256.$(basename "$f")"
      echo "  MOVED $(basename "$f") -> _ARCHIVE/"
      ;;
    *) echo "  KEEP  $(basename "$f")（真实训练产物，保留）" ;;
  esac
done
echo "  处理后: [$(ls -A "$SY" 2>/dev/null | tr '\n' ' ')]  （空 ⇒ 守卫放行）"
ls -la "$AR" | sed 's/^/    /'
