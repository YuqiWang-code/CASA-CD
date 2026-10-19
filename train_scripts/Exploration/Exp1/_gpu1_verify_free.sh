#!/usr/bin/env bash
# 核实 GPU1 是否真的空出（多口径互证 + 我们自己的进程是否真的没了）
set -u
echo "=== 时间 ==="; date '+%F %T %z'
echo
echo "=== 1) nvidia-smi 总览（显存/利用率/温度/功耗）==="
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv
echo
echo "=== 2) compute apps（逐 PID 归属）==="
python3 - <<'PY'
import subprocess
def run(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
uu = {}
for line in run("nvidia-smi --query-gpu=index,uuid --format=csv,noheader").splitlines():
    i, u = [x.strip() for x in line.split(",")]
    uu[u] = i
apps = run("nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader")
if not apps:
    print("  (无任何 compute 进程)")
for line in apps.splitlines():
    parts = [x.strip() for x in line.split(",")]
    uuid, pid, name, mem = parts[0], parts[1], parts[2], parts[3]
    idx = uu.get(uuid, "?")
    owner = run(f"ps -o user= -p {pid}").strip() or "?"
    cmd = run(f"tr '\\0' ' ' < /proc/{pid}/cmdline")[:90] or "?"
    print(f"  GPU{idx} pid={pid} user={owner} mem={mem} cmd={cmd}")
PY
echo
echo "=== 3) 我们自己的进程（应当全空）==="
for pat in "[t]rain_stgeo\.py" "[E]xploration/Exp1/run_all\.sh" "[w]ait_and_launch"; do
  out=$(pgrep -af "$pat" 2>/dev/null | cut -c1-100)
  if [ -n "$out" ]; then echo "  !! 仍在运行：$pat"; echo "$out" | sed 's/^/     /'; else echo "  ok: 无 $pat"; fi
done
echo
echo "=== 4) 我们的 python 进程里是否还有占卡的 ==="
ps -u "$(whoami)" -o pid=,rss=,etime=,cmd= 2>/dev/null | grep -E "train_stgeo|models_v1" | cut -c1-110 || echo "  （无）"
echo
echo "=== 5) 卡上是否有人刚起来的迹象（最近 3 分钟内的进程）==="
ps -eo pid=,user=,etimes=,cmd= --sort=etimes 2>/dev/null | awk '$3 < 180' | grep -viE "awk|ps -eo|grep|bash -c" | head -8 || true
echo
echo "=== 6) 暂停快照是否存在 ==="
ls -la /home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/_LAST_PAUSED_STATE.md 2>/dev/null || echo "  (缺)"
echo
echo "=== 7) 连续两次采样（间隔 20 s）看 GPU1 是否稳定为空 ==="
for k in 1 2; do
  echo "  sample $k: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' | ')"
  [ $k -eq 1 ] && sleep 20
done
