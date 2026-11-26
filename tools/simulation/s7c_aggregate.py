import csv, io, json, os, subprocess
ROOT="/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/analysis/simulation/S7c"
LOCAL=".tmp/s7c"; os.makedirs(LOCAL,exist_ok=True)
ARMS=[("R_FULL-2334","SYSU-CD-256"),("R_FULL-2334","LEVIR-CD-256"),
      ("R_FULL-2335","SYSU-CD-256"),("R_FULL-2335","LEVIR-CD-256")]
FILES=["probe_scores.csv","probe_paired_ci.csv","radius_stats.csv","pool_stats.csv","summary.json"]
def ssh(c):
    return subprocess.run(["python",".codex/ssh.py",c],capture_output=True).stdout
def pull():
    for a,d in ARMS:
        for f in FILES:
            dst=os.path.join(LOCAL,f"{a}_{d}__{f}")
            if os.path.exists(dst) and os.path.getsize(dst)>100: continue
            out=ssh(f"cat {ROOT}/{a}_{d}/{f}")
            if out.strip(): io.open(dst,"wb").write(out)
pull()
def rows(p):
    with io.open(p,encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
probe=[];paired=[];rad=[];pool=[]
for a,d in ARMS:
    for r in rows(os.path.join(LOCAL,f"{a}_{d}__probe_scores.csv")):
        probe.append(dict(arm=a,ds=d.split("-")[0],scale=int(r["source_scale"]),probe=r["probe"],
                          auc=float(r["auc"]),ap=float(r["ap"]),n_val=int(r["n_val"])))
    for r in rows(os.path.join(LOCAL,f"{a}_{d}__probe_paired_ci.csv")):
        k=list(r.keys()); paired.append(dict(arm=a,ds=d.split("-")[0],scale=int(r[k[0]]),
            contrast=r[k[1]],delta=float(r[k[2]]),lo=float(r[k[3]]),hi=float(r[k[4]])))
    for r in rows(os.path.join(LOCAL,f"{a}_{d}__radius_stats.csv")):
        if r["split"]!="val": continue
        rad.append(dict(arm=a,ds=d.split("-")[0],kind=r["kind"],scale=int(r["source_scale"]),
            median=float(r["median"]),cv=float(r["cv"]),p05=float(r["p05"]),p95=float(r["p95"])))
    for r in rows(os.path.join(LOCAL,f"{a}_{d}__pool_stats.csv")):
        if r.get("split")!="val": continue
        pool.append(dict(arm=a,ds=d.split("-")[0],kind=r["kind"],scale=int(r["source_scale"]),
            ratio=float(r["ratio"]),a_dir=float(r["a_dir"]),ess_frac=float(r["ess_frac"]),
            cv_within=float(r["cv_within_cell"]),proto=float(r["proto_median"]),
            mem=float(r["mem_median"]),tchk=float(r["time_check"])))
os.makedirs("docs/simulation/hyperbolic_v5/results",exist_ok=True)
def dump(name,data,cols):
    with io.open(f"docs/simulation/hyperbolic_v5/results/{name}","w",encoding="utf-8",newline="\n") as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for d in data: w.writerow({c:d[c] for c in cols})
dump("s7c_probe.csv",probe,["arm","ds","scale","probe","auc","ap","n_val"])
dump("s7c_paired.csv",paired,["arm","ds","scale","contrast","delta","lo","hi"])
dump("s7c_radius.csv",rad,["arm","ds","kind","scale","median","cv","p05","p95"])
dump("s7c_pool.csv",pool,["arm","ds","kind","scale","ratio","a_dir","ess_frac","cv_within","proto","mem","tchk"])
print("probe rows",len(probe),"paired",len(paired),"rad",len(rad),"pool",len(pool))
print("time_check max",max(abs(p["tchk"]) for p in pool))
