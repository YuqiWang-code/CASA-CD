import csv, glob, os, statistics as st, collections
D='/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/analysis/simulation/S7'
def rows(p):
    with open(p) as fh: return list(csv.DictReader(fh))
def f(x):
    try: return float(x)
    except: return float('nan')
def tag_of(p, pref):
    return os.path.basename(p)[len(pref):-4]

print('### Q radius (val, per scale)')
agg=collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(D,'s7_radius_stats_*.csv'))):
    t=tag_of(p,'s7_radius_stats_')
    for r in rows(p):
        if r['split']!='val' or r['scale']!='Q': continue
        agg[(t,r['scale_index'])].append((f(r['mean']),f(r['cv'])))
for k in sorted(agg,key=lambda x:(x[0],int(x[1]))):
    v=agg[k]; print(f"{k[0]:32s} s{k[1]} n={len(v):3d} rho_mean={st.mean(x[0] for x in v):.4f} cv={st.mean(x[1] for x in v):.4f}")

print('\n### support (val) mean/min/max')
agg=collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(D,'s7_radius_stats_*.csv'))):
    t=tag_of(p,'s7_radius_stats_')
    for r in rows(p):
        if r['split']!='val' or r['scale']!='support': continue
        agg[t].append((f(r['mean']),f(r['vmin']),f(r['vmax'])))
for k in sorted(agg):
    v=agg[k]; print(f"{k:32s} mean={st.mean(x[0] for x in v):.5f} min={min(x[1] for x in v):.3e} max={max(x[2] for x in v):.5f}")

print('\n### pool contraction ratio (proto/member) by scale and kind')
agg=collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(D,'s7_pool_stats_*.csv'))):
    t=tag_of(p,'s7_pool_stats_')
    for r in rows(p):
        if r['split']!='val': continue
        agg[(t,r['kind'],r['pool_scale'])].append(r)
for k in sorted(agg,key=lambda x:(x[0],x[1],int(x[2]))):
    v=agg[k]
    print(f"{k[0]:32s} {k[1]} s{k[2]} ratio={st.mean(f(x['ratio']) for x in v):.3f} "
          f"cvw={st.mean(f(x['cv_within']) for x in v):.4f} A={st.mean(f(x['a_dir']) for x in v):.3f} "
          f"ESS={st.mean(f(x['ess']) for x in v):7.1f}")

print('\n### probes AUC (train-fit / val-eval); guide_p vs radius vs both')
agg=collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(D,'s7_probe_results_*.csv'))):
    t=tag_of(p,'s7_probe_results_')
    for r in rows(p):
        agg[(t,r['scale'],r['probe'])].append((f(r['auc']),f(r['ap']),f(r['pos'])))
for k in sorted(agg,key=lambda x:(x[0],int(x[1]),x[2])):
    v=agg[k]
    print(f"{k[0]:32s} s{k[1]} {k[2]:14s} AUC={st.mean(x[0] for x in v):.4f} AP={st.mean(x[1] for x in v):.4f} pos={st.mean(x[2] for x in v):.3f}")

print('\n### per-image AUC radius (val, per scale, mean over images)')
agg=collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(D,'s7_per_image_*.csv'))):
    t=tag_of(p,'s7_per_image_')
    for r in rows(p):
        if r['split']!='val' or not r.get('auc_radius'): continue
        agg[(t,r['scale_index'])].append((f(r['auc_radius']),f(r['auc_guide'])))
for k in sorted(agg,key=lambda x:(x[0],int(x[1]))):
    v=agg[k]
    print(f"{k[0]:32s} s{k[1]} n={len(v):3d} AUC_radius={st.mean(x[0] for x in v):.4f} AUC_guide={st.mean(x[1] for x in v):.4f}")

print('\n### pixel-level error rates (val, mean per image)')
agg=collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(D,'s7_per_image_*.csv'))):
    t=tag_of(p,'s7_per_image_')
    for r in rows(p):
        if r['split']!='val' or not r.get('tp'): continue
        agg[t].append(r)
for k in sorted(agg):
    v=agg[k]
    tp=st.mean(f(x['tp']) for x in v); fp=st.mean(f(x['fp']) for x in v); fn=st.mean(f(x['fn']) for x in v)
    ch=st.mean(f(x['change_px']) for x in v)
    print(f"{k:32s} n={len(v):3d} TP={tp:7.0f} FP={fp:6.0f} FN={fn:6.0f} change_px={ch:7.0f}")
