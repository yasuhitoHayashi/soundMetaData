#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io, re, pathlib
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

import pandas as pd, numpy as np
import tomllib as tomli
import requests

DIR, GLOB = pathlib.Path("."), "*.toml"
URL_S     = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"
URL10     = ("https://www.data.jma.go.jp/stats/etrn/view/10min_a1.php"
             "?prec_no={}&block_no={:04}&year={}&month={}&day={}&view=")

# util -----------------------------------------------------------------
def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 2 * 6371 * asin((sin((lat2 - lat1) / 2) ** 2 +
                            cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2) ** .5)

def scal(x):
    if isinstance(x, pd.Series): x = x.iloc[0]
    if pd.isna(x): return None
    if isinstance(x, np.generic): return x.item()
    return x

# meta -----------------------------------------------------------------
def meta():
    rec = []
    for p in DIR.glob(GLOB):
        d = tomli.load(p.open("rb"))
        try:
            st = d["observation_info"]["date_info"]["start_date"]
            lat, lon = d["observation_info"]["location_info"]["position"]
        except KeyError: 
            continue
        dt = datetime.fromisoformat(st) if isinstance(st, str) else st
        jd, jt = (dt.date() - timedelta(1), "24:00") if dt.time() == datetime.min.time() else (dt.date(), dt.strftime("%H:%M"))
        rec.append(dict(file=p.name, jd=jd, jt=jt, lat=lat, lon=lon))
    return pd.DataFrame(rec)

# station --------------------------------------------------------------
def stn_tbl():
    df = pd.DataFrame(requests.get(URL_S).json()).T
    df["lat"] = df["lat"].apply(lambda x: x[0] + x[1] / 60)
    df["lon"] = df["lon"].apply(lambda x: x[0] + x[1] / 60)
    df.index  = df.index.map(str)
    return df

def nearest(lat, lon, df):
    d   = df.apply(lambda r: hav(lon, lat, r.lon, r.lat), axis=1)
    idx = d.idxmin()
    return idx, float(d[idx])

# JMA ------------------------------------------------------------------
def jma10(dt, prec, block):
    url  = URL10.format(prec, block, dt.year, dt.month, dt.day)
    print(url)
    resp = requests.get(url); resp.encoding = "shift_jis"
    try:
        raw = pd.read_html(io.StringIO(resp.text), header=[0])[0]
    except ValueError:
        return pd.DataFrame()
    cols = raw.iloc[0].tolist()
    df   = raw.iloc[3:].copy().reset_index(drop=True); df.columns = cols
    return df.rename(columns={cols[0]: "time",
                              cols[1]: "precip_mm",
                              cols[4]: "wind_avg_mps",
                              cols[6]: "wind_gust_mps"})

# patch ----------------------------------------------------------------
def patch(path, row, dist, aid):
    txt  = path.read_text("utf-8").splitlines(keepends=True)
    repl = dict(precip_mm=scal(row.precip_mm),
                wind_avg_mps=scal(row.wind_avg_mps),
                wind_gust_mps=scal(row.wind_gust_mps),
                station_distance_km=None if pd.isna(dist) else round(dist, 2),
                amedas_id=int(aid))
    out, inside, seen, done = [], False, False, set()
    for ln in txt:
        if ln.strip().startswith("[") and ln.strip().endswith("]"):
            if inside:
                for k, v in repl.items():
                    if k not in done and v is not None:
                        out.append(f"{k} = {v}\n")
            inside = ln.strip() == "[observation_info.location_info]"
            seen  |= inside
            out.append(ln); continue
        if inside:
            for k, v in repl.items():
                if v is None or k in done: continue
                pat = rf"^(\s*{re.escape(k)}\s*=\s*)([^\n#]*)(\s*#.*)?(\n?)$"
                if re.match(pat, ln):
                    ln = re.sub(pat,
                                lambda m: f"{m.group(1)}{v}{m.group(3) or ''}{m.group(4)}",
                                ln)
                    done.add(k); break
        out.append(ln)
    if not seen:
        out.append("\n[observation_info.location_info]\n")
        for k, v in repl.items():
            if v is not None: out.append(f"{k} = {v}\n")
    elif inside:
        for k, v in repl.items():
            if k not in done and v is not None:
                out.append(f"{k} = {v}\n")
    path.write_text("".join(out), "utf-8")

# main -----------------------------------------------------------------
def main():
    m = meta()
    total = len(m)
    if total == 0:
        print("no toml"); return
    print(f"found {total} toml files")

    st = stn_tbl()
    fk = pd.read_csv("fuken.csv").astype({"amedas_id": "Int64",
                                          "station_id": "Int64",
                                          "fuken_id": "Int64"})
    m[["aid", "dist"]] = m.apply(lambda r: pd.Series(nearest(r.lat, r.lon, st)), axis=1)

    for i, (_, r) in enumerate(m.iterrows(), 1):
        print(f"[{i}/{total}] {r.file}")
        hit = fk[fk.amedas_id == int(r.aid)]
        if hit.empty:
            print("    skip: id not in fuken.csv"); continue
        prec, block = int(hit.station_id.iloc[0]), int(hit.fuken_id.iloc[0])
        df = jma10(r.jd, prec, block)
        if df.empty or "time" not in df.columns:
            print("    skip: JMA table missing"); continue
        sel = df[df.time == r.jt]
        if sel.empty:
            print("    skip: time row missing"); continue
        patch(DIR / r.file, sel.iloc[0], r.dist, r.aid)
        print("    ✅ updated")

if __name__ == "__main__":
    main()