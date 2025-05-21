import io
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

URL10 = ("https://www.data.jma.go.jp/stats/etrn/view/10min_{}.php"
         "?prec_no={}&block_no={:04}&year=2024&month=11&day=1&view=")

def check_row(row):
    # 欠損値がある場合は0を返す
    if pd.isna(row.station_id) or pd.isna(row.fuken_id):
        return 0

    station_id = int(row.station_id)
    fuken_id = int(row.fuken_id)

    for kind in ("a1", "s1"):
        url = URL10.format(kind, station_id, fuken_id)
        try:
            resp = requests.get(url, timeout=10)
            resp.encoding = "shift_jis"
            if not resp.ok:
                continue
            try:
                pd.read_html(io.StringIO(resp.text), header=[0])[0]
            except ValueError:
                continue
            return 1
        except requests.RequestException:
            continue
    return 0

def check():
    df = pd.read_csv("fuken.csv").astype({"amedas_id": "Int64",
                                          "station_id": "Int64",
                                          "fuken_id": "Int64"})

    with ThreadPoolExecutor(max_workers=10) as executor:
        result = list(tqdm(executor.map(check_row, [row for _, row in df.iterrows()]),
                           total=len(df),
                           desc="Checking URLs"))

    df["access"] = result
    df.to_csv("fuken.csv", index=False)

if __name__ == "__main__":
    check()