"""
다운로드 전 파일 크기 미리 조회
usage: python check_size.py --mapping downloads/mapping.csv
"""
import csv
import json
import time
import argparse
import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

def check_sizes(mapping_csv: str):
    # mapping.csv에서 file_id 목록 읽기
    rows = []
    with open(mapping_csv) as f:
        rows = list(csv.DictReader(f))

    rna_ids = [r["rna_file_id"]  for r in rows if r["rna_file_id"]]
    wsi_ids = [r["wsi_file_id"]  for r in rows if r["wsi_file_id"]]

    def query_sizes(file_ids, label, batch=100):
        total_bytes = 0
        missing     = 0
        for i in range(0, len(file_ids), batch):
            b = file_ids[i:i+batch]
            filters = {"op":"in","content":{"field":"file_id","value":b}}
            resp = requests.get(GDC_FILES_ENDPOINT, params={
                "filters": json.dumps(filters),
                "fields": "file_id,file_size",
                "size": str(batch), "format": "json"
            }, timeout=30)
            resp.raise_for_status()
            hits = resp.json()["data"]["hits"]
            for hit in hits:
                total_bytes += hit.get("file_size", 0)
            missing += len(b) - len(hits)
            time.sleep(0.2)

        gb  = total_bytes / 1024**3
        avg = (total_bytes / len(file_ids)) / 1024**2 if file_ids else 0
        print(f"\n[{label}]")
        print(f"  파일 수     : {len(file_ids)}개")
        print(f"  총 용량     : {gb:.1f} GB  ({total_bytes/1024**2:.0f} MB)")
        print(f"  평균 용량   : {avg:.1f} MB / 파일")
        if missing:
            print(f"  크기 미조회 : {missing}개")
        return total_bytes

    rna_bytes = query_sizes(rna_ids, "RNA-seq")
    wsi_bytes = query_sizes(wsi_ids, "WSI (.svs)")
    total     = (rna_bytes + wsi_bytes) / 1024**3

    print(f"\n{'='*45}")
    print(f"  전체 합계 : {total:.1f} GB")
    print(f"{'='*45}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="downloads/mapping.csv")
    args = ap.parse_args()
    check_sizes(args.mapping)