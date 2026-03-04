"""
TCGA-LUAD 멀티모달 데이터 다운로드 스크립트
- RNA-seq : GDC API (open-access, 토큰/회원가입 불필요)
- WSI     : 1순위 GDC open-access → 2순위 TCIA PathDB (회원가입 불필요)

의존성: pip install requests tqdm  (pandas/numpy 불필요)

사용법:
  python download_multimodal.py --wsi_txt wsi_files.txt --dry_run
  python download_multimodal.py --wsi_txt wsi_files.txt --out_dir ./downloads --rna_only
  python download_multimodal.py --wsi_txt wsi_files.txt --out_dir ./downloads
"""

import os
import re
import csv
import json
import time
import argparse
import requests
from tqdm import tqdm
from pathlib import Path

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"
# TCIA PathDB - 병리 이미지 전용 포털 (로그인 불필요, public)
PATHDB_BASE        = "https://pathdb.cancerimagingarchive.net/api"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1. wsi_files.txt 파싱
# ══════════════════════════════════════════════════════════════════════════════
def parse_wsi_files(path: str) -> list:
    records, seen = [], set()
    pattern = re.compile(
        r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})"
        r"-([0-9A-Z]{3}-[0-9A-Z]{2}-[A-Z0-9]{3})"
        r"\.([0-9a-fA-F-]{36})",
        re.IGNORECASE
    )
    with open(path) as f:
        raw = f.read()
    for m in pattern.finditer(raw):
        uuid = m.group(3).lower()
        if uuid in seen:
            continue
        seen.add(uuid)
        records.append({
            "case_id":   m.group(1).upper(),
            "sample_id": m.group(1).upper() + "-" + m.group(2).upper(),
            "img_uuid":  uuid,   # slide UUID (경로에 박혀있는 값)
        })
    print(f"[PARSE] {len(records)}개 환자 파싱 완료")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2. GDC: case_id → RNA-seq file_id 조회
# ══════════════════════════════════════════════════════════════════════════════
def fetch_rnaseq_uuids(case_ids: list, workflow: str = "STAR - Counts",
                       batch: int = 50) -> dict:
    result = {}
    for i in tqdm(range(0, len(case_ids), batch), desc="RNA-seq 조회", unit="batch"):
        b = case_ids[i:i+batch]
        filters = {"op":"and","content":[
            {"op":"in","content":{"field":"cases.submitter_id","value":b}},
            {"op":"=", "content":{"field":"data_type","value":"Gene Expression Quantification"}},
            {"op":"=", "content":{"field":"experimental_strategy","value":"RNA-Seq"}},
            {"op":"=", "content":{"field":"analysis.workflow_type","value":workflow}},
            {"op":"=", "content":{"field":"access","value":"open"}},
        ]}
        resp = requests.get(GDC_FILES_ENDPOINT, params={
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id",
            "size": str(batch*3), "format":"json"
        }, timeout=30)
        resp.raise_for_status()
        for hit in resp.json()["data"]["hits"]:
            for case in hit.get("cases", []):
                cid = case["submitter_id"].upper()
                if cid not in result:
                    result[cid] = {"file_id": hit["file_id"], "file_name": hit["file_name"]}
        time.sleep(0.2)
    print(f"[GDC]  RNA-seq: {len(result)}/{len(case_ids)} 매핑")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3. GDC: case_id → WSI file_id 조회  (Slide Image)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_wsi_from_gdc(case_ids: list, batch: int = 50) -> dict:
    """
    Returns: {case_id: {"file_id", "file_name", "access"}}
    data_type = "Slide Image", data_format = "SVS"
    access = open  → 토큰 없이 다운로드 가능
    access = controlled → TCIA PathDB로 우회
    """
    result = {}
    for i in tqdm(range(0, len(case_ids), batch), desc="GDC WSI 조회", unit="batch"):
        b = case_ids[i:i+batch]
        filters = {"op":"and","content":[
            {"op":"in","content":{"field":"cases.submitter_id","value":b}},
            {"op":"=", "content":{"field":"data_type","value":"Slide Image"}},
            {"op":"=", "content":{"field":"data_format","value":"SVS"}},
        ]}
        resp = requests.get(GDC_FILES_ENDPOINT, params={
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,access,cases.submitter_id",
            "size": str(batch*5), "format":"json"
        }, timeout=30)
        resp.raise_for_status()

        for hit in resp.json()["data"]["hits"]:
            for case in hit.get("cases", []):
                cid = case["submitter_id"].upper()
                if cid not in result:
                    result[cid] = {
                        "file_id":   hit["file_id"],
                        "file_name": hit["file_name"],
                        "access":    hit.get("access", "unknown"),
                    }
        time.sleep(0.2)

    open_cnt = sum(1 for v in result.values() if v["access"] == "open")
    ctrl_cnt = sum(1 for v in result.values() if v["access"] == "controlled")
    print(f"[GDC]  WSI: {len(result)}/{len(case_ids)} 조회  "
          f"(open={open_cnt}, controlled={ctrl_cnt})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4. TCIA PathDB: case_id → 이미지 URL 조회  (controlled 우회용)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_wsi_from_pathdb(case_ids: list, collection: str = "TCGA-LUAD") -> dict:
    """
    TCIA PathDB REST API로 병리 이미지 다운로드 URL 조회.
    회원가입/토큰 불필요 (public collection).

    Returns: {case_id: {"url": str, "file_name": str}}
    """
    result = {}

    # PathDB 슬라이드 목록 조회
    # 엔드포인트: GET /api/slides?project=TCGA-LUAD
    endpoints_to_try = [
        f"{PATHDB_BASE}/slides",
        f"{PATHDB_BASE}/v1/slides",
        "https://pathdb.cancerimagingarchive.net/slides",
    ]

    slides = []
    for url in endpoints_to_try:
        try:
            params = {"project": collection, "limit": 10000}
            resp   = requests.get(url, params=params,
                                  headers={"Accept":"application/json"},
                                  timeout=30)
            print(f"  PathDB {url} → status={resp.status_code} bytes={len(resp.content)}")
            if resp.status_code == 200 and resp.content.strip():
                data = resp.json()
                if isinstance(data, list):
                    slides = data
                elif isinstance(data, dict):
                    slides = data.get("data", data.get("slides", []))
                if slides:
                    print(f"  PathDB 슬라이드 {len(slides)}개 수신")
                    break
        except Exception as e:
            print(f"  PathDB {url} 오류: {e}")

    case_set = set(case_ids)
    for s in slides:
        # PatientID 또는 case_id 필드 탐색
        pid = (s.get("PatientID") or s.get("patient_id") or
               s.get("case_id") or "").upper()
        # TCGA-XX-XXXX 앞 3파트만 비교
        pid = "-".join(pid.split("-")[:3]) if pid else ""
        if pid in case_set and pid not in result:
            dl_url = (s.get("download_url") or s.get("url") or
                      s.get("file_url") or "")
            fname  = (s.get("file_name") or s.get("filename") or
                      s.get("name") or pid + ".svs")
            if dl_url:
                result[pid] = {"url": dl_url, "file_name": fname}

    print(f"[PathDB] WSI URL 매핑: {len(result)}/{len(case_ids)}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5. 매핑 테이블 빌드 & CSV 저장
# ══════════════════════════════════════════════════════════════════════════════
def build_and_save_mapping(records, rnaseq_map, gdc_wsi_map,
                           pathdb_map, out_path) -> list:
    rows = []
    for rec in records:
        cid = rec["case_id"]
        rna = rnaseq_map.get(cid)
        gdc = gdc_wsi_map.get(cid)
        pdb = pathdb_map.get(cid)

        # WSI 소스 결정: open GDC > controlled GDC(skip) > PathDB
        if gdc and gdc["access"] == "open":
            wsi_src      = "gdc_open"
            wsi_file_id  = gdc["file_id"]
            wsi_file_name= gdc["file_name"]
            wsi_dl_url   = ""
        elif pdb:
            wsi_src      = "pathdb"
            wsi_file_id  = ""
            wsi_file_name= pdb["file_name"]
            wsi_dl_url   = pdb["url"]
        elif gdc and gdc["access"] == "controlled":
            wsi_src      = "gdc_controlled(skip)"
            wsi_file_id  = gdc["file_id"]
            wsi_file_name= gdc["file_name"]
            wsi_dl_url   = ""
        else:
            wsi_src = wsi_file_id = wsi_file_name = wsi_dl_url = ""

        wsi_ok = wsi_src in ("gdc_open", "pathdb")
        row = {
            "case_id":        cid,
            "sample_id":      rec["sample_id"],
            "wsi_src":        wsi_src,
            "wsi_file_id":    wsi_file_id,
            "wsi_file_name":  wsi_file_name,
            "wsi_dl_url":     wsi_dl_url,
            "rna_file_id":    rna["file_id"]   if rna else "",
            "rna_file_name":  rna["file_name"] if rna else "",
            "paired":         str(bool(wsi_ok and rna)),
        }
        rows.append(row)

    # CSV 저장
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["case_id","sample_id","wsi_src","wsi_file_id","wsi_file_name",
              "wsi_dl_url","rna_file_id","rna_file_name","paired"]
    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(rows)

    paired = sum(1 for r in rows if r["paired"]=="True")
    src_cnt = {}
    for r in rows:
        src_cnt[r["wsi_src"]] = src_cnt.get(r["wsi_src"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  전체 환자   : {len(rows)}")
    print(f"  RNA-seq 준비: {sum(1 for r in rows if r['rna_file_id'])}")
    print(f"  WSI 소스 분류:")
    for src, cnt in src_cnt.items():
        print(f"    {src or '없음':<30}: {cnt}")
    print(f"  페어링 성공 : {paired}")
    print(f"{'='*60}")
    print(f"[INFO] 매핑 테이블: {out_path}\n")

    # 미리보기
    print(f"{'case_id':<20} {'wsi_src':<25} {'wsi_file_name':<35} paired")
    print("-"*90)
    for r in rows[:10]:
        print(f"{r['case_id']:<20} {r['wsi_src']:<25} "
              f"{r['wsi_file_name'][:33]:<35} {r['paired']}")
    if len(rows) > 10:
        print(f"  ... ({len(rows)-10}개 더)")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6. 다운로드
# ══════════════════════════════════════════════════════════════════════════════
def download_file(url: str, save_path: str, token: str = None) -> bool:
    if os.path.exists(save_path):
        return True
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hdrs = {}
    if token:
        hdrs["X-Auth-Token"] = token
    try:
        resp = requests.get(url, headers=hdrs, stream=True, timeout=300)
        if resp.status_code == 401:
            print(f"  [401] controlled-access: {os.path.basename(save_path)}")
            return False
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(save_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=os.path.basename(save_path)[:40], leave=False
        ) as bar:
            for chunk in resp.iter_content(1024*1024):
                f.write(chunk); bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        if os.path.exists(save_path): os.remove(save_path)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 파싱
    records  = parse_wsi_files(args.wsi_txt)
    case_ids = list(dict.fromkeys(r["case_id"] for r in records))

    # 2. RNA-seq (GDC open-access)
    rnaseq_map = fetch_rnaseq_uuids(case_ids, workflow=args.workflow)

    # 3. WSI from GDC (case_id 기반으로 file_id 조회)
    gdc_wsi_map = fetch_wsi_from_gdc(case_ids)

    # 4. controlled인 케이스 → TCIA PathDB로 우회
    controlled_ids = [cid for cid, v in gdc_wsi_map.items()
                      if v["access"] == "controlled"]
    no_gdc_ids     = [r["case_id"] for r in records
                      if r["case_id"] not in gdc_wsi_map]
    pathdb_targets = list(set(controlled_ids + no_gdc_ids))

    pathdb_map = {}
    if pathdb_targets:
        print(f"\n[INFO] {len(pathdb_targets)}개 케이스 → TCIA PathDB로 우회 시도")
        pathdb_map = fetch_wsi_from_pathdb(pathdb_targets)

    # 5. 매핑 테이블
    mapping_path = str(out_dir / "mapping.csv")
    rows = build_and_save_mapping(records, rnaseq_map,
                                  gdc_wsi_map, pathdb_map, mapping_path)

    if args.dry_run:
        print("[DRY RUN] 다운로드 없이 종료.")
        return

    # 6. RNA-seq 다운로드
    if not args.wsi_only:
        rna_rows = [r for r in rows if r["rna_file_id"]]
        print(f"\n=== RNA-seq 다운로드 ({len(rna_rows)}개) ===")
        ok = fail = 0
        for r in rna_rows:
            url  = f"{GDC_DATA_ENDPOINT}/{r['rna_file_id']}"
            path = str(out_dir / "rnaseq" / r["case_id"] / r["rna_file_name"])
            if download_file(url, path): ok += 1
            else: fail += 1
        print(f"  완료: {ok} | 실패: {fail}")

    # 7. WSI 다운로드
    if not args.rna_only:
        wsi_rows = [r for r in rows if r["paired"]=="True"]
        print(f"\n=== WSI 다운로드 ({len(wsi_rows)}개) ===")
        ok = fail = 0
        for r in wsi_rows:
            path = str(out_dir / "wsi" / r["case_id"] / r["wsi_file_name"])
            if r["wsi_src"] == "gdc_open":
                url = f"{GDC_DATA_ENDPOINT}/{r['wsi_file_id']}"
            else:  # pathdb
                url = r["wsi_dl_url"]
            if download_file(url, path): ok += 1
            else: fail += 1
        print(f"  완료: {ok} | 실패: {fail}")

    print(f"\n[DONE] 저장 위치: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_txt",  required=True)
    ap.add_argument("--out_dir",  default="./downloads")
    ap.add_argument("--workflow", default="STAR - Counts",
                    choices=["STAR - Counts","STAR - FPKM","STAR - FPKM-UQ"])
    ap.add_argument("--dry_run",  action="store_true")
    ap.add_argument("--rna_only", action="store_true")
    ap.add_argument("--wsi_only", action="store_true")
    args = ap.parse_args()
    main(args)