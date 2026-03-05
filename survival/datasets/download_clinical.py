"""
TCGA-LUAD 임상 데이터 (OS, OS.time, censorship) 다운로드
GDC API - 로그인/토큰 불필요 (open-access)

사용법:
  python download_clinical.py \
    --mapping ./downloads/mapping.csv \
    --out     ./downloads/clinical.csv
"""

import csv
import json
import time
import argparse
import requests

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


def fetch_clinical(case_ids: list, batch_size: int = 50) -> dict:
    """
    Returns:
      {case_id: {"os_days": float, "censored": int}}
      censored: 1 = 사망(event 발생), 0 = 중도절단(생존/추적불가)
    """
    result = {}

    for i in range(0, len(case_ids), batch_size):
        batch = case_ids[i : i + batch_size]

        filters = {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": batch
            }
        }
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join([
                "submitter_id",
                "demographic.vital_status",
                "demographic.days_to_death",
                "diagnoses.days_to_last_follow_up",
                "diagnoses.vital_status",
            ]),
            "size": str(batch_size),
            "format": "json",
        }

        resp = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()

        for hit in resp.json()["data"]["hits"]:
            cid = hit["submitter_id"].upper()

            # vital_status & survival time 추출
            demo      = hit.get("demographic", {})
            diagnoses = hit.get("diagnoses", [{}])
            diag      = diagnoses[0] if diagnoses else {}

            vital = (demo.get("vital_status") or
                     diag.get("vital_status") or "").lower()

            # days_to_death: 사망한 경우
            # days_to_last_follow_up: 생존/중도절단인 경우
            days_to_death   = demo.get("days_to_death")
            days_to_followup = diag.get("days_to_last_follow_up")

            if vital == "dead" and days_to_death is not None:
                os_days  = float(days_to_death)
                censored = 1   # event 발생 (사망)
            elif days_to_followup is not None:
                os_days  = float(days_to_followup)
                censored = 0   # 중도절단 (생존)
            else:
                # 둘 다 없으면 스킵
                print(f"  [WARN] {cid}: 생존 정보 없음 (vital={vital})")
                continue

            if os_days <= 0:
                print(f"  [WARN] {cid}: os_days={os_days} ≤ 0, 스킵")
                continue

            result[cid] = {
                "os_days":  os_days,
                "censored": censored,
                "vital_status": vital,
            }

        time.sleep(0.2)

    return result


def main(args):
    # mapping.csv에서 case_id 목록 읽기
    case_ids = []
    with open(args.mapping) as f:
        for row in csv.DictReader(f):
            if row.get("paired", "").strip() == "True":
                case_ids.append(row["case_id"].upper())

    case_ids = list(dict.fromkeys(case_ids))  # 중복 제거
    print(f"[INFO] 임상 데이터 조회: {len(case_ids)}개 케이스")

    clinical = fetch_clinical(case_ids)

    # 결과 통계
    dead      = sum(1 for v in clinical.values() if v["censored"] == 1)
    censored  = sum(1 for v in clinical.values() if v["censored"] == 0)
    print(f"\n[결과] 총 {len(clinical)}개")
    print(f"  사망(event=1)   : {dead}")
    print(f"  중도절단(event=0): {censored}")
    print(f"  미수집          : {len(case_ids) - len(clinical)}")

    # CSV 저장
    fields = ["case_id", "os_days", "censored", "vital_status"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cid, v in clinical.items():
            writer.writerow({"case_id": cid, **v})

    print(f"\n[DONE] 저장: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="./downloads/mapping.csv")
    ap.add_argument("--out",     default="./downloads/clinical.csv")
    args = ap.parse_args()
    main(args)