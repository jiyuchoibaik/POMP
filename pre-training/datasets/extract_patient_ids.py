import re
import json

input_txt = "/Users/choijiyubaik/Documents/DAC/26춘계_의정학/POMP/pre-training/datasets/wsi_files.txt"
output_json = "patient_ids.json"

patient_ids = []

with open(input_txt, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    # 환자 ID 패턴: TCGA-XX-YYYY
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", line)
    if m:
        patient_ids.append(m.group(1))

# 중복 제거 + 정렬
patient_ids = sorted(list(set(patient_ids)))

print(f"총 환자 수: {len(patient_ids)}")
print("예시:", patient_ids[:10])

# JSON 저장
with open(output_json, "w") as f:
    json.dump(patient_ids, f, indent=2)

print(f"\npatient_ids.json 파일 생성 완료!")