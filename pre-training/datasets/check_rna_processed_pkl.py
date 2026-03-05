import pickle
import numpy as np

data = pickle.load(open('./rna_processed.pkl', 'rb'))

print(data.keys())
# dict_keys(['case_ids', 'x_rna', 'wsi_paths', 'hvg_genes', 'n_genes'])

print("케이스(환자 수)"+str(len(data['case_ids']))  )   # 373
print("케이스 환자 ID 중 3개"+str(data['case_ids'][:3]))         # ['TCGA-99-8032', 'TCGA-86-8673', ...]

print("첫 환자의 처음 5개 유전자 발현값"+str(data['x_rna'][0][:5]))             # 첫 환자의 처음 5개 유전자 발현값

print("선택된 분산이 높은 유전자 유전자 ID 목록 중 5개"+str(data['hvg_genes'][:5]))    # 선택된 유전자 ID 목록
print("HVG 수"+str(data['n_genes'])       )       # 2000
print("RNA와 매핑된 WSI 경로"+str(data['wsi_paths'][0])   )      # regions.npy 경로