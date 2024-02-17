from .GAO_VMD_SE import GAO_VMD_SE

alpha = 289
K = 20

dataset = load_data(data_name)
omega_threshold, e_multi_score_threshold, u_multi_score_threshold = load_threshold()

result = {}
    
for key,value in dataset.items():
    result[key] = {}
    data_1 = value.iloc[:, 1]
    data_2 = value.iloc[:, 2]
    data_3 = value.iloc[:, 3]
    data_4 = value.iloc[:, 4]
    # print("key: ",key)

    result[key]['Hz'] = value.iloc[:, 0]/1000000000
    result[key]["1"] = GAO_VMD_SE(data_1, alpha, K, omega_threshold, e_multi_score_threshold)
    result[key]["2"] = GAO_VMD_SE(data_2, alpha, K, omega_threshold, e_multi_score_threshold)
    result[key]["3"] = GAO_VMD_SE(data_3, alpha, K, omega_threshold, u_multi_score_threshold)
    result[key]["4"] = GAO_VMD_SE(data_4, alpha, K, omega_threshold, u_multi_score_threshold)

print(result)
