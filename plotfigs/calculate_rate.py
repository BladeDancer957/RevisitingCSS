

# 4719/13962
# 1889/4444
# 885/2878
# 373/874
# 6546/18500
# 10353/27113
# 12385/30592
# 15299/39920
# 4660/13743
# 1707/4621
# 920/3041
# 380/870
# 7705/18894
# 12120/27369
# 14091/30914
# 18391/40298

# 257*5.5/58035521
# 257*3/58035521
# 1285/58036549
# 257/58035521
# 12850/58074071
# 2570*3/58068931
# 1285*5.5/58068289
# 12850*2/58067646
# 257*5.5/94813561
# 257*3/94813561
# 1285/94814589
# 257/94813561
# 12850/94852111
# 2570*3/94846971
# 1285*5.5/94846329
# 12850*1.5/94845686

VOC_datas = {
    "Res101": {
        "VOC 10-1 (11 steps)": {"sota": [13962, 58035521], "ours": [4719, 257*5.5]},
        "VOC 15-1 (6 steps)": {"sota": [4444, 58035521], "ours": [1889, 257*3]},
        "VOC 15-5 (2 steps)": {"sota": [2878, 58036549], "ours": [885, 1285]},
        "VOC 19-1 (2 steps)": {"sota": [874, 58035521], "ours": [373, 257]},
        "ADE 100-50 (2 steps)": {"sota": [18500, 58074071], "ours": [6546, 12850]},
        "ADE 100-10 (6 steps)": {"sota": [27113, 58068931], "ours": [10353, 2570*3]},
        "ADE 100-5 (11 steps)": {"sota": [30592, 58068289], "ours": [12385, 1285*5.5]},
        "ADE 50-50 (3 steps)": {"sota": [39920, 58067646], "ours": [15299, 12850*1.5]},
    },
    "Swin-B": {
        "VOC 10-1 (11 steps)": {"sota": [13743, 94813561], "ours": [4660, 257*5.5]},
        "VOC 15-1 (6 steps)": {"sota": [4621, 94813561], "ours": [1707, 257*3]},
        "VOC 15-5 (2 steps)": {"sota": [3041, 94814589], "ours": [920, 1285]},
        "VOC 19-1 (2 steps)": {"sota": [870, 94813561], "ours": [380, 257]},
        "ADE 100-50 (2 steps)": {"sota": [18894, 94852111], "ours": [7705, 12850]},
        "ADE 100-10 (6 steps)": {"sota": [27369, 94846971], "ours": [12120, 2570*3]},
        "ADE 100-5 (11 steps)": {"sota": [30914, 94846329], "ours": [14091, 1285*5.5]},
        "ADE 50-50 (3 steps)": {"sota": [40298, 94845686], "ours": [18391, 12850*1.5]},
    },
}

def calculate_rate(sota_time, sota_params, ours_time, ours_params):
    """
    Calculate the rate of our method compared to the SOTA method.
    
    :param sota_time: Time taken by the SOTA method (in minutes).
    :param sota_params: Parameters used by the SOTA method (in millions).
    :param ours_time: Time taken by our method (in minutes).
    :param ours_params: Parameters used by our method (in millions).
    :return: A tuple containing the time rate and parameters rate.
    """
    time_rate = ours_time / sota_time
    params_rate = ours_params / sota_params
    return time_rate, params_rate

if __name__=='__main__':
    time_rate_list = []
    params_rate_list = []
    for model, data in VOC_datas.items():
        print(f"Model: {model}")
        for key, values in data.items():
            sota_time, sota_params = values["sota"]
            ours_time, ours_params = values["ours"]
            time_rate, params_rate = calculate_rate(sota_time, sota_params, ours_time, ours_params)
            print(f"{key}: Time Rate = {time_rate:.4f}, Params Rate = {params_rate:.8f}")
            time_rate_list.append(time_rate)
            params_rate_list.append(params_rate)
    
    # min and max rates
    min_time_rate = min(time_rate_list)
    max_time_rate = max(time_rate_list)
    min_params_rate = min(params_rate_list)
    max_params_rate = max(params_rate_list)
    print(f"\nOverall Min/Max Time Rate: {min_time_rate:.4f}/{max_time_rate:.4f}")
    print(f"Overall Min/Max Params Rate: {min_params_rate:.8f}/{max_params_rate:.8f}")
    # average rates
    # avg_time_rate = sum(time_rate_list) / len(time_rate_list)
    # avg_params_rate = sum(params_rate_list) / len(params_rate_list)
    # print(f"Average Time Rate: {avg_time_rate:.4f}")
    # print(f"Average Params Rate: {avg_params_rate:.4f}")