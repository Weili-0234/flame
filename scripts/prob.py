def calculate_uniform_probabilities(row_numbers):  
    """  
    根据每个数据集的行数计算均匀采样概率  
      
    Args:  
        row_numbers: List[int] - 每个数据集的行数  
      
    Returns:  
        List[float] - 归一化的概率列表，总和为1.0  
    """  
    total_rows = sum(row_numbers)  
    probabilities = [rows / total_rows for rows in row_numbers]  
      
    # 确保概率总和精确为1.0（处理浮点数精度问题）  
    prob_sum = sum(probabilities)  
    if abs(prob_sum - 1.0) > 1e-10:  
        # 调整最后一个概率以确保总和为1  
        probabilities[-1] += (1.0 - prob_sum)  
      
    return probabilities  
  
# 根据你的数据集行数  
row_numbers = [930514, 578043, 814015, 1558305, 1940458, 3562513]  
  
# 计算概率  
probabilities = calculate_uniform_probabilities(row_numbers)  
  
# 格式化为逗号分隔的字符串，用于命令行参数  
prob_string = ','.join([f"{p:.6f}" for p in probabilities])  
  
print("数据集行数:", row_numbers)  
print("计算出的概率:", probabilities)  
print("概率总和:", sum(probabilities))  
print("用于--data_probs的字符串:")  
print(prob_string)