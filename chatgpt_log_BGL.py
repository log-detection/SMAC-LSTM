import openai
import pandas as pd

# 设置API密钥
openai.api_key = ""


system_intel = """
You need to extract the Content of the logs and replace the dynamic variables with templates. Please output the results directly without explanation.
1. Replacement Rules:
   - Time Replacement: Replace all timestamps and specific times with "TIME".
   - IP Address Replacement: Replace IP addresses with "IP".
   - File Path Replacement: Replace file paths with "ADDR", ensuring to keep the colon after the path (e.g., "ADDR:").
   - Number Replacement: Replace all numbers in log entries with "NUM"
2. Example Log Replacement:
   - Original Log: 1134724900 2005.12.16 R43-M1-NC-I:J18-U01 2005-12-16-01.21.40.588712 R43-M1-NC-I:J18-U01 RAS APP FATAL ciod: Error reading message prefix on CioStream socket to 172.16.96.116:39416, Link has been severed
   - Replaced Log: ciod: Error reading message prefix on CioStream socket to "IP", Link has been severed
Please extract the log template from this log message:
"""


# 要读取的log文件路径
log_file_path = 'BGL2.log'
# 要保存的CSV文件路径
output_csv_path = 'processed_data_BGL.csv'

# 初始化一个空列表来保存处理后的数据
data = []

# 读取log文件并处理每一行
with open(log_file_path, 'r') as file:
    for line_number, line in enumerate(file, 1):
        # 使用OpenAI直接调用API
        try:
            result = openai.ChatCompletion.create(model="gpt-4",
                                                  messages=[{"role": "system", "content": system_intel},
                                                            {"role": "user", "content": line}])
            processed_text = result['choices'][0]['message']['content']
            print(f"Line {line_number}: {processed_text}")
            # 将原始和处理后的数据添加到列表
            data.append({'Original Text': line.strip(), 'Processed Text': processed_text})
        except Exception as e:
            print(f"An error occurred: {e}")

# 使用收集的数据创建DataFrame
df = pd.DataFrame(data)

# 将DataFrame保存到CSV文件
df.to_csv(output_csv_path, index=False)

print("处理完成，数据已保存到", output_csv_path)
