import os
import logging
from typing import Dict, List, Optional, Sequence


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def get_all_datapath0(dir_name: str) -> List[str]:
    all_file_list = []

    for root, dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if file_name.endswith('.json'):  # 确保仅处理.json文件
                standard_path = os.path.join(root, file_name)
                all_file_list.append(standard_path)

    return all_file_list

if __name__ == '__main__':
    dir_name = '/blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/sft_data'
    print(get_all_datapath(dir_name), "\n")
    print(get_all_datapath0(dir_name), "\n")
    all_file_list = get_all_datapath0(dir_name)
    print(all_file_list[0].split(".")[-1], "\n")

    data_files = {'train': all_file_list}
    print(data_files, "\n")