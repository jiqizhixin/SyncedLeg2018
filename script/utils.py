import os
import re

import numpy as np
import pandas as pd
import xlwt


def generate_param_list(param_grid):
    """
    给定一个参数网，返回所有可能参数组合的一个 list
    """
    param_index = np.zeros(len(param_grid), dtype=int)  # 参数索引
    carry_list = [len(param_grid[key]) for key in param_grid]  # 每个参数可选值数目

    param_list = []  # 生成的参数列表

    # 模拟进位方法，产生所有可能参数的列表
    while True:
        params = {}

        carry = (param_index[0] + 1) // carry_list[0]
        param_index[0] = (param_index[0] + 1) % carry_list[0]

        for i in range(1, len(param_index)):
            temp = (param_index[i] + carry) % carry_list[i]
            carry = (param_index[i] + carry) // carry_list[i]
            param_index[i] = temp

        for i, key in enumerate(param_grid):  # 将参数 index 转换成具体数值
            params[key] = param_grid[key][param_index[i]]

        param_list.append(params)

        if np.sum(param_index) == 0:
            break

    return param_list


def generate_filename(params, comment):
    filename = comment + '_' + params['hot_method'] + '_' \
               + params['cut_method'] + '_' \
               + 'lw' + str(params['like_weight']) + '_' \
               + 'tw' + str(params['title_weight']) + '_'

    if params['normalize_rd_lk']:
        filename += 'norm-rl_'
    else:
        filename += 'nonorm-rl_'

    if params['follower']:
        filename += 'foll_'
    else:
        filename += 'nofoll_'

    if params['normalize_title_content']:
        filename += 'norm-tc'
    else:
        filename += 'nonorm-tc'

    return filename


def batch_get_excel_rows(dir, topN):
    """
    批量对某一目录下的所有 excel 文件进行抽取前N行(注意保存方式为覆盖原文件)
    :param dir: 目录
    :param topN: 前 N 行
    :return: None
    """

    def get_excel_rows(read_from, write_to, topN=100):
        """
        获取一个 excel 文件的前n行
        :param read_from: 读取的 excel
        :param write_to: 写入的 excel
        :param topN: 前n行(int)
        :return: None
        """
        df = pd.read_excel(read_from, header=None, nrows=topN)
        df.to_excel(write_to, header=None, index=None)

    filename_list = os.listdir(dir)
    for filename in filename_list:
        if filename[-4:] == '.xls' or filename[-5:] == '.xlsx':
            path = os.path.join(dir, filename)
            get_excel_rows(read_from=path, write_to=path, topN=topN)


def select_data(excel_file, select_param, write_to=None):
    """
    从所有数据中抽取符合条件的数据

    :param excel_file: excel文件
    :param select_param: 格式如下
        select_param = {
            'date_range': ['2017-09-11', '2018-09-11'],  # 起始日期和结束日期
            'readNum_range': [0, 9999999],   # 阅读人数范围
            'likeNum_range': [0, 9999999],   # 点赞人数范围
            'msgIdx': [1, 2, 3, 4],  # 是否头条文章
            'sourceUrl': [True, False],  # 有无sourceURL
            'remove': ['公告', '活动', '报名']    # 如果关键字出现在标题中，则剔除
        }
    :param write_to: 写入的文件名，如果为None则不写入
    :return: 选择后的数据(DataFrame)
    """
    df = pd.read_excel(excel_file, header=0, encoding='utf-8')
    # content
    df = df.loc[pd.isnull(df['content']) == False]  # 移除缺失content的数据

    # title & content 大写转小写
    df['title'] = df['title'].str.lower()
    df['content'] = df['content'].str.lower()

    # follower
    if 'follower' in df.columns:
        df = df.loc[pd.isnull(df['follower']) == False]

    # readNums
    if 'readNum_range' in select_param.keys():
        assert len(select_param['readNum_range']) == 2
        mask = (df['readNum'] >= select_param['readNum_range'][0]) & (
                df['readNum'] <= select_param['readNum_range'][1])
        df = df.loc[mask]

    # likeNums
    if 'likeNum_range' in select_param.keys():
        assert len(select_param['likeNum_range']) == 2 or 'likeNum_range' not in select_param.keys()
        mask = (df['likeNum'] >= select_param['likeNum_range'][0]) & (
                df['likeNum'] <= select_param['likeNum_range'][1])
        df = df.loc[mask]

    # 日期选择
    if 'date_range' in select_param.keys():
        df['publishAt'] = pd.to_datetime(df['publishAt'])
        mask = (df['publishAt'] > select_param['date_range'][0]) & (df['publishAt'] < select_param['date_range'][1])
        df = df.loc[mask]

    # msgIdx
    if 'msgIdx' in select_param.keys():
        mask = np.zeros(df['msgIdx'].shape[0])
        for i in range(mask.shape[0]):
            if df['msgIdx'].iloc[i] in select_param['msgIdx']:
                mask[i] = True
            else:
                mask[i] = False
        df = df.loc[mask == 1]

    # sourceURL
    if 'sourceUrl' in select_param.keys():
        if len(select_param['sourceUrl']) == 1 and select_param['sourceUrl'][0] == True:
            df = df.loc[pd.isnull(df['sourceUrl']) == False]
        elif len(select_param['sourceUrl']) == 1 and select_param['sourceUrl'][0] == False:
            df = df.loc[pd.isnull(df['sourceUrl'])]
        else:
            pass

    # remove
    if 'remove' in select_param.keys():
        mask = np.zeros(df['title'].shape[0])
        for i in range(mask.shape[0]):
            if any(key in df['title'].iloc[i] for key in select_param['remove']):
                mask[i] = False
            else:
                mask[i] = True
        df = df.loc[mask == 1]

    df = df.reset_index(drop=True)
    # 写入文件
    if write_to is not None:
        df.to_excel(write_to, index=None)
    return df


def write2excel(data, write_to, datatype=list):
    """
    将字典/列表 data 中的数据写入到 excel 文件 (specified by write_to) 中.
    使用字典的好处在于方便去重～

    :param data: 类型为 dict/list.
        如果类型为字典，例如 data = {'a':['b', 'c', 'd'],
                         '0':['1', '2', '3']}
        则在 excel 中为:
            a   b   c   d
            0   1   2   3

        如果类型为列表, 例如 data = [['a', 'b', 'c', 'd'],
                         [ 0,   1,   2,  3]]
        则在 excel 中为:
            a 0
            b 1
            c 2
            d 3

    :param write_to: 写入的文件名
    :datatype: 传入的 data 类型, list or dict
    :return: None
    """
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(
        'sheet1', cell_overwrite_ok=True)  # 新建一个 sheet

    if datatype == dict and type(data) == dict:
        curr_row = 0
        for key in data.keys():
            worksheet.write(curr_row, 0, key)  # 第 0 列写入key
            for col in range(len(data[key])):  # 写入其他列
                worksheet.write(curr_row, col + 1, data[key][col])
            curr_row += 1
    elif datatype == list and type(data) == list:
        # 例如 data = [['A', 'B', 'C'], ['1', '2', '3'], ['AI', 'ML', 'DL']]
        # 则 ['A', 'B', 'C'] 写入 excel 的第一列, 以此类推
        for col in range(len(data)):
            for row in range(len(data[col])):
                worksheet.write(row, col, data[col][row])
    else:
        print("type must be either dict or list.")
        return

    print("文件已保存为 " + write_to)
    workbook.save(write_to)


def remove_text(text, type='number'):
    """
    从文本中移除特定文本，例如数字或标点

    :param text: 文本
    :param type: 移除的文本类型, 可选'number', 'punc', 'both'
    :return: 移除后的文本
    """
    from zhon.hanzi import punctuation
    import string
    if type == 'number':
        text = re.sub('[0-9]+', '', text)
    elif type == 'punc':
        text = re.sub("[{}]+".format(punctuation), " ", text)
    elif type == 'both':
        text = re.sub("[0-9]+|[{}]+|[{}]+".format(punctuation, string.punctuation), " ", text)

    return text


def add_label(excel_dir, dict_dir):
    """
    给 excel_dir 目录下的所有 excel 文件添加 dict_dir 目录下的字典标签，标签名为字典文件名前缀。

    :param excel_dir: 需要添加标签的 excel 所在目录
    :param dict_dir: 字典目录
    :return: None
    """

    excel_filename_list = os.listdir(excel_dir)
    # read them in
    excel_filename_list = [filename for filename in excel_filename_list if
                           len(filename) > 4 and filename[-4:] in ['.xls', 'xlsx']]

    dict_filename_list = os.listdir(dict_dir)
    dict_filename_list = [filename for filename in dict_filename_list if
                          filename[-4:] == '.txt']

    my_dict = {}  # 构造字典

    for filename in sorted(dict_filename_list):
        lines = [x.strip().replace('@@n', '').lower() for x in open(os.path.join(dict_dir, filename)).readlines()]
        for x in lines:
            prefix, _ = os.path.splitext(filename)
            my_dict[x.strip().replace('@@n', '').lower()] = prefix

    for i in range(len(excel_filename_list)):
        df = pd.read_excel(os.path.join(excel_dir, excel_filename_list[i]), header=None)
        error = -1
        df['label'] = ''
        index = df.shape[1]
        for j in range(df.shape[0]):
            if str(df.iloc[j, 0]) == 'nan':
                error = j
            try:
                df.iloc[j, index - 1] = my_dict[df.iloc[j, 0]]
            except:
                df.iloc[j, index - 1] = '其他'
        if error != -1:
            df = df.drop(error)
        df.to_excel(os.path.join(excel_dir, excel_filename_list[i]), header=None, index=None)
