import jieba
import jieba.analyse
import scipy.integrate as integrate

import JTextRank
from tfidf import cal_tfidf
from utils import *


class Analyser(object):
    def __init__(self, dict_dir=None, stop_words_path=None):
        """
        :param dict_dir: 存放词典文件的目录，一是用来jieba分词之用，而是给热词结果添加标签
        :param stop_words_path: 停用词文件的路径(建议停用词文件使用 .dat 作为后缀,以和词典文件区分)
        """
        # 加载 dict_dir 目录下的字典
        if dict_dir is not None:
            filename_list = os.listdir(dict_dir)
            for filename in filename_list:
                if filename[-4:] == '.txt':
                    path = os.path.join(dict_dir, filename)
                    jieba.load_userdict(path)

        if stop_words_path is not None:
            jieba.analyse.set_stop_words(stop_words_path)  # 加载停用词文件
            self.stopwords_set = set([x.strip() for x in open(stop_words_path).readlines()])

    def _get_article_hot(self, readNums, likeNums, like_weight=0.8, normalize_rd_lk=False, follower=None):
        """
        定义热度，简单地通过下面的公式进行定义:
        热度 = 阅读数 * (1 - weight) + 点赞数 * weight

        :param readNums: 阅读数
        :param likeNums: 点赞数
        :param like_weight: 权重
        :param normalize_rd_lk: 是否对阅读数和点赞数进行归一化
        :return: 热度
        """
        if follower is not None:
            readNums = readNums / follower
            likeNums = likeNums / follower

        if normalize_rd_lk:
            read_mean = np.mean(readNums)
            read_std_var = np.std(readNums)
            like_mean = np.mean(likeNums)
            like_std_var = np.std(likeNums)

            norm_read_nums = np.zeros(readNums.shape)
            norm_like_nums = np.zeros(likeNums.shape)

            for i in range(len(readNums)):
                norm_read_nums[i] = integrate.quad(lambda x: 1 / (np.sqrt(2 * np.pi) * read_std_var) * np.e ** (
                        -((x - read_mean) ** 2) / (2 * (read_std_var ** 2))), -np.inf, readNums[i])[0]
                norm_like_nums[i] = integrate.quad(lambda x: 1 / (np.sqrt(2 * np.pi) * like_std_var) * np.e ** (
                        -((x - like_mean) ** 2) / (2 * (like_std_var ** 2))), -np.inf, likeNums[i])[0]

            article_hot = norm_read_nums * (1 - like_weight) + norm_like_nums * like_weight
        else:
            article_hot = readNums * (1 - like_weight) + likeNums * like_weight
        return article_hot

    def _get_word_hot(self, article_hot, weight):
        """
        获得单篇文章某个关键字的热度

        :param article_hot: 关键字所在文章的热度
        :param weight: 该关键字占该篇文章的权重
        :return: 关键字的热度
        """
        word_hot = article_hot * weight
        return word_hot

    def get_keywords(self, text_list, cut_method='tfidf', normalize_title_content=True):
        """
        给定文本，返回该文本的关键字以及对应权重

        :param text_list: 文本列表
        :param cut_method: 采用的分词方法, 可选 'tf-idf', 'JTextRank'
        :param normalize_title_content: 是否对关键字权重进行归一化
        :return: 关键字及其对应权重
        """
        keyword_weight_list = []

        if cut_method == 'tfidf':
            corpus = ['' for i in range(len(text_list))]
            for i, text in enumerate(text_list):
                text = remove_text(text, 'both')
                print("TFIDF: 分析第 " + str(i + 1) + '/' + str(len(text_list)) + " 段文本")
                seg_list = jieba.cut(text, cut_all=False)
                for word in seg_list:
                    if word not in self.stopwords_set and word.strip() != '':
                        corpus[i] += word + ' '
            keyword_weight_list = cal_tfidf(corpus)

        else:  # JTextRank
            for i, text in enumerate(text_list):
                text = remove_text(text, type='both')  # 移除数字
                print("JTextRank: 正在分析第 " + str(i + 1) + '/' + str(len(text_list)) + " 段文本")

                try:
                    tr = JTextRank.TextRank(text, stopwords=self.stopwords_set)
                    word_weight_dict = dict(tr.do())
                    word_weight_dict = dict(
                        map(lambda x: (x, word_weight_dict[x]), list(word_weight_dict.keys())))
                    keyword_weight_list.append(word_weight_dict)
                except AttributeError:
                    keyword_weight_list.append({})
                    pass

        if normalize_title_content == True:  # 归一化
            print("进行标题/全文关键字权重归一化...")
            for my_dict in keyword_weight_list:

                if cut_method == 'JTextRank':
                    my_dict = {k: v for k, v in my_dict.items() if v >= 1}  # JTextRank算法取归一化前分数>=1的词
                if cut_method == 'tfidf':
                    my_dict = {k: v for k, v in my_dict.items() if v >= 0.03}  # tfidf算法取归一化前分数>=0.03的词

                weight_list = list(my_dict.values())
                for key in my_dict:
                    my_dict[key] = np.e ** (my_dict[key]) / np.sum(np.e ** np.array(weight_list))  # softmax 归一化
        return keyword_weight_list

    def cal_keywords_weight(self, title_weight_list, content_weight_list, title_weight=0.8):
        """
        合并 title 中的关键字权重与 content 中的关键字权重. 仅在content中出现的关键字并不舍去.

        :param title_weight_list: title 的关键字与对应权重 (list[dict])
        :param content_weight_list: content 的关键字与对应权重 (list[dict])
        :param title_weight: 标题中关键字的权重 (float).
            计算方式为: title_weight * title_weight + content_weight * (1 - title_weight)
                如果 title_weight = 1, 则仅考虑 title 中的关键字; (title_weight*1 + content_weight*0)
                如果 title_weight = 0, 则仅考虑content中的关键字. (title_weight*0 + content_weight*1)
        :return: 合并后的关键字与对应权重 (list[dict])
        """
        assert len(title_weight_list) == len(content_weight_list)

        keyword_weight_list = []
        for i in range(len(title_weight_list)):
            current_weight = {}
            keys = list(
                set(title_weight_list[i].keys()) | set(content_weight_list[i].keys()))  # title & content 所有关键字的并集
            for key in keys:
                try:
                    current_weight[key] = title_weight_list[i][key] * title_weight + content_weight_list[i][key] * (
                            1 - title_weight)  # 关键字在 title 和 content 中均出现
                except:
                    if key not in title_weight_list[i].keys():
                        current_weight[key] = content_weight_list[i][key] * (1 - title_weight)  # 仅在 content 中出现的关键字
                    else:
                        current_weight[key] = title_weight_list[i][key] * title_weight  # 仅在 title 中出现的关键字
            keyword_weight_list.append(current_weight)

        for i in range(len(keyword_weight_list)):
            keyword_weight_list[i] = {k: v for k, v in keyword_weight_list[i].items() if
                                      v > 0}  # 去掉关键词权重为0的关键字

        return keyword_weight_list

    def get_ranked_words(self, data, hot_method='avg', like_weight=0.8, normalize_rd_lk=True, follower=True):
        """
        获取热词列表

        :param data: 输入的 excel 数据
        :param hot_method: 如果同一个热词出现在多篇文章中，对这个热词的热度采用的计算方法. 可选 'sum' 和 'avg'
        :param like_weight: 点赞权重，阅读权重为(1-like_weight)
        :param normalize_rd_lk: 是否归一化阅读点赞数
        :param follower: 是否有关注人数
        :return: 热词列表
        """
        ranked_word_list = {}

        # 读取数据
        readNums = data['readNum']  # 阅读人数
        likeNums = data['likeNum']  # 点赞人数
        keyword_weight = data['keyword_weight']  # keyword: weight, 类似 {'京东': 0.24, '裴健': 0.76}
        followerNums = data['follower'] if follower else None  # 相对关注人数

        print("正在计算文章热度...")
        popularity = self._get_article_hot(readNums, likeNums, like_weight=like_weight,
                                           normalize_rd_lk=normalize_rd_lk, follower=followerNums)  # 文章热度

        # 统计热词是否重复出现
        for i in range(len(keyword_weight)):
            print("统计第" + str(i) + '/' + str(len(keyword_weight)) + "篇文章的热词是否重复出现...")
            for key, weight in keyword_weight[i].items():
                if (weight >= 0) == False:  # 会出现关键字权重为nan情况
                    break
                if key in ranked_word_list.keys():  # 该热词已出现在其他文章中
                    if normalize_rd_lk:
                        ranked_word_list[key].append(
                            self._get_word_hot(np.e ** (10 * popularity[i]), weight))
                    else:
                        ranked_word_list[key].append(
                            self._get_word_hot(popularity[i], weight))

                else:  # 该热词未出现在其他文章中
                    if normalize_rd_lk:
                        ranked_word_list[key] = [
                            self._get_word_hot(np.e ** (10 * popularity[i]), weight)]
                    else:
                        ranked_word_list[key] = [
                            self._get_word_hot(popularity[i], weight)]

        # 提供两种方法计算同一个热词在多篇文章出现的情况
        if hot_method == 'sum':
            print("using 'sum'...")
            ranked_word_list = dict(
                map(lambda x: (x, np.sum(ranked_word_list[x])), ranked_word_list))  # 求和
        elif hot_method == 'avg':
            print("using 'avg'...")
            ranked_word_list = dict(
                map(lambda x: (x, np.mean(ranked_word_list[x])), ranked_word_list))  # 平均值
        else:  # 中位数
            print("using 'medium'...")
            ranked_word_list = dict(
                map(lambda x: (x, np.median(ranked_word_list[x])), ranked_word_list))

        ranked_word_list = sorted(ranked_word_list.items(), key=lambda x: x[1], reverse=True)  # 热词排序

        return ranked_word_list

    def main(self, data, param_list, comment='', topN=10000):
        """
        主程序

        :param data: 输入数据
        :param param_list: 参数列表(list[dict])
        :param comment: 注释，会添加到生成的文件名首部
        :return: None
        """

        print("共有" + str(len(param_list)) + '组参数')
        for i, params in enumerate(param_list):
            print("\t正在实验第" + str(i + 1) + '/' + str(len(param_list)) + '组参数')

            # 获取标题关键字与相应权重
            title_weight_list = self.get_keywords(data['title'],
                                                  cut_method=params['cut_method'],
                                                  normalize_title_content=params['normalize_title_content'])
            # 获取内容关键字与相应权重
            content_weight_list = self.get_keywords(data['content'],
                                                    cut_method=params['cut_method'],
                                                    normalize_title_content=params['normalize_title_content'])
            # 合并标题关键字与内容关键字的权重
            data['keyword_weight'] = self.cal_keywords_weight(title_weight_list,
                                                              content_weight_list,
                                                              title_weight=params['title_weight'])

            # data.to_csv('../output/keyword_weight.csv')   # 阈值 debug
            print("正在生成热词列表...")
            # 生成热词列表
            sorted_ranked_word_list = self.get_ranked_words(data,
                                                            hot_method=params['hot_method'],
                                                            like_weight=params['like_weight'],
                                                            normalize_rd_lk=params['normalize_rd_lk'],
                                                            follower=params['follower'])[:topN]

            title_list = ['' for i in range(len(sorted_ranked_word_list))]
            # 在结果中，将文章的标题一起输出
            for i in range(len(sorted_ranked_word_list)):
                print("寻找第" + str(i) + '/' + str(len(sorted_ranked_word_list)) + '个词的标题')
                for j in range(data.shape[0]):
                    if sorted_ranked_word_list[i][0] in data['keyword_weight'][j].keys():
                        if len(title_list[i]) < 30000:
                            title_list[i] += (data['title'][j] + '(' + str(data['readNum'][j]) + ',' + str(
                                data['likeNum'][j]) + ')\n')

            assert len(title_list) == len(sorted_ranked_word_list)

            # 保存
            write2excel([list(map(lambda x: x[0], sorted_ranked_word_list)),
                         list(map(lambda x: x[1], sorted_ranked_word_list)),
                         title_list],
                        output_dir + '/' + generate_filename(params, comment) + '.xls',
                        datatype=list)


if __name__ == '__main__':
    # 路径设置
    dict_dir = '../dict'  # 词典目录
    stop_words_path = '../dict/stop_words.dat'  # 停用词文件路径
    excel_path = '../input/Sample_data.xlsx'  # 输入 excel 路径
    selected_data_save_path = '../input/select_data2018.xls'  # 筛选后的输入数据路径文件名(具体到文件名哦), 可为 None
    output_dir = '../output'  # 保存热词结果 excel 的目录
    comment = "2018"  # 热词结果文件名的添加的前缀
    topN = 10000  # 保留前 n 个热词

    # 参数网
    param_grid = {
        'like_weight': [0.8],  # 在计算热度的时候，点赞量的权重；阅读量的权重为(1-like_weight)
        'title_weight': [1],  # 在计算文章关键字的权重时，标题关键字的权重
        'cut_method': ['tfidf'],  # 分词方法
        'hot_method': ['avg'],  # 同一热词出现在多篇文章的时候，采用 sum 或 avg
        'normalize_rd_lk': [True],  # 对阅读数，点赞数的归一化
        'normalize_title_content': [True],  # 对标题，全文关键字权重的归一化
        'follower': [False]  # 是否有follower
    }

    # 数据筛选
    select_param = {
        'date_range': ['2018-08-01', '2018-10-01'],  # 起始日期,结束日期
        'msgIdx': [1, 2, 3, 4, 5, 6, 7],  # 文章位置(例如2,3,4表示非头条文章)
        'sourceUrl': [True, False],  # 有无sourceUrl
        'readNum_range': [0, 4000000],  # 阅读数范围
        'likeNum_range': [0, 2000000],  # 点赞数范围
        'remove': ['公告 ', '活动 ', '报名 ', '招聘 ', '活动报名 ', '机器之心GMIS赠票']  # 如果关键字出现在标题中，则剔除
    }

    analyser = Analyser(dict_dir=dict_dir, stop_words_path=stop_words_path)

    # 筛选数据，读取的 excel 应当包含 ['title', 'content', 'readNum', 'likeNum', 'follower']
    data = select_data(excel_path, select_param, write_to=selected_data_save_path)

    # 根据 param_grid 生成所有可能的参数列表
    param_list = generate_param_list(param_grid)

    analyser.main(data, param_list, comment=comment, topN=topN)

    add_label(output_dir, dict_dir)  # 添加标签
