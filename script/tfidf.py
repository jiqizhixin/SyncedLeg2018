import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def cal_tfidf(corpus):
    """
    给定语料库，返回每一篇文档的关键字权重
    :param corpus: 关键字之间使用空格分隔
        例如: corpus = ["我 来到 北京 清华大学",
                "他 来到 了 网易 杭研 大厦",
                "小明 硕士 毕业 与 中国 科学院",
                "我 爱 北京 天安门"]
    :return: 每篇文章的关键字与其对应的权重 (list[dict])
    """
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = np.array(vectorizer.get_feature_names())
    weight = tfidf.toarray()

    keyword_weight_list = []
    for i in range(weight.shape[0]):
        print("正在抽取第 " + str(i + 1) + " 个词的权重")
        idx = np.where(weight[i, :] != 0)

        my_dict = dict(zip(word[idx], weight[i, :][idx]))  # high efficiency
        keyword_weight_list.append(my_dict)
    return keyword_weight_list


if __name__ == '__main__':
    data = pd.read_excel('../input/posts.xls', header=0).iloc[:100, :]

    corpus = ['' for i in range(data.shape[0])]
    for i in range(data.shape[0]):
        seg_list = jieba.cut(data['title'][i], cut_all=False)
        for word in seg_list:
            corpus[i] += word + ' '
    l = cal_tfidf(corpus)
