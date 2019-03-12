import jieba
import numpy as np


class TextRank(object):

    def __init__(self, sentence, window=3, alpha=0.85, iternum=700, stopwords=[]):
        self.sentence = sentence
        self.window = window
        self.alpha = alpha
        self.stopwords = stopwords
        self.edge_dict = {}  # 记录节点的边连接字典
        self.iternum = iternum  # 迭代次数
        self.result = None

    # 对句子进行分词
    def cutSentence(self):
        seg_result = cut_stopwords(self.sentence, self.stopwords)
        self.word_list = [word for word in seg_result]

    # 根据窗口，构建每个节点的相邻节点,返回边的集合
    def createNodes(self):
        tmp_list = []
        word_list_len = len(self.word_list)
        for index, word in enumerate(self.word_list):
            if word not in self.edge_dict.keys():
                tmp_list.append(word)
                tmp_set = set()
                left = index - self.window + 1  # 窗口左边界
                right = index + self.window  # 窗口右边界
                if left < 0: left = 0
                if right >= word_list_len: right = word_list_len
                for i in range(left, right):
                    if i == index:
                        continue
                    tmp_set.add(self.word_list[i])
                self.edge_dict[word] = tmp_set

    # 根据边的相连关系，构建矩阵
    def createMatrix(self):
        self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
        self.word_index = {}  # 记录词的index
        self.index_dict = {}  # 记录节点index对应的词

        for i, v in enumerate(set(self.word_list)):
            self.word_index[v] = i
            self.index_dict[i] = v
        for key in self.edge_dict.keys():
            for w in self.edge_dict[key]:
                self.matrix[self.word_index[key]][self.word_index[w]] = 1
                self.matrix[self.word_index[w]][self.word_index[key]] = 1
        # 归一化
        for j in range(self.matrix.shape[1]):
            sum = 0
            for i in range(self.matrix.shape[0]):
                sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                self.matrix[i][j] /= sum

    # 根据textrank公式计算权重
    def calPR(self):
        self.PR = np.ones([len(set(self.word_list)), 1])
        for i in range(self.iternum):
            self.PR = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.PR)

    def printResult(self):
        word_pr = {}
        for i in range(len(self.PR)):
            word_pr[self.index_dict[i]] = self.PR[i][0]
        self.result = sorted(word_pr.items(), key=lambda x: x[1], reverse=True)

    def do(self):
        self.cutSentence()
        self.createNodes()
        self.createMatrix()
        self.calPR()
        self.printResult()
        return self.result


def cut_stopwords(text, stop_words):
    result = list()
    seg_list = jieba.cut(text, cut_all=False)
    for word in seg_list:
        if word not in stop_words:
            if word != '\t':
                if len(word) != 1:
                    if not word.isdigit():
                        word = word.strip()
                        result.append((word))
    while '' in result:
        result.remove('')
    return result
