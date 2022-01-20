import numpy as np
from scipy import sparse
import re
import nltk
import os

"""
Necessary retriever classes and other classes
"""


class DocumentProcessed:
    """
    文档预处理类
    """

    def __init__(self, in_path):
        """
        :param in_path: 依赖数据文件目录
        """
        self.data_path = in_path
        self.files_path = []
        self.doc_dict = {}
        self.token_dict = {}
        self.num_docs = 0  # 文档总数
        self.documents = []
        self.tf = sparse.csc_matrix([])  # TF矩阵（CSC Sparse Matrix)
        self.df = {}  # DF向量（字典）
        self.inverted_index = {}  # e.g. 'yes':[1,2]
        self.tfidf = sparse.csc_matrix([])  # TF-iDF矩阵（CSC Sparse Matrix)

        self.process(self.data_path)  # 处理文档，构建除了倒排表、TF-iDF之外的其他属性
        self.inverted_index_create()
        self.tfidf_create()

    def process(self, in_path):
        """
        :param in_path: 依赖数据文件目录
        """
        print('文档预处理中。。。。')
        for fpath, dirs, fs in os.walk(in_path):
            for fname in fs:
                self.files_path.append(os.path.abspath(os.path.join(fpath, fname)))
        # ####################
        # 暂时取前100 000文档
        self.files_path = self.files_path[:50000]
        # ####################
        self.num_docs = len(self.files_path)
        self.doc_dict = {self.files_path[i]: i for i in range(len(self.files_path))}
        token_all = []
        for fname in self.files_path:  # 遍历每个文档
            if self.doc_dict[fname] % 10000 == 0:
                print('正在扫描处理第{}个文档。。。。'.format(self.doc_dict[fname]))
            with open(fname, 'r') as f:
                try:
                    txt = f.read().lower()  # 全都小写化读取
                except UnicodeDecodeError:
                    txt = ''  # 解码问题是由于文件存在乱码造成的，大约20多个；跳过

                # 去除邮件头部
                # message-id: 从此
                # subject: 仅保留该行
                # x-filename: 至此
                txt_lines = txt.split('\n')
                txt_new = []
                flag_reserve = False
                for line in txt_lines:
                    if flag_reserve:
                        txt_new.append(line)
                    elif 'subject:' in line:
                        txt_new.append(line[len('subject:'):])
                    elif 'x-filename:' in line:
                        flag_reserve = True
                txt_new = '\n'.join(txt_new)

                words = nltk.word_tokenize(txt_new)  # 分词
                words = [w for w in words if w.isalpha()]  # 去除标点符号
                en_stop_words = nltk.corpus.stopwords.words('english')
                words = [w for w in words if w not in en_stop_words]  # 去除停用词
                snowball = nltk.SnowballStemmer('english')
                words = [snowball.stem(w) for w in words]  # 主干提取
                wnl = nltk.WordNetLemmatizer()
                words = [wnl.lemmatize(w) for w in words]  # 词性还原
                self.documents.append(words)  # 后续有待归并
                token_all += list(set(words))  # 词条归并, 为构建DF向量（字典）

        self.df = dict(sorted(nltk.probability.FreqDist(token_all).most_common(1000)))  # 只使用文档频率前1000的token，并按照ascii码排序
        token_unit = list(self.df.keys())
        token_unit.sort()
        self.token_dict = {token_unit[i]: i for i in range(len(token_unit))}  # 文档频率前1000的token按照ascii码排序

        ###########
        # 构建TF稀疏矩阵(伪DataFrame)
        print('正在构建TF稀疏矩阵。。。。')
        self.tf = sparse.csc_matrix(np.zeros([len(self.df), len(self.files_path)]))
        for fname in self.files_path:
            if self.doc_dict[fname] % 10000 == 0:
                print('正根据第{}个文档构造TF矩阵。。。。'.format(self.doc_dict[fname]))
            words = self.documents[self.doc_dict[fname]].copy()  # 每个doc中的words里包含的token还需要取其前1000
            words = list(set(words))  # 归并
            words = [w for w in words if w in token_unit]  # 属于前1000的token
            self.documents[self.doc_dict[fname]] = words
            self.tf[[self.token_dict[w] for w in words], np.repeat(self.doc_dict[fname], len(words))] += np.repeat(
                1, len(words))

    def inverted_index_create(self):
        """
        三步走算法构造倒排表
        """
        print('倒排表构造中。。。。。')
        docID_tmp = np.array([], dtype=np.int)  # dtype: int
        token_tmp = np.array([])  # dtype: object

        # 检索每篇文档，获得 < 词项，文档ID > 对，并写入临时索引 - 1
        for i in range(self.num_docs):
            words = self.documents[i]
            docID_tmp = np.append(docID_tmp, [i] * len(words)).astype(np.int)
            token_tmp = np.append(token_tmp, words)

        # 对临时索引中的词项进行排序 - 2
        idx_sort = token_tmp.argsort()
        token_tmp.sort()
        docID_tmp = docID_tmp[idx_sort]

        # 遍历临时索引，对于相同词项的文档ID进行合并 - 3
        self.inverted_index = {w: [] for w in self.token_dict.keys()}
        for i in range(len(token_tmp)):  # 遍历临时索引
            self.inverted_index[token_tmp[i]].append(docID_tmp[i])

    def tfidf_create(self):
        print('TF-iDF矩阵构造中。。。。。')
        try:
            self.tfidf = self.tf.copy()  # 与TF矩阵（DataFrame）形状相同，索引名称相同（token-docName)
            tf_arr = self.tf.toarray()  # 2-D ndarray
            df_arr = np.array(list(self.df.values()), dtype=np.float).reshape(-1, 1)  # 1/2-D ndarray
            tf_arr[tf_arr > 0] = 1 + np.log10(tf_arr[tf_arr > 0])
            tf_arr[tf_arr == 0] = 0
            self.tfidf = sparse.csc_matrix(tf_arr * np.log10(self.num_docs / df_arr))
        except Exception as e:
            print('TF-iDF矩阵中异常：')
            print(type(e), e)

    def attr_show(self):
        """
        显示属性数据格式等
        """
        print('Total documents:', self.num_docs)
        print('DF长度：', len(self.df))
        print('TF Shape:', self.tf.shape)
        print('Inverted List 长度:', len(self.inverted_index))


class Retriever:
    """
    Base Class for retrieving
    """

    def __init__(self, query: str, *args, **kwargs):
        """
        :param query: 输入的查询命令
        """
        self.searched = False  # 初始化时未执行过查询:
        self.query = query
        self.result = []  # 文档集合（文件绝对路径名）

    def action(self, *args, **kwargs):
        if self.searched:
            pass
        else:
            # 执行查询
            self.searched = True
            self.parse()

    def represent(self, *arg, **kwargs):
        pass


class BoolRetriever(Retriever):
    def __init__(self, query: str):
        """
        :param query: str类型query命令，需将其解析为suffix expression（list类型）
        """
        super(BoolRetriever, self).__init__(query)
        self.suffix = []

    def parse(self):
        """
        Parse the retrieval command
        """
        infix = re.split('(\W)', self.query)
        result = []
        stack = []
        op_priority = {'not': 3, 'and': 2, 'or': 1}
        for c in infix:
            if c == '' or c == ' ':
                continue
            if c == '(':
                stack.append(c)
            elif c == ')':
                while stack[-1] != '(':
                    result.append(stack.pop())
                stack.pop()
            elif c in op_priority.keys():
                while stack and stack[-1] in op_priority.keys() and op_priority[c] <= op_priority[stack[-1]]:
                    result.append(stack.pop())
                stack.append(c)
            else:
                # 将检索词条词根化
                snowball = nltk.SnowballStemmer('english')
                c = snowball.stem(c)
                wnl = nltk.WordNetLemmatizer()
                c = wnl.lemmatize(c)
                result.append(c)  # 此时将token加入后缀表达式
        while stack:
            result.append(stack.pop())
        self.suffix.clear()
        self.suffix = result

    def action(self, document: DocumentProcessed):
        if self.searched:
            pass
        else:
            self.searched = True
            self.parse()

            # 读取后缀表达式，执行查询
            docID_set = set(range(document.num_docs))  # 这是个集合类型，便于后续操作
            token_set = set(document.token_dict.keys())
            stack = []  # 只存储运算结果集合的栈

            for c in self.suffix:
                if c == 'not':
                    docID = stack.pop()
                    stack.append(docID_set - docID)
                elif c == 'and':
                    docID2 = stack.pop()
                    docID1 = stack.pop()
                    stack.append(docID1 & docID2)
                elif c == 'or':
                    docID2 = stack.pop()
                    docID1 = stack.pop()
                    stack.append(docID1 | docID2)
                else:
                    if c in token_set:  # 是待查询词条
                        docID = set(document.inverted_index[c])
                        stack.append(docID)  # 词条对应文档ID集合直接压栈
                    else:  # 空集合压栈
                        stack.append(set())

            for id in stack[-1]:
                self.result.append(document.files_path[id])  # result是文档ID对应的文档路径名（list）

    def represent(self, num=10):
        """
        :param num: 最大显示数量
        """
        length = len(self.result)
        if length == 0:
            print("There is no document satisfying your query requirement!")
        elif length <= num:
            print("Results (absolute file path) is/are:")
            for i in range(length):
                print(self.result[i])
        else:
            print("Results (absolute file path) is/are:")
            for i in range(num):
                print(self.result[i])
            print('There is/are still {} results not shown'.format(length - num))


class SemanticRetriever(Retriever):
    """
    语义检索类
    """

    def __init__(self, query: str):
        super(SemanticRetriever, self).__init__(query)
        self.query_arr = np.array([])

    def action(self, document: DocumentProcessed):
        """
        :param document: 基于该DocumentProcessed类中的tfidf矩阵进行查询和排序
        """
        if self.searched:
            pass  # 已经生成查询结果的不用执行该方法
        else:
            self.searched = True
            # str类型query命令待解析为1-D ndarray类型
            query_list = self.query.split()

            # 词根化
            snowball = nltk.SnowballStemmer('english')
            query_list = [snowball.stem(w) for w in query_list]
            wnl = nltk.WordNetLemmatizer()
            query_list = [wnl.lemmatize(w) for w in query_list]

            # 向量化
            self.query_arr = np.zeros(document.tfidf.shape[0])
            for token in query_list:
                if token in document.token_dict.keys():
                    self.query_arr[document.token_dict[token]] = 1

            # 归一化向量，执行查询（内积），进行比较和排序
            tfidf_arr = document.tfidf.toarray()
            mat_norm = np.linalg.norm(tfidf_arr, axis=0)
            mat_norm[mat_norm == 0] = 1  # 列向量norm为0说明该列元素全为0
            tfidf_arr /= mat_norm
            self.query_arr /= np.linalg.norm(self.query_arr)

            result_vec = np.array([np.dot(tfidf_arr[:, i], self.query_arr) for i in range(document.tfidf.shape[1])])
            order = np.flip(result_vec.argsort())
            self.result = [document.files_path[i] for i in order]  # result是文档ID对应的文档路径名（list）

    def represent(self, num=10):
        """
        :param num: 最大显示数量
        """
        if len(self.result) == 0:
            print("Your query has not been executed!")
        else:
            print("Results (absolute file path) is/are:")
            for i in range(num):
                print(self.result[i])
