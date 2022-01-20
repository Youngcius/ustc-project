import pickle
import os
import argparse
import warnings
from retriever import DocumentProcessed, BoolRetriever

DOCUMENT_NAME = 'document.pk'

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Bool search')
    parser.add_argument('-i', '--data_path', default='../dataset', type=str, help='set the data directory')
    parser.add_argument('-o', '--output_path', default='../output', type=str, help='set the output directory')
    parser.add_argument('-n', '--num_result', default=10, type=int, help='number of results (in order)')
    args = parser.parse_args()

    # 文档词条化预处理
    if DOCUMENT_NAME not in os.listdir(args.output_path):
        document = DocumentProcessed(in_path=args.data_path)
        document.attr_show()
        with open(os.path.join(args.output_path, DOCUMENT_NAME), 'wb') as f:
            pickle.dump(document, f)
    else:
        with open(os.path.join(args.output_path, DOCUMENT_NAME), 'rb') as f:
            document = pickle.load(f)

    cmd = ''  # 输入检索命令
    print("===============Bool retrieval system for mails===============")
    print("'and', 'or', 'not', '(', ')' are available operators for bool query; input '#' to end search action")

    while True:
        cmd = input("Input your query command: ").lower()  # 小写化输入
        if cmd == '#':
            break
        bool_retriever = BoolRetriever(query=cmd)
        bool_retriever.action(document=document)  # 解析检索命令, 执行检索算法, 返回检索结果
        bool_retriever.represent(num=args.num_result)  # 检索结果展示

    print("Query has exited. Thank you for your performance!")
