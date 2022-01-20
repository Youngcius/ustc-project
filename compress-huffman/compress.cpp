#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<iostream>

using namespace std;
#define MAX 256 
#define NAMESIZE 20

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0



typedef int Status;
typedef struct {
	unsigned int weight;
	unsigned int parant, lchild, rchild;
}HTNode, *HuffmanTree;
typedef char **Huffmancode;
/*全局变量*/
int weights[MAX] = { 0 };

/*辅助队列*/
typedef unsigned char QElemtype;
typedef struct QNode {
	QElemtype data;
	struct QNode *next;
}QNode, *QueuePtr;
typedef struct {
	QueuePtr front;
	QueuePtr rear;
}LinkQueue;
Status InitQueue(LinkQueue &Q);
Status DestroyQueue(LinkQueue &Q);
int QueueLength(LinkQueue Q);
bool QueueEmpty(LinkQueue Q);
Status EnQueue(LinkQueue &Q, QElemtype man);
Status DeQueue(LinkQueue &Q, QElemtype &man);
unsigned char BitQueue(LinkQueue &Q) {//取八位01010101’字符‘变成一个unsigned char
	unsigned char ch1; unsigned char ch = 0;
	unsigned char left = 1;
	int i, j;
	for (i = 1; i <= 8; i++) {
		if (QueueEmpty(Q))break;
		DeQueue(Q, ch1);
		ch1 -= 48;////
		ch1 <<= (8 - i);
		ch = ch | ch1;
	}
	for (j = 8; j >= i; j--) {
		ch = ch | left;
		left <<= 1;
	}
	return ch;
}
unsigned char TakeLeftQueue(LinkQueue &Q) {
	unsigned char ch = 0, ch1;
	for (int i = 1; i <= 8; i++) {
		if (QueueEmpty(Q))break;
		DeQueue(Q, ch1);
		ch1 -= 48;
		ch1 <<= (8 - i);
		ch = ch | ch1;
	}
	return ch;
}
/*辅助函数*/

void Bytechars(unsigned char ch, char *&Code) {///////////////////////
	unsigned char ch1 = ch; int i;
	for (i = 1; i <= 8; i++) {
		ch1 = ch;
		ch1 >>= (8 - i);
		ch1 = ch1 & 1;
		Code[i - 1] = ch1 + 48;///////0,,1变成'0',,'1'
	}
}
Status ByteQueue(LinkQueue &Q, char *s) {
	for (char *p = s; *p; p++) {
		EnQueue(Q, (unsigned char)*p);
	}
	return OK;
}
void Huffmancoding(HuffmanTree &HT, Huffmancode &HC, int *w, int n);
void SelectMin(HuffmanTree HT, int n, int &s1, int &s2);
void SelectMin(HuffmanTree HT, int n, int &s1, int &s2) {//HT[s1].weight < HT[s2].weught
	HT[0].weight = 1000000000;
	int i; s1 = s2 = 0;
	for (i = 1; i <= n; i++) {
		if (HT[i].parant == 0) {
			if (HT[i].weight < HT[s1].weight)s1 = i;
		}//找出权最小的对应序号s1
	}
	for (i = 1; i <= n; i++) {
		if (HT[i].parant == 0) {
			if ((i != s1) && (HT[i].weight < HT[s2].weight))s2 = i;
		}//找出权次小的s2
	}
}
void Huffmancoding(HuffmanTree &HT, Huffmancode &HC, int *weights, int n) {//weights：0----n-1
	if (n <= 1)return;
	HuffmanTree p; int i, s1, s2;
	int m = n * 2 - 1;
	HT = (HuffmanTree)malloc((m + 1) * sizeof(HTNode));
	if (!HT)exit(10);
	HT[0].weight = HT[0].lchild = HT[0].rchild = HT[0].parant = 0;
	for (i = 1; i <= n; i++) {
		HT[i].weight = weights[i - 1];
		HT[i].lchild = HT[i].rchild = HT[i].parant = 0;
	}
	for (; i <= m; i++)HT[i].weight = HT[i].lchild = HT[i].rchild = HT[i].parant = 0;
	for (i = n + 1; i <= m; i++) {//构建HUffman树
		SelectMin(HT, i - 1, s2, s1);//权重s2<s1
		HT[s1].parant = HT[s2].parant = i;
		HT[i].lchild = s1; HT[i].rchild = s2;//左孩子权大，右孩子权小
		HT[i].weight = HT[s1].weight + HT[s2].weight;
	}
	/*从叶子向根部逆向求哈夫曼编码*/
	HC = (Huffmancode)malloc((n + 1) * sizeof(char *)); if (!HC)exit(10);
	char *code = (char *)malloc(n * sizeof(char));//非叶子节点n-1个，故而编码长度最大n-1
	code[n - 1] = '\0';
	int start, c, f;
	for (i = 1; i <= n; i++) {
		start = n - 1;
		for (c = i, f = HT[i].parant; f != 0; c = f, f = HT[f].parant) {
			if (HT[f].lchild == c)code[--start] = '0';
			else code[--start] = '1';
		}
		HC[i] = (char *)malloc((n - start) * sizeof(char));
		strcpy(HC[i], &code[start]);
	}
	free(code);
	//
}

void ScanWeight(char *FileName) {
	FILE *fp;
	for (int i = 0; i < MAX; i++)weights[i] = 0;
	if (!(fp = fopen(FileName, "rb"))) {
		cout<<"不能打开待压缩文件（文件不存在在或不在当前目录）"<<endl;
		exit(1);
	}	
	unsigned char ch;
	while (!feof(fp)) {
		ch = fgetc(fp);
		weights[ch]++;
	}
	fclose(fp);
}
void Compress(char *FileName) {
	FILE *fp1, *fp2;
	HuffmanTree HT; Huffmancode HC;
	ScanWeight(FileName);
	Huffmancoding(HT, HC, weights, MAX);///////
	char ZipName[NAMESIZE] = { 0 }; char *p;
	for (p = FileName; *p != '.'; p++);
	strncpy(ZipName, FileName, p - FileName);
	strcat(ZipName, ".huff");////////
	if (!(fp2 = fopen(ZipName, "wb"))) {//写入压缩文件
		cout<<"不能创建压缩文件"<<endl; 
		exit(1);
	}
	char expandname[4]; strcpy(expandname, p + 1);
	fwrite(expandname, sizeof(char), 4, fp2);//4个字节存储拓展名
	int mm = 2 * (MAX * 2 - 1);//

	fwrite(HT, sizeof(HTNode), 2 * MAX, fp2);
	/*队列，8位写入*/
	if (!(fp1 = fopen(FileName, "rb"))) {//再读待压缩文件
		cout<<"不能打开待压缩文件（文件不存在在或不在当前目录）"<<endl;
		exit(1);
	}
	unsigned char ch; char s[MAX];
	LinkQueue Q; InitQueue(Q);
	while (!feof(fp1)) {
		ch = fgetc(fp1);
		strcpy(s, HC[ch + 1]);//字符ch对应一串01编码字符串表示为s；；；
		ByteQueue(Q, s);//入队
		while (1) {
			if (QueueLength(Q) >= 8) {//满8位转换成一个字节信息写入fp2 
				ch = BitQueue(Q);/////////////重要函数
				fputc(ch, fp2);
			}
			else break;
		}

	}//是否需要记录其是否满8位，补了几位
	if (QueueLength(Q)) {//最后未满8位补"111"
		ch = TakeLeftQueue(Q);
		fputc(ch, fp2);
	}
	fclose(fp1);
	fclose(fp2);
}
void Decompress(char *ZipName) {
	FILE *fp1, *fp2;
	HuffmanTree HT; //Huffmancode HC;
	int mm = 2 * MAX - 1; int i, j;
	HT = (HuffmanTree)malloc((mm + 1) * sizeof(HTNode)); if (!HT)exit(10);
	char DeZipName[NAMESIZE] = { 0 };
	char *p;/*设置文件名*/
	for (p = ZipName; *p != '.'; p++);
	strncpy(DeZipName, ZipName, p - ZipName);
	strcat(DeZipName, "_dezip.");////////
	if (!(fp1 = fopen(ZipName, "rb"))) {
		cout<<"不能打开待解压文件（文件不存在或不在当前目录）"<<endl;
		exit(1);
	}
	for (p = DeZipName; *p != '.'; p++);
	fread(p + 1, sizeof(char), 4, fp1);
	fread(HT, sizeof(HTNode), mm + 1, fp1);//构建哈夫曼树
	HT[0].weight = 100000;

	if (!(fp2 = fopen(DeZipName, "wb"))) {
		cout<<"不能创建解压文件"<<endl;
		exit(1);
	}//构建解压文件;;;边读边写
	char *Code; unsigned char ch;
	if (!(Code = (char *)malloc(8 * sizeof(char))))exit(10);
	unsigned char ascii;//解码待求序号
	j = mm;
	while (!feof(fp1)) {
		ch = fgetc(fp1);
		Bytechars(ch, Code);//将ch转换为01编码字符串存在Code内 Code:010101'\0'

		for (i = 0; i < 8; i++) {////根据huffman树从根向叶子找  
			if (Code[i] == '\0')putchar('@');//////////////////////////
			if (Code[i] == '0')j = HT[j].lchild;
			if (Code[i] == '1')j = HT[j].rchild;
			if (HT[j].lchild == 0) {// && HT[j].rchild == 0
				ascii = (unsigned char)(j - 1);/////
				fputc(ascii, fp2);
				j = mm;/////再从根开始
			}
		}
	}
	free(Code); free(HT);
	fclose(fp1);
	fclose(fp2);
}


int main(){
	char FileName[NAMESIZE], ZipName[NAMESIZE], DeZipName[NAMESIZE];
	int choice;
	cout << "Please choose your operation:" << endl;
	cout << "1.Compress---------------2.Decopress---------------" << endl;
	cout << "Choice:  ";
	cin >> choice;
	switch (choice) {
	case 1:

		cout << "Please input the name of the file you want to copress:   ";
		scanf("%s", FileName);
		Compress(FileName);
		cout << "The compression has been finished." << endl;
		break;
	case 2:

		cout << "Please input the name of the file you want to decopress:   ";
		scanf("%s", ZipName);
		Decompress(ZipName);

		cout << "The decompression has been finished." << endl;
		break;
	default:
		cout << "The choice is an invalid operation!" << endl;
		break;
	}




	return 0;
}
Status InitQueue(LinkQueue &Q) {
	Q.front = Q.rear = (QueuePtr)malloc(sizeof(QNode));
	if (!Q.front || !Q.rear)return ERROR;
	Q.front->next = NULL;
	return OK;
}
Status DestroyQueue(LinkQueue &Q) {
	//销毁队列Q
	while (Q.front) {
		Q.rear = Q.front->next;
		free(Q.front);
		Q.front = Q.rear;
	}
	return OK;
}
int QueueLength(LinkQueue Q) {
	if (Q.front == Q.rear)return 0;
	QueuePtr p = Q.front; int n = 0;
	for (p = p->next; p != Q.rear; p = p->next, n++);
	n++;
	return n;
}
bool QueueEmpty(LinkQueue Q) {
	//	if (!Q.front)exit(1);
	if (Q.front == Q.rear)return true;
	else return false;
}
Status EnQueue(LinkQueue &Q, QElemtype man) {
	QueuePtr p = (QueuePtr)malloc(sizeof(QNode));
	if (!p)return ERROR;
	p->data = man; p->next = NULL;
	Q.rear->next = p;
	Q.rear = p;
	return OK;
}
Status DeQueue(LinkQueue &Q, QElemtype &man) {
	if (Q.front == Q.rear)return ERROR;
	QueuePtr p;
	p = Q.front->next;
	man = p->data;
	Q.front->next = p->next;
	if (Q.rear == p)Q.rear = Q.front;
	free(p);
	return OK;
}

