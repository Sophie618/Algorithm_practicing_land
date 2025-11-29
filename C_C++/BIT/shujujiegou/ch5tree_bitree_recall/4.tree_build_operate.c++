#include<bits/stdc++.h>
using namespace std;

struct Node{//定义多叉树节点
    char data;
    vector<Node *> children;//子节点列表存储指向子节点的指针
    Node(char c) : data(c) {}//构造函数初始化节点数据
    ~Node(){//析构函数，递归删除子节点
        for(auto child:children){
            delete child;
        }
    }
};

Node* buildTree(string subs){
    if(subs.empty()){
        return nullptr;//如果子串为空，直接返回null
    }
    char rootData = subs[0];//提取首字符作为根节点数据
    Node *root = new Node(rootData);//建立根节点

    if(subs.empty())
        return nullptr;//如果是空串()就直接返回空指针

    int sublen = subs.size();//计算串长度
    if (sublen <= 1)// 假若子串无任何子表，长度为一
    {               
        return root;//直接返回根节点
    }

    int start = subs.find('(');//找到第一个子表的左括号位置
    if(start==string::npos){
        return root;//假如没找到(说明没有子表，直接返回根节点
    }
    int depth = 0;//记录第一个子表起始索引及括号深度
    for (int i = start; i < sublen;++i){
        if(subs[i]=='('){
            depth++;//遇到左括号深度加一
        }
        else if(subs[i]==')'){
            depth--;//当前层级括号结束，深度减一
        }

        if(subs[i]==','&&depth==0){//遇到括号且深度为一时才划分子表,缺点是最后一个子表的后面是没有逗号的，所以会遗留一个子表待处理
            string childTable = subs.substr(start, i - start);
            string childcontent = childTable.substr(1, childTable.size() - 2);
            root->children.push_back(buildTree(childcontent));//返回的类型直接是节点指针类型，可以直接进入指针数组存贮，不必重新开一个变量专门存储
            start = i + 1;//移动到不是逗号的字符处（即更新到下一个子表的起始位置）
        }
    }

    // start还在最后一个子表的前括号处
    if (start < sublen)
    {
        string childTable = subs.substr(start, sublen - start);
        string childcontent = childTable.substr(1, childTable.size() - 2);
        root->children.push_back(buildTree(childcontent));
    }
    return root;//返回已经挂载了子树的根节点
}

//深度优先遍历打印输出
void Print(Node* root, int indentDepth){
    if(root==nullptr)
        return;//如果是空节点直接返回

    for (int i = 0; i < indentDepth * 4; i++){
        cout << ' ';//每层的缩进相差四个空格
    }
    cout << root->data << endl;

    for(auto child:root->children){
        Print(child, indentDepth + 1);//递归打印子节点，缩进深度加一
    }
}

//统计树的度和各个度所包含的节点数
void countDegree(Node* root,int& maxDegree,vector<int>& degreeCount){//记得用取址号修改实参，数组只有字符串可以不用取址
    if(root==nullptr){
        // degreeCount[0]++;//叶子节点数量++..错！！这里是没有节点！在递归上一层已经
        return;
    }

    int degree = root->children.size();//计算当前节点的度
    if(degree>=degreeCount.size()){
        degreeCount.resize(degree + 1, 0);//假如超过原来数组大小，进行合理扩展，并且初始化为0
    }
    degreeCount[degree]++;//相应度节点数++，如果degreeCount数组没有对应的度数空间，则扩展数组，并且这已经统计了叶子节点了！

    maxDegree = degree > maxDegree ? degree : maxDegree;//更新度数
    for(auto child:root->children){
        countDegree(child, maxDegree, degreeCount);//依次遍历孩子节点并进行统计
    }
}

int main(){
    string s;
    int maxDegree = 0;//相当于是开辟一片空间，maxDegree指向这片空间
    vector<int> degreeCount(1,0);//一个元素，内存占一个int大小，初始值为0
    getline(cin, s);//直接读取整行输入
    string input = s.substr(1, s.size() - 2);//从下标为1的位置开始取长度为size-2的子字符串，去掉输入中的最外层括号
    //开始构建多叉树
    Node *root = buildTree(input);//得到挂载好多棵树的根节点
    //dfs打印
    Print(root, 0);
    //打印度和相应节点数
    countDegree(root, maxDegree, degreeCount);
    cout << "Degree of tree: " << maxDegree << endl;
    for (int i = 0; i <= maxDegree;i++){
        cout << "Number of nodes of degree " << i << ": " << degreeCount[i] << endl;
    }
        return 0;
}