#include<algorithm>
#include<bits/stdc++.h>
using namespace std;

// 方式一：多叉树直接表示（vector存孩子）
typedef struct TreeNode
{
    char val;
    vector<TreeNode *> children;
    TreeNode(char x) : val(x) {}
}TreeNode;

TreeNode *genTree(const char *pre[], const int *deg[], int &pos,int n){//这里加&是因为pos是作为全局变量，需要修改
    if(pos>=n){//位置已经超过了节点数
        return nullptr;
    }
    TreeNode*T = new TreeNode(pre[pos]);
    pos++;//挪到下一个位置作准备
    for (int j = 0; j < deg[i]; j++)
    {
        T->children.push_back() = genTree(pre, deg, &pos,n);
    }
    return T;
}

TreeNode* build(int n,char pre[],int deg[]){
    int pos = 0;
    TreeNode *Root = genTree(pre[], deg[], pos,n);
}