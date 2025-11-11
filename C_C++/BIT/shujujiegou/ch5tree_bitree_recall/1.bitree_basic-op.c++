#include<bits/stdc++.h>
using namespace std;

struct ElemType{
    int value;
};//便于后续维护data的类型

typedef struct BiTNode{//递归定义
    ElemType data;
    struct BiTNode *lchild, *rchild;//左子树和右子树
}BiTNode, *BiTree;