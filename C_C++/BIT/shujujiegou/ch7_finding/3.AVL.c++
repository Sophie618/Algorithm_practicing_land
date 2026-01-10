#include <bits/stdc++.h>
using namespace std;

typedef struct BiNode{
    char data;
    int height;//新增：当前节点高度
    struct BiNode *lchild, *rchild;
    BiNode(char data):data(data),height(1),lchild(nullptr),rchild(nullptr){}
} BiNode, *BiTree;

void PreOrder(BiTree T){
    if(T==nullptr){
        return;
    }
    cout << T->data;
    PreOrder(T->lchild);
    PreOrder(T->rchild);
}

void InOrder(BiTree T){
    if (T == nullptr)
    {
        return;
    }
    InOrder(T->lchild);
    cout << T->data;
    InOrder(T->rchild);
}

    
void PostOrder(BiTree T){
    if (T == nullptr)
    {
        return;
    }
    PostOrder(T->lchild);
    PostOrder(T->rchild);
    cout << T->data;
}

void PrintIndent(BiTree T,int h){
    if(T==nullptr){
        return;
    }
    PrintIndent(T->rchild,h+1);
    cout << string(4 * h , ' ') << T->data << endl;
    PrintIndent(T->lchild, h + 1);
}

int height(BiTree T)
{ // 计算某一节点的高度,最小值为1
    if (T == nullptr)
    {
        return 0;
    }
    return T->height; // 直接返回存储的高度
}

BiTree LeftRotation(BiTree T){//T是指向拐点的
    BiNode *rCh = T->rchild;//记录R处右节点的地址
    BiNode *rch_lch = rCh->lchild;//记录右节点的左孩子地址

    rCh->lchild = T;
    T->rchild = rch_lch;//这里是为了防止拐点处原有的右孩子指针出问题，要指向null

    //注意高度要更新
    T->height = 1 + max(height(T->lchild), height(T->rchild));
    rCh->height = 1 + max(height(rCh->lchild), height(rCh->rchild));

    return rCh;
}

BiTree RightRotation(BiTree T){
    BiNode *lCh = T->lchild;//记录第二个左孩子的地址，后面要变成根节点
    BiNode *lch_rch = lCh->rchild;

    lCh->rchild= T;
    T->lchild = lch_rch; // 也是避免原来的左孩子指针变成野指针

    // 注意高度要更新
    T->height = 1 + max(height(T->lchild), height(T->rchild));//从矮的开始更新
    lCh->height = 1 + max(height(lCh->lchild), height(lCh->rchild));

    return lCh;
}


BiTree Insert(BiTree T,char val){
    if(T==nullptr){
        return new BiNode(val);
    }
    if(val<T->data){
        T->lchild = Insert(T->lchild, val);
    }
    else
    {
        T->rchild = Insert(T->rchild, val);
    }
    //更新当前节点高度
    T->height = 1 + max(height(T->lchild), height(T->rchild));
    //检查是否平衡
    int balance = height(T->lchild) - height(T->rchild);
    if (balance > 1)//左边更重
    {
        if(val< T->lchild->data){//LL型
            return RightRotation(T);
        }
        else{//LR型，先左旋变LL再转回LL型处理
            T->lchild = LeftRotation(T->lchild);
            return RightRotation(T);
        }
    }
    if (balance < -1) // 右边更重
    {
        if (val > T->rchild->data)
        { // RR型
            return LeftRotation(T);
        }
        else
        {//RL
            T->rchild = RightRotation(T->rchild);
            return LeftRotation(T);
        }
    }

    //若平衡，则直接返回当前节点
    return T;
}

int main()
{
    string s;
    BiTree root=nullptr;
    cin >> s;
    for (size_t i = 0; i < s.length();i++)
    {                             // len(s)不是函数！！s.length()
        root = Insert(root, s[i]);//root在插入新节点的过程中是在不断更新的，最后返回的node就是调整后的根节点
    }
    cout << "Preorder: ";
    PreOrder(root);
    cout << endl;

    cout << "Inorder: ";
    InOrder(root);
    cout << endl;

    cout << "Postorder: ";
    PostOrder(root);
    cout << endl;

    cout << "Tree:" << endl;
    PrintIndent(root, 0);
    return 0;
}