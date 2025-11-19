#include<bits/stdc++.h>
using namespace std;
int idx ;

typedef struct BiNode{//递归定义
    char data;
    struct BiNode *lchild, *rchild;//左子树和右子树
    BiNode(char val='#'):data(val),lchild(nullptr),rchild(nullptr){}//包装为一个函数，内部值为字符型x
}BiNode, *BiTree;//BiTree是指向节点的指针

void CreateBiTree(BiTree *T, const string& s){//T是指向节点的指针的指针
    if (idx >= s.size())
    {
        (*T) = nullptr;//函数内部使用*是为了取当前指针所对应的值
        return;
    }
    char ch = s[idx++];

    // 若为'#'，当前节点为空
    if (ch == '#')
    {
        (*T) = nullptr;
        return;
    }
    *T= new BiNode(ch);
    CreateBiTree(&((*T)->lchild), s);
    CreateBiTree(&((*T)->rchild), s);
}

void PreOrder(BiTree T,string& res){
    if(T==nullptr)
        return;
    res += T->data;
    PreOrder(T->lchild, res);
    PreOrder(T->rchild, res);
}

void InOrder(BiTree T,string& res){
    if(T==nullptr)
        return;
    InOrder(T->lchild, res);//先把左边遍历完
    res += T->data;//回来的时候第二次经过再记录
    InOrder(T->rchild, res);
}

void PostOrder(BiTree T, string &res)
{
    if (T == nullptr)
        return;
    PostOrder(T->lchild, res);
    PostOrder(T->rchild, res);
    res += T->data; 
}

void SwapSubTree(BiTree T){
    if(T==nullptr)
        return;
    BiNode *temp = T->lchild;
    T->lchild = T->rchild;
    T->rchild = temp;
    SwapSubTree(T->lchild);
    SwapSubTree(T->rchild);
}

void PrintIndent(BiTree T,int level){
    if(T==nullptr){
        return;
    }
    cout << string(4 * (level - 1), ' ') << T->data << endl;
    PrintIndent(T->lchild, level + 1);
    PrintIndent(T->rchild, level + 1);
}

void DestroyBiTree(BiTree& T){
    if(T==nullptr){
        return;
    }
    DestroyBiTree(T->lchild);
    DestroyBiTree(T->rchild);
    delete T;
    T = nullptr;
}

int CountLeaf(BiTree T){
    if(T==nullptr){
        return 0;//空树叶子数为0
    }
    
    if(T->lchild==nullptr
        && T->rchild==nullptr){
        return 1;
    }
    return CountLeaf(T->lchild) + CountLeaf(T->rchild);
}

int main(){
    string input;
    BiTree root = nullptr;
    cin >> input;
    CreateBiTree(&root, input);
    cout << "BiTree" << endl;
    PrintIndent(root, 1);
    string pre, in, post;
    PreOrder(root, pre);
    InOrder(root, in);
    PostOrder(root, post);
    cout << "pre_sequence  : " << pre << endl;
    cout << "in_sequence   : " << in << endl;
    cout << "post_sequence : " << post << endl;

    //统计叶子数并输出
    int leaf;
    leaf=CountLeaf(root);
    cout << "Number of leaf: " << leaf << endl;

    //交换左右子树后凹入输出
    cout << "BiTree swapped" << endl;
    SwapSubTree(root);
    PrintIndent(root, 1);

    pre.clear(); in.clear(); post.clear();
    PreOrder(root, pre);
    InOrder(root, in);
    PostOrder(root, post);
    cout << "pre_sequence  : " << pre << endl;
    cout << "in_sequence   : " << in << endl;
    cout << "post_sequence : " << post << endl;
    DestroyBiTree(root);
    return 0;
}