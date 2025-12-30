#include <bits/stdc++.h>
using namespace std;

typedef struct BiNode{
    int data;
    struct BiNode *lchild, *rchild;
    BiNode(int val=0):data(val),lchild(nullptr),rchild(nullptr){}
} BiNode, *BiTree;

void PrintIndent(BiTree T, int level){//凹入表示
    if (T == nullptr) return;
    PrintIndent(T->lchild, level + 1);
    cout << string(4 * (level - 1), ' ') << T->data << endl;
    // 再打印左子树
    PrintIndent(T->rchild, level + 1);
}

void InOrder(BiTree T){
    if(T==nullptr){
        return;
    }
    InOrder(T->lchild);
    cout << ' ' << T->data;
    InOrder(T->rchild);
}

// 插入节点到以R为根的二叉排序树（引用传参以便修改根指针）
void Insert(BiTree &R, BiTree T){//注意⚠️！这里加上&引用符号就能取到实参的值，但不加R就只会拷贝一份传入的root的地址，不能真正改变root的值
    if (T == nullptr) return;
    if (R == nullptr){
        R = T; // 这个操作只能也必须放在这里，如果是直接return后再执行R->lchild = T;就是没有考虑到如果执行到更深处的情况，那时即便递归已经把节点插到更深的位置，之后 R->lchild = T 仍然把原来的左子树整个替换成 T，导致丢失子树或排序错误。
        return;
    }
    if (T->data < R->data) Insert(R->lchild, T);
    else if (T->data > R->data) Insert(R->rchild, T);
}

int main(){
    int val;
    if (!(cin >> val)) return 0;
    if (val == 0) return 0;
    BiTree root = new BiNode(val);
    while (cin >> val){
        if (val == 0) break;
        BiTree node = new BiNode(val);
        Insert(root, node);
    }
    PrintIndent(root,1);
    cout << endl;
    InOrder(root);
    cout << endl;
}