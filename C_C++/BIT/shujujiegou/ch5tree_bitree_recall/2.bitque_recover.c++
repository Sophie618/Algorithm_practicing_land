#include<bits/stdc++.h>
using namespace std;

typedef struct BiNode{
    char data;
    BiNode *lchild, *rchild;
    BiNode(char val='#'):data(val),lchild(nullptr),rchild(nullptr) {}
} BiNode, *BiTree;//*BiTree是定义了一个指向BiNode节点类型的指针

// queue<BiTree> q;的等价定义
// typedef struct LinkNode{//链式队列的节点
//     BiTree *data;//数据存放指向树节点的指针而不是节点
//     struct LinkNode *next;
// } LinkNode;

// typedef struct{//队列具有前驱和后驱指针
//     LinkNode *front;
//     LinkNode *rear;
// } LinkQueue;

int idx;
void Print(BiTree T){//层级打印
    if (T == nullptr)
    {
        return;
    }
    queue<BiTree> q;
    q.push(T);
    while(!q.empty()){
        BiTree curr = q.front();//取出队首节点
        q.pop();
        cout << curr->data;

        if(curr->lchild!=nullptr){
            q.push(curr->lchild);
        }
        if(curr->rchild!=nullptr){
            q.push(curr->rchild);
        }
    }
}

BiTree CreateBitree(const string& inOrder, const string& postOrder){
    if (inOrder.empty() || postOrder.empty())
    {
        return nullptr;
    }
    BiTree root = new BiNode(postOrder.back()); //.back()取到后序序列的最后一个元素作为根节点
    //在中序序列中找到根节点
    int rootIdx = inOrder.find(root->data);
    //划分左右子树的中序序列
    string leftIn = inOrder.substr(0, rootIdx);//从0到rootIdx-1
    string rightIn = inOrder.substr(rootIdx + 1);//从rootIdx+1到结尾
    //划分左右子树的后序序列
    string leftPost = postOrder.substr(0, leftIn.size());//节点个数相等
    string rightPost = postOrder.substr(leftIn.size(), postOrder.size() - 1-leftIn.size());

    //递归构建左右子树并挂载到根节点上
    root->lchild = CreateBitree(leftIn, leftPost);
    root->rchild = CreateBitree(rightIn, rightPost);
    
    return root;
}

int main(){
    BiTree root = nullptr;
    string inOrder, postOrder;

    getline(cin, inOrder); // 读取第一行（直到换行，丢弃换行符）
    getline(cin, postOrder); // 读取第二行（直到换行，丢弃换行符）

    root = CreateBitree(inOrder, postOrder);
    Print(root);
    cout << endl;

    return 0;
}