#include<bits/stdc++.h>
using namespace std;

typedef enum{
    ATOM,
    LIST
} ElemTag;

typedef int Status;

typedef struct GLNode{
    ElemTag tag;
    union{
        char atom;
        struct{
            struct GLNode *hp, *tp;
        } ptr;
    };
} *GList, GLNode;

int idx=0;
Status CreateGList(GList &L, char *S)
{
    char ch = S[idx++];//读取后下标已经挪到后一位
    if(ch=='\0'||ch=='\n'){
        L = NULL;
        return 1;
    }
    if(ch=='('){
        L = new GLNode;
        L->tag = LIST;
        if(S[idx]==')'){//空表
            L->ptr.hp = NULL;
            L->ptr.tp = NULL;
            idx++;
            return 1;
        }
        CreateGList(L->ptr.hp, S);
        if(S[idx]==','){
            idx++;
        }
        if(S[idx]==')'){
            L->ptr.tp = NULL;
            idx++;
        }
        else{
            GList tail = new GLNode;
            tail->tag = LIST;
            CreateGList(tail->ptr.hp, S);
            tail->ptr.tp = NULL;
            L->ptr.tp = tail;
            while(S[idx]==','){
                idx++;
                GList newtail = new GLNode;
                newtail->tag = LIST;
                CreateGList(newtail->ptr.hp, S);
                newtail->ptr.tp = NULL;
                tail->ptr.tp = newtail;
                tail = newtail;
            }
            if(S[idx]==')'){
                idx++;
            }
        }
    }
    else if(ch>='a'&&ch<='z'){
        L = new GLNode;
        L->tag = ATOM;
        L->atom = ch;
    }
    else{
        L = NULL;
    }
    return 1;
}

GList GetHead(GList L){
    if(L==NULL||L->tag==ATOM){
        return NULL;
    }
    return L->ptr.hp;
}

GList GetTail(GList L){
    if(L==NULL||L->tag==ATOM){
        return NULL;
    }
    return L->ptr.tp;
}

void DestroyGList(GList &L){
    if(L==NULL){
        return;
    }
    if (L->tag == LIST){
        DestroyGList(L->ptr.hp);
        DestroyGList(L->ptr.tp);
    }
    delete L;
    L = NULL;
}

void PrintGList(GList L){
    if(L==NULL){
        return;
    }
    if(L->tag==ATOM){
        cout << L->atom;
    }
    else{
        cout << "(";
        GList p = L;
        bool first = true;
        while(p!=NULL){
            if(p->ptr.hp!=NULL){
                if (!first)
                {
                    cout << ",";
                }
                PrintGList(p->ptr.hp);
                first = false;
            }
            p = p->ptr.tp;
        }
        cout << ")";
    }
}

int main()
{
    char S[1000];
    cin.getline(S, 1000);

    idx = 0;
    GList L;
    CreateGList(L, S);
    cout << "generic list: ";
    PrintGList(L);
    cout << endl;
    int op;
    while(cin>>op){
        if(L==NULL||L->tag==ATOM){
            break;
        }
        if(op==1){
            GList tail = GetTail(L);
            cout << "destroy tail" << endl;
            if(tail!=NULL){
                DestroyGList(tail);
            }
            GList head = GetHead(L);
            cout<<"free list node" << endl;
            delete L;
            L = head;
            cout << "generic list: ";
            PrintGList(L);
            cout << endl;
        }
        else if (op == 2)
        {
            GList head = GetHead(L);

            if (head != NULL)
            {
                if(head->tag==ATOM){
                    cout<<"free head node" << endl;
                    delete head;
                }
                else{
                    cout << "free head node" << endl;
                    DestroyGList(head);
                }
            }
            GList tail = GetTail(L);
            cout << "free list node" << endl;
            delete L;
            L = tail;
            cout << "generic list: ";
            if (L == NULL)
            {
                cout << "()";
            }
            else
            {
                PrintGList(L);
            }
            cout << endl;
        }
    }
    DestroyGList(L);
    return 0;
}

/*题后总结*/
/*
1.每次判断先行判断null
2.判断完空再判断某些自定义条件，比如是否是第一个元素
3.delete操作后要及时更新指针为null，防止野指针
================================================================================*/