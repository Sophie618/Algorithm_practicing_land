#include <iostream>
#include <cstring>
#include <cstdlib>
using namespace std;

typedef enum
{
    ATOM,
    LIST
} ElemTag; // ATOM==0: 原子, LIST==1: 子表
typedef int Status;

// 广义表的头尾链表存储表示
typedef struct GLNode
{
    ElemTag tag; // 公共部分，用于区分原子结点和表结点
    union
    {              // 原子结点和表结点的联合部分
        char atom; // atom 是原子结点的值域，char 类型
        struct
        { // 表结点的指针域 hp 和 tp
            struct GLNode *hp, *tp;
        } ptr;
    };
} *GList, GLNode;

/*
================================================================================
ASCII 示意图：广义表的头尾链表存储（示例： (a,b,(c,d,e)) ）

说明（与教材框图一致）：
- 表结点 [LIST]：携带指针域 hp / tp。hp 指向“本层当前元素”（可能是原子或子表），
    tp 指向“本层下一个表结点”（同层后继）。
- 原子结点 [ATOM]：仅有值域 atom，不带 tp；同层横向链接由外层的表结点通过 tp 完成。

顶层 (a,b,(c,d,e)) 的结构：

    [LIST L0] --tp--> [LIST L1] --tp--> [LIST L2] --tp--> NULL
            |hp               |hp               |hp
            v                 v                 v
     [ATOM a]          [ATOM b]          [LIST S0]        // S0 是子表 (c,d,e) 的首表结点

子表 c(d,e) 的结构：

    [LIST S0] --tp--> [LIST S1] --tp--> [LIST S2] --tp--> NULL
            |hp               |hp               |hp
            v                 v                 v
     [ATOM c]          [ATOM d]          [ATOM e]

要点回顾：
1) 同一层的“兄弟元素”由一串表结点通过 tp 串联；每个表结点的 hp 才指向真实元素。
2) 若元素是原子，则 hp→[ATOM x]；若元素是子表，则 hp→[LIST ...]（再递归同样的模式）。
3) 空表用 NULL 表示（教材记作 NIL）。

该示意图可直接对照本文件中的 PrintGList/DestroyGList 的遍历与递归方向：
- 打印：for(p = L; p; p = p->ptr.tp) 对 p->ptr.hp 打印（原子直接输出，子表递归）。
- 销毁：表结点先 Destroy(hp) 再 Destroy(tp)；原子直接 delete。
================================================================================
*/

int idx; // 全局变量，用于解析字符串
// 创建广义表
Status CreateGList(GList &L, char *S)
{
    char ch = S[idx++];

    if (ch == '\0' || ch == '\n')
    {
        L = NULL;
        return 1;
    }
    if (ch == '(')
    { // 开始创建子表
        L = new GLNode;
        L->tag = LIST;

        // 检查空表
        if (S[idx] == ')')
        {
            L->ptr.hp = NULL;
            L->ptr.tp = NULL;
            idx++;
            return 1;
        }

        // 递归创建表头
        CreateGList(L->ptr.hp, S);

        // 跳过逗号
        if (S[idx] == ',')
        {
            idx++;
        }

        // 创建表尾
        if (S[idx] == ')')
        {
            L->ptr.tp = NULL;
            idx++;
        }
        else
        {
            // 表尾是剩余元素组成的表
            GList tail = new GLNode;
            tail->tag = LIST;
            CreateGList(tail->ptr.hp, S);
            tail->ptr.tp = NULL;
            L->ptr.tp = tail;

            // 继续处理后续元素
            while (S[idx] == ',')
            {
                idx++;
                GList newTail = new GLNode;
                newTail->tag = LIST;
                CreateGList(newTail->ptr.hp, S);
                newTail->ptr.tp = NULL;
                tail->ptr.tp = newTail;
                tail = newTail;
            }

            if (S[idx] == ')')
            {
                idx++;
            }
        }
    }
    else if (ch >= 'a' && ch <= 'z')
    { // 原子
        L = new GLNode;
        L->tag = ATOM;
        L->atom = ch;
    }
    else
    {
        L = NULL;
    }

    return 1;
}

// 取表头
GList GetHead(GList L)
{
    if (L == NULL || L->tag == ATOM)
    {
        return NULL;
    }
    return L->ptr.hp;
}

// 取表尾
GList GetTail(GList L)
{
    if (L == NULL || L->tag == ATOM)
    {
        return NULL;
    }
    return L->ptr.tp;
}

// 销毁广义表
void DestroyGList(GList &L)
{
    if (L == NULL)
        return;

    if (L->tag == LIST)
    {
        DestroyGList(L->ptr.hp);
        DestroyGList(L->ptr.tp);
    }

    delete L;
    L = NULL;
}

// 打印广义表
void PrintGList(GList L)
{
    if (L == NULL)
    {
        return;
    }

    if (L->tag == ATOM)
    {
        cout << L->atom;
    }
    else
    {
        cout << "(";
        GList p = L;
        bool first = true;
        while (p != NULL)
        {
            if (p->ptr.hp != NULL)
            {
                if (!first)
                    cout << ",";
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
    while (cin >> op)
    {
        if (L == NULL || L->tag == ATOM)
        {
            break;
        }

        if (op == 1)
        { // 取表头
            GList tail = GetTail(L);

            // 释放表尾（无论是否为空都输出destroy tail）
            cout << "destroy tail" << endl;
            if (tail != NULL)
            {
                DestroyGList(tail);
            }

            // 保存表头
            GList head = GetHead(L);

            // 释放当前表结点
            cout << "free list node" << endl;
            delete L;

            L = head;

            cout << "generic list: ";
            PrintGList(L);
            cout << endl;
        }
        else if (op == 2)
        { // 取表尾
            GList head = GetHead(L);

            // 释放表头
            if (head != NULL)
            {
                if (head->tag == ATOM)
                {
                    cout << "free head node" << endl;
                    delete head;
                }
                else
                {
                    cout << "free head node" << endl;
                    DestroyGList(head);
                }
            }

            // 保存表尾
            GList tail = GetTail(L);

            // 释放当前表结点
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

    // 清理剩余资源
    if (L != NULL)
    {
        DestroyGList(L);
    }

    return 0;
}