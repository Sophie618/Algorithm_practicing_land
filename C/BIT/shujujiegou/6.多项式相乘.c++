/* PRESET CODE BEGIN - NEVER TOUCH CODE BELOW */  
 
#include <stdio.h>  
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <map>

typedef struct node  
{   int    coef, exp;  
    struct node  *next;  
} NODE;
  
void multiplication( NODE *, NODE * , NODE * );  
void input( NODE * );  
void output( NODE * );  
  
void input( NODE * head )  
{   int flag, sign, sum, x;  
    char c;  
  
    NODE * p = head;  
  
    while ( (c=getchar()) !='\n' )  
    {  
        if ( c == '<' )  
        {    sum = 0;  
             sign = 1;  
             flag = 1;  
        }  
        else if ( c =='-' )  
             sign = -1;  
        else if( c >='0'&& c <='9' )  
        {    sum = sum*10 + c - '0';  
        }  
        else if ( c == ',' )  
        {    if ( flag == 1 )  
             {    x = sign * sum;  
                  sum = 0;  
                  flag = 2;  
          sign = 1;  
             }  
        }  
        else if ( c == '>' )  
        {    p->next = ( NODE * ) malloc( sizeof(NODE) );  
             p->next->coef = x;  
             p->next->exp  = sign * sum;  
             p = p->next;  
             p->next = NULL;  
             flag = 0;  
        }  
    }  
}  
  
void output( NODE * head )  
{  
    while ( head->next != NULL )  
    {   head = head->next;  
        printf("<%d,%d>,", head->coef, head->exp );  
    }  
    printf("\n");  
}  

void multiplication( NODE *head1, NODE *head2, NODE *head3 )
{
    NODE *p1 = head1->next;
    NODE *p2 = head2->next;
    NODE *p3 = head3;
    NODE *q3 = head3;
    while (p1 != NULL)
    {
        while (p2 != NULL)
        {
            p3->next = (NODE *)malloc(sizeof(NODE));
            p3=p3->next;
            p3->coef = p1->coef * p2->coef;
            p3->exp = p1->exp + p2->exp;
            p3->next = NULL;
            p2 = p2->next;
        }
        p2=head2->next;
        p1=p1->next;
    }//乘毕


    if (head3->next == NULL) {//检验空结果
  	NODE *z = (NODE*)malloc(sizeof(NODE));
  	z->coef = 0; z->exp = 0; z->next = NULL;
  	head3->next = z;
  	return;
  }

    std::vector<NODE*> exp_nodes;
    p3=head3->next;
    while (p3 != NULL)
    {
        bool found=false;
        int index=-1;
        for (int i = 0; i < exp_nodes.size(); i++){
            if(exp_nodes[i]->exp==p3->exp){
                found=true;
                index=i;
                break;
            }
        }
        if (found){
            exp_nodes[index]->coef+=p3->coef;
            q3->next=p3->next;
            p3->next=NULL;//删除p3指向的重复节点
            p3=q3->next;
        }
        else{
            exp_nodes.push_back(p3);
            p3=p3->next;
            q3 = q3->next;
        }
    }//合并相同指数的节点

    // 使用有序映射按指数聚合
    std::map<int, int> expToCoef;
    for (NODE *p = head3->next; p != NULL; p = p->next) {
        expToCoef[p->exp] += p->coef;
    }

    // 断开旧链表并按指数升序重建，只保留非零系数
    head3->next = NULL;
    NODE *tail = head3;
    for (std::map<int, int>::iterator it = expToCoef.begin(); it != expToCoef.end(); ++it) {
        if (it->second == 0) continue;
        NODE *nd = (NODE *)malloc(sizeof(NODE));
        nd->coef = it->second;
        nd->exp = it->first;
        nd->next = NULL;
        tail->next = nd;
        tail = nd;
    }

    // 若所有项相互抵消为 0，多项式为 <0,0>
    if (head3->next == NULL) {
        NODE *z = (NODE *)malloc(sizeof(NODE));
        z->coef = 0;
        z->exp = 0;
        z->next = NULL;
        head3->next = z;
    }
}

int main()  
{   NODE * head1, * head2, * head3;  
  
    head1 = ( NODE * ) malloc( sizeof(NODE) );  
    input( head1 );  
  
    head2 = ( NODE * ) malloc( sizeof(NODE) );  
    input( head2 );  
  
    head3 = ( NODE * ) malloc( sizeof(NODE) );  
    head3->next = NULL;  
    multiplication( head1, head2, head3 );  
  
    output( head3 );  
    return 0;  
}  
  
/* PRESET CODE END - NEVER TOUCH CODE ABOVE */ 