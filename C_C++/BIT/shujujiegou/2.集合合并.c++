/* PRESET CODE BEGIN - NEVER TOUCH CODE BELOW */  
 
#include <stdio.h>  
#include <stdlib.h>  
typedef struct node  
{   int         data;  
    struct node * next;  
} NODE;  
  
NODE * find( NODE * , int * );  
void outputring( NODE * );  
void change( int , int , NODE * );  
void outputring( NODE * pring )  
{   NODE * p;  
    p = pring;  
    if ( p == NULL )  
        printf("NULL");  
    else  
        do  {   printf("%d", p->data);  
            p = p->next;  
        } while ( p != pring );  
    printf("\n");  
    return;  
}  
  
int main()  
{   int n, m;  
    NODE * head, * pring;  
  
    scanf("%d%d", &n, &m);  
    head = (NODE *)malloc( sizeof(NODE) );  
    head->next = NULL;  
    head->data = -1;  
  
    change( n, m, head );  
    pring = find( head, &n );  
    printf("ring=%d\n", n);  
    outputring( pring );  
  
    return 0;  
}  

#include <vector>
#include <algorithm>
using namespace std; 
void change(int n, int m, NODE *head)
{
    NODE *p = head;
    vector<int> remainder;
    vector<NODE*> nodes;
    while (n > 0)
    {
        if (find(remainder.begin(), remainder.end(), n) != remainder.end())//找到重复的余数
        {
            int index =find(remainder.begin(), remainder.end(), n) - remainder.begin();
            p->next=nodes[index];
            return;
        }
        else{
            p->next = (NODE *)malloc(sizeof(NODE));
            p->next->data=n*10/m;
            remainder.push_back(n);
            nodes.push_back(p->next);
        }
        n = (n * 10)%m;
        p = p->next;
    }
    p->next = NULL;  // 有限小数，结尾设为NULL
}
  
 
NODE * find( NODE * head, int * n ) 
{ 
    vector<NODE*> nodes;
    NODE *p = head;
    while (p->next != NULL){
        if (find(nodes.begin(), nodes.end(), p->next) != nodes.end()){
            int index =find(nodes.begin(), nodes.end(), p->next) - nodes.begin();
            *n=nodes.size()-index;
            return nodes[index];
        }
        nodes.push_back(p->next);
        p = p->next;
    }
    *n=0;
    return NULL;
} 
  
  
/* PRESET CODE END - NEVER TOUCH CODE ABOVE */ 



