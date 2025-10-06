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
  
/* Here is waiting for you. 
void change( int n, int m, NODE * head ) 
{  
} 
 
NODE * find( NODE * head, int * n ) 
{ 
} 
*/  
  
/* PRESET CODE END - NEVER TOUCH CODE ABOVE */

#include <vector>
#include <algorithm>
using namespace std;

void change(int n, int m, NODE *head) {
    NODE *p = head;
    vector<int> remainder;
    vector<NODE*> nodes;
    
    while (n != 0) {
        if (find(remainder.begin(), remainder.end(), n) != remainder.end()) {
            int index = find(remainder.begin(), remainder.end(), n) - remainder.begin();
            p->next = nodes[index];
            return;
        }
        
        p->next = (NODE *)malloc(sizeof(NODE));
        p->next->data = (n * 10) / m;
        
        remainder.push_back(n);
        nodes.push_back(p->next);
        
        p = p->next;
        n = (n * 10) % m;
    }
    
    p->next = NULL;
}

NODE * find(NODE * head, int * n) {
    if (head->next == NULL) {
        *n = 0;
        return NULL;
    }
    
    NODE *slow = head->next;
    NODE *fast = head->next;
    
    // 第一步：快慢指针找相遇点
    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
        
        if (slow == fast) {
            break;
        }
    }
    
    // 没有环
    if (fast == NULL || fast->next == NULL) {
        *n = 0;
        return NULL;
    }
    
    // 第二步：找环的起点
    slow = head->next;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    
    // 第三步：计算环的长度
    int count = 1;
    NODE *p = slow->next;
    while (p != slow) {
        count++;
        p = p->next;
    }
    
    *n = count;
    return slow;
}

/* 快慢指针思路：
// 1. 快慢指针找相遇点
// 2. 找环的起点，计算环的长度 */

/*📊 测试用例追踪
以 29/33 为例（链表：head -> 8 -> 7 -> 回到8）
第一步：找相遇点
时刻0: slow=8, fast=8
时刻1: slow=7, fast=7  ← 相遇！

第二步：找环起点
slow回到8, fast在7
时刻0: slow=8, fast=8  ← 在起点相遇！

第三步：计算环长
从8开始：8 -> 7 -> 8
count = 2

结果：ring=2, 返回节点8*/