#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int n, k, m;

typedef struct person
{
    int id;
    struct person *next;
} person;

int main()
{
    scanf("%d,%d,%d", &n, &k, &m);
    if (n < k)
    {
        printf("k should not bigger than n.\n");
        return 0;
    }
    else if (n < 1 || k < 1 || m < 1)
    {
        printf("n,m,k must bigger than 0.\n");
        return 0;
    }
    else if (n == 1)
    {
        printf("1\n");
        return 0;
    }
    else
    {
        person *head = (person *)malloc(sizeof(person)); // 记录头节点
        head->id = 1;
        person *current = head;

        for (int i = 2; i <= n; i++)
        {
            person *newNode = (person *)malloc(sizeof(person));
            newNode->id = i;
            current->next = newNode;
            current = newNode;
        }
        current->next = head; // 形成环

        person *prev = head;
        while (prev->next->id != k)
        {
            prev = prev->next;
        } // 找到第k个人的前一个

        int count = 0;
        int remain = n;

        while (remain > 1) // 当链表中只剩一个人时结束
        {
            for (int i = 1; i < m; i++)
            {
                prev = prev->next;
            }
            person *todelete = prev->next;
            printf("%d", todelete->id);
            prev->next = todelete->next;
            free(todelete);
            remain--;
            count++;

            // 输出控制
            if (remain > 0)
            {
                if (count % 10 == 0)
                    printf("\n");
                else
                    printf(" ");
            }
        }
        printf("%d\n", prev->id);
        free(prev);
    }
    return 0;
}