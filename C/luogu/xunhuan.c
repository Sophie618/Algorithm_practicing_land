#include <stdio.h>

int factorial[100], sum[100];
int fac_len = 1, sum_len = 1;

int main()
{
    int n;
    scanf("%d", &n);

    factorial[0] = sum[0] = 1;

    for (int i = sum_len - 1; i >= 0; i--)
    {
        printf("%d", sum[i]);
    }
    printf("\n");

    return 0;
}