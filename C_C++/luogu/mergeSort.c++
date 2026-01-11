#include <iostream>
#include <vector>
using namespace std;

// 辅助数组 B，用于暂存合并的数据
// 考试时如果允许，可以定义为全局变量，省去传参麻烦
int *B;

// 1. 核心合并函数：将 A[low...mid] 和 A[mid+1...high] 合并
// 这部分逻辑和你写的“合并链表”一模一样！
void Merge(int A[], int low, int mid, int high)
{
    int i, j, k;

    // 先把 A 中的数据复制到辅助数组 B 中
    for (k = low; k <= high; k++)
    {
        B[k] = A[k];
    }

    // i 指向左半边的起点，j 指向右半边的起点，k 指向归位的位置
    for (i = low, j = mid + 1, k = low; i <= mid && j <= high; k++)
    {
        if (B[i] <= B[j])
        { // 谁小谁下来（<= 保证了稳定性）
            A[k] = B[i++];
        }
        else
        {
            A[k] = B[j++];
        }
    }

    // 处理剩下的尾巴（类似于链表里的 if(p) ...）
    while (i <= mid)
        A[k++] = B[i++];
    while (j <= high)
        A[k++] = B[j++];
}

// 2. 递归主函数
void MergeSort(int A[], int low, int high)
{
    if (low < high)
    {                                // 只要还没切成单个元素，就继续切
        int mid = (low + high) / 2;  // 从中间切开
        MergeSort(A, low, mid);      // 排左边
        MergeSort(A, mid + 1, high); // 排右边
        Merge(A, low, mid, high);    // 左右归并
    }
}

// 考试调用示例
int main()
{
    int A[] = {49, 38, 65, 97, 76, 13, 27};
    int n = 7;

    // 初始化辅助数组（必须分配空间！这是 O(n) 空间的来源）
    B = (int *)malloc((n) * sizeof(int));

    MergeSort(A, 0, n - 1); // 注意是 n-1

    // 打印验证
    for (int i = 0; i < n; i++)
        cout << A[i] << " ";
    return 0;
}