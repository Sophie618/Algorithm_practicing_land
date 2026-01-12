#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * 堆调整函数 (Sift Down / Heap Adjust)
 * 作用：将 start 节点的子树调整为大根堆
 * @param arr   待调整的数组
 * @param start 当前需要调整的节点下标 (0-based)
 * @param size  当前堆的大小 (调整范围是 0 到 size-1)
 */
void HeapAdjust(vector<int> &arr, int start, int size)
{
    int parent = start;
    int child = 2 * parent + 1; // 左孩子下标
    int temp = arr[parent];     // 暂存根节点的值

    while (child < size)
    {
        // 如果右孩子存在，且右孩子比左孩子大，则将 child 指向右孩子
        if (child + 1 < size && arr[child] < arr[child + 1])
        {
            child++;
        }

        // 如果父节点的值已经大于等于较大的孩子，说明已经满足堆性质，退出
        if (temp >= arr[child])
        {
            break;
        }

        // 否则，将孩子上移
        arr[parent] = arr[child];

        // 继续向下筛选
        parent = child;
        child = 2 * parent + 1;
    }

    // 将原根节点的值放入最终位置
    arr[parent] = temp;
}

int main()
{
    // 1. 优化输入输出效率
    ios::sync_with_stdio(false);
    cin.tie(0);

    int n;
    if (!(cin >> n))
        return 0;

    vector<int> arr(n);
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    // 2. 初始建堆 (Build Max Heap)
    // 从最后一个非叶子节点开始，从下往上进行调整
    // 最后一个非叶子节点下标 = n / 2 - 1
    for (int i = n / 2 - 1; i >= 0; i--)
    {
        HeapAdjust(arr, i, n);
    }

    // 输出 1: 建堆结果
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    // 3. 第一次筛选
    // 交换堆顶和堆尾
    swap(arr[0], arr[n - 1]);
    // 对根节点进行调整，此时堆大小为 n-1
    HeapAdjust(arr, 0, n - 1);

    // 输出 2: 第一次筛选结果
    for (int i = 0; i < n - 1; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    // 4. 第二次筛选
    // 交换堆顶和堆尾 (注意此时堆尾是 arr[n-2])
    swap(arr[0], arr[n - 2]);
    // 对根节点进行调整，此时堆大小为 n-2
    HeapAdjust(arr, 0, n - 2);

    // 输出 3: 第二次筛选结果
    for (int i = 0; i < n - 2; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}