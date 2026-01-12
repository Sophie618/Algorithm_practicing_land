#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

// 用于存储递归过程中产生的 Median3 返回值
vector<int> median_records;

// 插入排序：处理小数组
void InsertionSort(vector<int> &arr, int left, int right)
{
    for (int p = left + 1; p <= right; p++)
    {
        int tmp = arr[p];
        int j;
        for (j = p; j > left && arr[j - 1] > tmp; j--)
        {
            arr[j] = arr[j - 1];
        }
        arr[j] = tmp;//这里是因为上面j已经--到了前面的一位了，这个位置就是插入的位置
    }
}

// 三者取中逻辑
// 作用：找到左、中、右的中值，并按照题目要求将其交换到首位(left)
int Median3(vector<int> &arr, int left, int right)
{
    int center = (left + right) / 2;

    // 也就是对 arr[left], arr[center], arr[right] 进行排序
    if (arr[left] > arr[center])
        swap(arr[left], arr[center]);
    if (arr[left] > arr[right])
        swap(arr[left], arr[right]);
    if (arr[center] > arr[right])
        swap(arr[center], arr[right]);

    // 此时关系为：arr[left] <= arr[center] <= arr[right]
    // arr[center] 即为中值

    // 题目要求：如median3不在首位，需要和首位元素交换位置
    // 将中值交换到 left 位置作为枢轴
    swap(arr[left], arr[center]);

    // 记录并在排序结束后输出
    median_records.push_back(arr[left]);

    return arr[left]; // 返回枢轴值
}

// 快速排序核心递归函数
void QSort(vector<int> &arr, int left, int right)
{
    int len = right - left + 1;

    // Cutoff值为5，不足(<=)5使用插入排序
    if (len > 5)
    {
        // 1. 选枢轴（三者取中并移至首位）
        int pivot = Median3(arr, left, right);

        // 2. 划分 (Partition)
        int i = left;
        int j = right;

        while (i < j)
        {
            // 从右向左找比枢轴小的
            while (i < j && arr[j] >= pivot)
                j--;
            // 从左向右找比枢轴大的
            while (i < j && arr[i] <= pivot)
                i++;

            if (i < j)
            {
                swap(arr[i], arr[j]);
            }
        }

        // 3. 将枢轴归位 (此时 i==j，且该位置的值一定 <= pivot)
        swap(arr[left], arr[i]);

        // 4. 递归处理左右子数组
        QSort(arr, left, i - 1);
        QSort(arr, i + 1, right);
    }
    else
    {
        // 长度 <= 5，使用插入排序
        InsertionSort(arr, left, right);
    }
}

int main()
{
    vector<int> arr;
    int val;

    // 处理输入，遇到非数字（如 #）停止
    while (cin >> val)
    {
        arr.push_back(val);
    }
    // 清除错误标志并消耗掉停止符，虽然对于本题逻辑不是必须，但为了严谨
    cin.clear();
    string dummy;
    cin >> dummy;

    if (!arr.empty())
    {
        QSort(arr, 0, arr.size() - 1);
    }

    // 输出排序结果
    cout << "After Sorting:" << endl;
    for (int i = 0; i < arr.size(); i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    // 输出 Median3 返回值
    cout << "Median3 Value:" << endl;
    if (median_records.empty())
    {
        cout << "none" << endl;
    }
    else
    {
        for (int i = 0; i < median_records.size(); i++)
        {
            cout << median_records[i] << " ";
        }
        cout << endl;
    }

    return 0;
}