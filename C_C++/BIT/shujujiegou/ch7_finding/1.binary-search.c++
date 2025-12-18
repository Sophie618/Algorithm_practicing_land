#include <bits/stdc++.h>
using namespace std;
int const MAXN = 1000005;
int T[MAXN];

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);

    int n;
    if (!(cin >> n))
        return 0;

    for (int i = 0; i < n; i++)
    {
        cin >> T[i];
    }

    // 题目说“排好序后”，所以必须排序。
    // 如果输入已经有序，sort 也会很快。
    sort(T, T + n);

    bool found = false;
    for (int i = 0; i < n; i++)
    {
        if (T[i] == i)
        {
            cout << i << " ";
            found = true;
        }
        // 剪枝优化：如果 T[i] > i，因为是不同的整数，后面肯定都 > i，可以直接 break
        else if (T[i] > i)
        {
            break;
        }
    }

    if (!found)
    {
        cout << "No ";
    }
    cout << endl;

    return 0;
}