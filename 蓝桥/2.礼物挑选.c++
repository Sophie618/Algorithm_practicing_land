#include <bits/stdc++.h>
using namespace std;
static int counter[100005];
int main()
{
    int N, X;
    long long maxValue = 0;
    cin >> N >> X;
    
    for(int i = 0; i < N; i++){
        long long val;
        cin >> val;
        counter[val]++;
        maxValue = max(maxValue, val);
    }
    
    long long ans = 0;  // 记录最大总价值
    for(int j = maxValue; j >= 1; j--){  // 从大到小遍历
        if(counter[j] > 0){  // 该价值有商品
            long long take = min(counter[j], X);  // 最多取X件
            ans = max(ans, (long long)take * j);  // 更新最大值
        }
    }
    
    cout << ans << endl;
    return 0;
}

/*题后总结：
1.审题，题目说的是“最多”x件，不是说<x件的就不纳入考虑当中；
2.注意边界条件，已知几个变量的范围是：N 最大 10^5
X 最大 10^5
A_i 最大 N (10^5)，则当maxValue*X=10^5**2=10^10时，超过了int最大范围10^9
3.max()和min()函数，里面比较的两个数必须是同类型，比如都是int或者都是longlong*/
