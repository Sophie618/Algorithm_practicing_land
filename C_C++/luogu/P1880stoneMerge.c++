// https: // www.luogu.com.cn/problem/P1880
// 隐藏的坑点在于圆形摆放第一个和第N个分别在石头序列的最两端，所以使用两倍数组断环成链，复制一遍是为了构造出所有可能的环形展开情况（滑动窗口）。
// 动态规划DP

/*1.确定含义：dp[i][j]表示从第i堆合并到第j堆石子所需要的最小得分
2.确定递推公式（状态转移方程）：dp[i][j] = min{dp[i][k] + dp[k+1][j] + sum(i,j)} (i<=k<j)
3.初始化：dp[i][i] = 0&MAX
4.确定遍历顺序：
  外层优先遍历区间长度len从2到n，内层遍历起点i从1到n-len+1，终点j=i+len-1，因为后面待求的长度依赖于前面的短的长度
5.举例推导/打印出dp数组看看是否和自己想的一样*/

#include<iostream>
#include<algorithm>
#include<cstring>

using namespace std;

int dpmax[205][205], dpmin[205][205];//存最大得分和最小得分
int a[205];//存石子数组（2倍数组
int s[205];//前缀和数组，便于O(1)
const int MAX = 0x3f3f3f3f; // 不可取最大整数INT_MAX，加1就直接溢出变成负数造成比较错误；0x3f3f3f3f刚刚好，两个 0x3f3f3f3f 相加不会溢出变成负数
/*memset 是按字节 (Byte) 填充的，它不管你传的是多少，它只取最低的 8 位（1个字节）。
MAX 是 0x3f3f3f3f，它的最低字节是 0x3f。 memset 会把每个字节都填成 0x3f。 因为 int 有 4 个字节，所以填完后，正好变成了 0x3f 0x3f 0x3f 0x3f。 结论：你的写法完全有效！以后考场上就这么写没问题。如果为了书写简单，通常高手会写 memset(dpmin, 0x3f, sizeof(dpmin))，效果一模一样。
*/
    int main()
{
    int N;//N堆石子
    cin >> N;
    for (int i = 1; i <= N;i++){
        cin >> a[i];
        a[i + N] = a[i];//复制到后面N个
    }
    s[0] = 0;
    for(int i = 1; i <= 2 * N; i++){
        s[i] = s[i - 1] + a[i];//计算前缀和
    }
    // 初始化dp数组（对角线为0）
    memset(dpmax, 0, sizeof(dpmax));//变量名，初始化值，存储大小
    memset(dpmin, MAX, sizeof(dpmin));

    for (int i = 1; i <= 2 * N;i++){
        dpmax[i][i] = 0;
        dpmin[i][i] = 0;
    }

    //开始循环
    for (int len = 2; len <= N;len++){
        for (int i = 1; i <= 2*N-len+1;i++){//因为j不能超过2N，所以i必须留足空间，至少要在2N-len的位置
            int j = i + len - 1;//得到该枚举长度下的终点
            for (int k = i; k < j; k++)
            {
                dpmax[i][j] = max(dpmax[i][j], dpmax[i][k] + dpmax[k + 1][j]+s[j]-s[i-1]);//注意这里必须是s[i-1]否则i自己本身的重量漏算了！
                dpmin[i][j] = min(dpmin[i][j], dpmin[i][k] + dpmin[k + 1][j]+ s[j] - s[i-1]);
            }
        } 
    }
    int Max = 0, Min = MAX;
    //取出最终结果
    for (int i = 1; i <= N;i++){
        Max = max(Max, dpmax[i][i + N - 1]);
        Min = min(Min, dpmin[i][i + N - 1]);
    }
    cout << Min << endl
         << Max << endl;

    return 0;
}