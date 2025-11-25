#include <bits/stdc++.h>
using namespace std;

int res;//全局变量result的简称

void calculate(int n,int m){
    if (n < m + 1)
    { // 总结点数小于要求计算层数
        res += 0;//写加号是为了方便递归
        return;//结束递归返回上一层
    }
    else if (n == m + 1)
    { // 总结点数恰好满足每层一个节点
        res += 1;
        return;
    }
    else//这种情况下n一定大于等于m+1，也就是说至少有第一层斜边，res初始就为1
    {
        res += 1;//当前层已经有一个节点了噢
        int layer = 1;//直接从第一层开始算因为第0层已经有了
        n = n - 1 - m;//去除第一层的单列节点
        if(n<=0||layer>m){
            return;
        }
        int need = pow(2, layer) - 1;//要填满当前层还需要的节点数
        while(n>need){//先遍历到n<当前层填满所需节点数时
            n -= need;
            res += pow(2, layer - 1);
            layer++;//layer++的位置很重要，一定要先把res改了再++
            need = pow(2, layer) - 1;
            if (layer > m) // layer++完会导致其与m相等，但是还差一层才填满，不能等于
            {//没有这个停止条件假如n真的特别多的情况下，将会无限加层，但是已经超所需层数范围了          
                return; // 检查是否已经把m层完全填满，若已经填满，则不需要再管剩下的其他节点了
            }
        }
        if(n==need){//如果正好能放完一层
            res += pow(2, layer - 1);
            return;
        }
        else{//递归计算剩余节点数最多能产生多少贡献
            calculate(n, layer-1);
        }
    }
}


int main(){
    int T, n, m;//n个节点第m层，m层从0开始计数
    cin >> T;
    while(T--){//输入T组数据
        res = 0;//一定记得每轮计算前先把计数器清零
        cin >> n >> m;//会根据空白自动分割
        calculate(n, m);//直接通过全局变量传递过来
        cout << res << endl;
    }
    return 0;//main函数记得写返回值
}