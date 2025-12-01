#include<bits/stdc++.h>
using namespace std;

int result = 0;//放置方式的数量
vector<pair<int, int>> queens;//存储皇后的xy坐标位置,全局变量无需传参

bool isFullControl(int n){
    bool controlled[10][10] = {false};//初始化默认都没有被控制
    //根据皇后位置坐标依次去标记被控制的范围
    for(auto &q:queens){//&q是引用，避免拷贝的开销
        int r=q.first;
        int c = q.second;

        //标记同行
        for (int i = 0; i < n;i++){
            controlled[r][i] = true;
        }
        //标记同列
        for (int i = 0; i < n; i++)
        {
            controlled[i][c] = true;
        }
        //标记对角线
        for (int i = 0; i < n;i++){
            for (int j = 0; j < n;j++){
                if(abs(i-r)==abs(j-c)){//相当于斜率为1，毕竟就是对角线
                    controlled[i][j] = true;
                }
            }
        }
    }

    //标记完成以后，检查整个棋盘被控制情况
    for (int i = 0; i < n;i++){
        for (int j = 0; j < n;j++){
            if(controlled[i][j]==false){
                return false;//只要有一个没有被控制，就没有全部控制
            }
        }
    }
    return true;//全部都被控制，则全部控制了
}

bool isSafe(int row,int col){
    for(auto &q:queens){
        int r = q.first;//q的第一个元素是行号
        int c = q.second;//q的第二个元素是列号
        //检查同行同列对角线情况
        if(col==c||row==r||abs(r-row)==abs(c-col)){
            return false;//与其他皇后位置冲突了
        }
    }
    return true;//没有冲突可以放置
}

void backtrack(int n,int m,int startRow){
    //结束条件：已经放置了m个皇后，检查是否控制了整个棋盘，如果是则放置方式多一种
    if(queens.size()==m){//当前队列中已经有了目标个数的皇后
        if (isFullControl(n))
        {
            result++;//全部控制了，说明方法数多一种
        }
        return;//返回
    }

    //从startrow开始尝试放置皇后
    for (int r = startRow; r < n;r++){
        for (int c = 0; c < n;c++){
            if(isSafe(r,c)){//如果在r行c列放置不会与其他皇后位置冲突才考虑放置和回溯
                queens.push_back({r, c});//选择该位置放置，加入路径
                backtrack(n, m, r);//递归，基于当前选择探索余下可能，并且同行继续尝试
                queens.pop_back();//回溯，撤销当前位置放置
            }
        }
    }
}

int main(){
    int n,m;
    cin >> n>>m;//输入棋盘大小n和皇后数量m
    backtrack(n, m, 0);//从零开始搜索和递归
    cout << result << endl;
    return 0;
}

/*题后总结：
1.根据题意翻译成数学关系，找好规律很重要
2.回溯的递归一定记得检查是否有返回
3.回溯简单但是要记得检查一些边界条件
4.一般一开始就先检查是否到达结束条件了，然后回溯的函数参数一般会包含一个与上一层递归相关的累计状态变量，比如当前行号，当前重量等等
5.然后一般来说结果变量和位置记录数组都写成全局变量就可以不用到处传参了*/