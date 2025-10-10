#include<bits/stdc++.h>
using namespace std;
#define MAXN 105

int n;
int maze[MAXN][MAXN];
int visited[MAXN][MAXN];
int pathlong=0;
int path[MAXN*MAXN][2];
int dx[] = {1, 0, -1, 0};//下右上左（行）
int dy[] = {0, 1, 0, -1};//下右上左（列）

bool dfs(int x, int y) {//(x,y)当前位置
    visited[x][y] = 1;
    path[pathlong][0]=x;
    path[pathlong][1]=y;
    pathlong++;
    if(x==n-1&&y==n-1) return true;
    //四个方向试探
    for(int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        //检测边界，墙
        if(nx>=0&&nx<n&&ny>=0&&ny<n&&maze[nx][ny]==0&&visited[nx][ny]==0){
        if(dfs(nx,ny)) {
            return true;
        }
    } 
    }

    //行不通
    visited[x][y]=0;
    pathlong--;
    return false;
}

int main(){
    cin>>n>>n;
    for(int j=0;j<n;j++){//0-base
        for(int i=0;i<n;i++){
            cin>>maze[j][i];
        }
    }
    //快速剪枝
    if(maze[0][0]==1||maze[n-1][n-1]==1){
        cout<<"There is no solution!"<<endl;
    }
    else if(dfs(0,0)){
        for(int i=0;i<pathlong;i++){
            cout<<"<"<<path[i][0]+1<<","<<path[i][1]+1<<"> ";
        }
        cout<<endl;
    }
    else{
        cout<<"There is no solution!"<<endl;
    }
}

/*基本思路：
1. 迷宫问题是一个经典的搜索问题，可以使用深度优先搜索（DFS）来解决。
2. 在搜索过程中，需要记录当前位置和路径，以便在找到解时输出路径。
3. 需要检测边界和墙，以及是否访问过当前位置。
4. 需要回溯，以便在行不通时返回上一个位置。

易错点：
1.对于复杂的矩阵问题，尽量直接使用row，col来表示行列
2.使用坐标x，y来表示行列容易出错，因为这更容易把x对应到行，y对应到列（通常的直角坐标系）
3.但是实际上，maze[x][y]是先列后行,与直觉不一致   
4.标准的ai代码见7.迷宫问题ans.c++
*/