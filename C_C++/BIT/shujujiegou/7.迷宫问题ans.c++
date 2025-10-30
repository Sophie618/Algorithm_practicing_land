#include <bits/stdc++.h> 
using namespace std; 
#define MAXN 105 
 
int n; 
int maze[MAXN][MAXN]; 
int visited[MAXN][MAXN]; 
int path[MAXN * MAXN][2];  // 保存路径 [行, 列] 
int pathLen = 0; 
 
// 方向数组：南(下)、东(右)、北(上)、西(左) 
int dr[] = {1, 0, -1, 0}; 
int dc[] = {0, 1, 0, -1}; 
 
// DFS 搜索函数 
// r, c: 当前位置 (0-based) 
// 返回: true 表示找到路径，false 表示无解 
bool dfs(int r, int c) { 
    // 标记当前位置已访问，加入路径 
    visited[r][c] = 1; 
    path[pathLen][0] = r; 
    path[pathLen][1] = c; 
    pathLen++; 
     
    // 到达终点 
    if (r == n-1 && c == n-1) { 
        return true; 
    } 
     
    // 按四个方向尝试 
    for (int i = 0; i < 4; i++) { 
        int nr = r + dr[i]; 
        int nc = c + dc[i]; 
         
        // 检查边界、墙、是否访问过 
        if (nr >= 0 && nr < n && nc >= 0 && nc < n &&  
            maze[nr][nc] == 0 && !visited[nr][nc]) { 
             
            if (dfs(nr, nc)) { 
                return true;  // 找到解，返回 
            } 
        } 
    } 
     
    // 回溯：所有方向都失败，退出当前格 
    pathLen--; 
    visited[r][c] = 0; 
    return false; 
} 
 
int main() { 
    // 读取迷宫尺寸和数据 
     cin>>n>>n;  // 题目给两个数，都是 n 
     
    for (int i = 0; i < n; i++) { 
        for (int j = 0; j < n; j++) { 
            cin>>maze[i][j]; 
        } 
    } 
     
    // 快速剪枝：起点或终点是墙 
    if (maze[0][0] == 1 || maze[n-1][n-1] == 1) { 
        cout<<"There is no solution!"<<endl; 
        return 0; 
    } 
     
    // 开始 DFS 搜索 
    if (dfs(0, 0)) { 
        // 输出路径 (转换为 1-based) 
        for (int i = 0; i < pathLen; i++) { 
            cout<<"<"<<path[i][0] + 1<<","<<path[i][1] + 1<<"> "; 
        } 
        cout<<endl; 
    } else { 
        cout<<"There is no solution!"<<endl; 
    } 
     
    return 0; 
}  