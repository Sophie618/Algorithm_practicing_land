#include <iostream>
#include <algorithm>
using namespace std;

const int MAXN = 1005;
const int MAXM = 3005;

// 边的结构
struct Edge
{
    int u, v; // 两个顶点
    int cost; // 成本

    bool operator<(const Edge &other) const
    {
        return cost < other.cost;
    }
};

// 并查集
int parent[MAXN];

// 初始化并查集
void initUnionFind(int n)
{
    for (int i = 1; i <= n; i++)
    {
        parent[i] = i;
    }
}

// 查找根节点（带路径压缩）
int find(int x)
{
    if (parent[x] != x)
    {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

// 合并两个集合
bool unite(int x, int y)
{
    int rootX = find(x);
    int rootY = find(y);

    if (rootX == rootY)
    {
        return false; // 已经在同一集合，会形成环
    }

    parent[rootX] = rootY;
    return true;
}

int main()
{
    int N, M;
    cin >> N >> M;

    Edge edges[MAXM];

    // 读入所有边
    for (int i = 0; i < M; i++)
    {
        cin >> edges[i].u >> edges[i].v >> edges[i].cost;
    }

    // 按成本从小到大排序
    sort(edges, edges + M);

    // 初始化并查集
    initUnionFind(N);

    int totalCost = 0;
    int edgeCount = 0;

    // Kruskal算法：依次选择最小成本的边
    for (int i = 0; i < M; i++)
    {
        if (unite(edges[i].u, edges[i].v))
        {
            totalCost += edges[i].cost;
            edgeCount++;

            // 最小生成树有 N-1 条边
            if (edgeCount == N - 1)
            {
                break;
            }
        }
    }

    // 检查是否所有楼宇都连通
    if (edgeCount == N - 1)
    {
        cout << totalCost << endl;
    }
    else
    {
        cout << -1 << endl;
    }

    return 0;
}