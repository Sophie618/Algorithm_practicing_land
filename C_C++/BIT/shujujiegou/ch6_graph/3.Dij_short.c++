#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>
using namespace std;

const int MAXN = 30; // 最多26个字母
const int INF = INT_MAX;

struct Edge
{
    int to;     // 目标节点
    int weight; // 权重
};

vector<Edge> graph[MAXN]; // 邻接表
int dist[MAXN];           // 距离数组
bool visited[MAXN];       // 是否已确定最短路径
int n;                    // 节点数量
char nodes[MAXN];         // 存储节点字母（用于排序输出）
int nodeCount = 0;

// 字符转索引
int charToIndex(char c)
{
    return c - 'a';
}

// 索引转字符
char indexToChar(int idx)
{
    return 'a' + idx;
}

// Dijkstra 算法
void dijkstra(char start)
{
    // 初始化
    for (int i = 0; i < MAXN; i++)
    {
        dist[i] = INF;
        visited[i] = false;
    }

    int startIdx = charToIndex(start);
    dist[startIdx] = 0;

    // 优先队列：<距离, 节点索引>
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, startIdx});

    while (!pq.empty())
    {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();

        // 如果已经访问过，跳过
        if (visited[u])
            continue;
        visited[u] = true;

        // 松弛操作：更新邻接点的距离
        for (const Edge &e : graph[u])
        {
            int v = e.to;
            int w = e.weight;

            if (!visited[v] && dist[u] != INF && dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
}

int main()
{
    int e;
    char start;

    // 读取输入：n,e,start
    scanf("%d,%d,%c", &n, &e, &start);

    // 读取边
    for (int i = 0; i < e; i++)
    {
        char u, v;
        int w;
        scanf(" <%c,%c,%d>", &u, &v, &w);

        int uIdx = charToIndex(u);
        int vIdx = charToIndex(v);

        // 添加边（有向图）
        graph[uIdx].push_back({vIdx, w});
    }

    // 收集所有出现过的节点
    bool appeared[MAXN] = {false};
    appeared[charToIndex(start)] = true;

    for (int i = 0; i < MAXN; i++)
    {
        for (const Edge &e : graph[i])
        {
            appeared[i] = true;
            appeared[e.to] = true;
        }
    }

    for (int i = 0; i < MAXN; i++)
    {
        if (appeared[i])
        {
            nodes[nodeCount++] = indexToChar(i);
        }
    }

    // 排序节点（按字母顺序）
    sort(nodes, nodes + nodeCount);

    // 执行 Dijkstra
    dijkstra(start);

    // 输出结果（按字母升序）
    for (int i = 0; i < nodeCount; i++)
    {
        char node = nodes[i];
        int idx = charToIndex(node);
        cout << node << ":" << dist[idx] << endl;
    }

    return 0;
}