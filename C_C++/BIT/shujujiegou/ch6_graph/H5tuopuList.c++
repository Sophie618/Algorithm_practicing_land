#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <algorithm>
#include <climits>
using namespace std;

struct Edge
{
    int to, weight;
};

struct InputEdge
{
    int from, to, weight;
};

vector<string> split(const string &s, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

bool topologicalSort(int n, vector<vector<Edge>> &graph, vector<int> &inDegree, vector<int> &topoOrder)
{
    vector<int> tempInDegree = inDegree;
    priority_queue<int, vector<int>, greater<int>> pq; // 小顶堆，保证按序号从小到大

    for (int i = 0; i < n; i++)
    {
        if (tempInDegree[i] == 0)
        {
            pq.push(i);
        }
    }

    while (!pq.empty())
    {
        int u = pq.top();
        pq.pop();
        topoOrder.push_back(u);

        for (auto &edge : graph[u])
        {
            int v = edge.to;
            tempInDegree[v]--;
            if (tempInDegree[v] == 0)
            {
                pq.push(v);
            }
        }
    }

    return topoOrder.size() == n;
}

void calculateVeVl(int n, vector<vector<Edge>> &graph, vector<int> &topoOrder,
                   vector<int> &ve, vector<int> &vl)
{
    // 计算ve（最早发生时间）
    ve.assign(n, 0);
    for (int u : topoOrder)
    {
        for (auto &edge : graph[u])
        {
            int v = edge.to;
            int w = edge.weight;
            ve[v] = max(ve[v], ve[u] + w);
        }
    }

    // 计算vl（最迟发生时间）
    vl.assign(n, INT_MAX);
    vl[topoOrder.back()] = ve[topoOrder.back()];

    for (int i = topoOrder.size() - 1; i >= 0; i--)
    {
        int u = topoOrder[i];
        for (auto &edge : graph[u])
        {
            int v = edge.to;
            int w = edge.weight;
            vl[u] = min(vl[u], vl[v] - w);
        }
    }
}

void findAllPaths(int u, int end, vector<vector<Edge>> &graph,
                  vector<int> &ve, vector<int> &vl,
                  vector<int> &path, vector<vector<int>> &allPaths)
{
    path.push_back(u);

    if (u == end)
    {
        allPaths.push_back(path);
        path.pop_back();
        return;
    }

    for (auto &edge : graph[u])
    {
        int v = edge.to;
        int w = edge.weight;
        // 判断是否为关键活动
        int e = ve[u];     // 活动最早开始时间
        int l = vl[v] - w; // 活动最迟开始时间
        if (e == l)
        {
            findAllPaths(v, end, graph, ve, vl, path, allPaths);
        }
    }

    path.pop_back();
}

int main()
{
    string line;

    // 读取第一行
    getline(cin, line);
    vector<string> parts = split(line, ',');
    int n = stoi(parts[0]);
    int m = stoi(parts[1]);

    // 读取节点名称
    getline(cin, line);
    vector<string> nodes = split(line, ',');

    // 读取边
    getline(cin, line);
    vector<InputEdge> inputEdges;
    vector<vector<Edge>> graph(n);
    vector<int> inDegree(n, 0);

    // 解析边
    line = line.substr(1, line.length() - 2); // 去掉首尾的 '<' 和 '>'
    size_t pos = 0;
    while (pos < line.length())
    {
        size_t start = pos;
        size_t end = line.find(">,<", pos);
        if (end == string::npos)
            end = line.length();

        string edgeStr = line.substr(start, end - start);
        if (edgeStr[0] == '<')
            edgeStr = edgeStr.substr(1);
        if (edgeStr.back() == '>')
            edgeStr.pop_back();

        vector<string> edgeParts = split(edgeStr, ',');
        int from = stoi(edgeParts[0]);
        int to = stoi(edgeParts[1]);
        int weight = stoi(edgeParts[2]);

        inputEdges.push_back({from, to, weight});
        graph[from].push_back({to, weight});
        inDegree[to]++;

        pos = end + 3;
    }

    // 拓扑排序
    vector<int> topoOrder;
    if (!topologicalSort(n, graph, inDegree, topoOrder))
    {
        cout << "NO TOPOLOGICAL PATH" << endl;
        return 0;
    }

    // 输出拓扑排序
    for (int i = 0; i < topoOrder.size(); i++)
    {
        cout << nodes[topoOrder[i]];
        if (i < topoOrder.size() - 1)
            cout << "-";
    }
    cout << endl;

    // 计算ve和vl
    vector<int> ve, vl;
    calculateVeVl(n, graph, topoOrder, ve, vl);

    // 找出所有关键路径
    vector<vector<int>> allPaths;
    vector<int> path;
    int start = topoOrder[0];
    int end = topoOrder.back();
    findAllPaths(start, end, graph, ve, vl, path, allPaths);

    // 排序并输出关键路径
    sort(allPaths.begin(), allPaths.end());

    for (auto &p : allPaths)
    {
        for (int i = 0; i < p.size(); i++)
        {
            cout << nodes[p[i]];
            if (i < p.size() - 1)
                cout << "-";
        }
        cout << endl;
    }

    return 0;
}