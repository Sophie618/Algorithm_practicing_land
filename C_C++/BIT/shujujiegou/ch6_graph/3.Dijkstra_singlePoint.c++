#include<bits/stdc++.h>
using namespace std;

const int MAXN = 30;//最多26个字母，不会超过30
const int INF = INT_MAX;//表示int型表示数上限，无穷大，路径不存在或无法到达

struct Edge{
    int to;//下一点索引
    int length;//路径长度
};

int n;//总节点数
char nodes[MAXN];//存储节点字母便于排序输出
int nodeCount = 0;//计数器记录实际节点数量
vector<Edge> graph[MAXN];//图数组，也是邻接表形式，下标是当前节点索引，每个元素都是一个vector，存储多个Edge结构体数组，表示到达的目标节点和路径长度
int dist[MAXN];//距离数组，下标是对应节点索引，值是从源点到该节点的最短距离，不能直接={INF}初始化，因为这样只能把第一个元素初始化为INF，其他元素是0
bool visited[MAXN]={false};//记录是否找到当前节点到源点的最短距离，这里可以把所有的元素初始化为false

void dijkstra(int start){
    //初始化距离数组
    for (int i = 0; i < MAXN;i++){
        dist[i] = INF;
    }
    // 定义一个优先队列完成每次最小距离节点出队
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq; // 优先队列：<距离, 节点索引>标准定义是priority_queue<类型，底层容器，比较规则>利用优先队列取出最小距离元素
    dist[start] = 0;//初始化源点到自己的距离
    pq.push({0, start}); // pair的比较规则是先比first再比second，所以把距离放在pair的第一个位置

    while(!pq.empty()){
        int d = pq.top().first;//得到与该节点距离最小的距离
        int u = pq.top().second;//得到该节点索引
        pq.pop();//出队
        if(visited[u]){
            continue;//如果已经找到了最短距离直接下一个
        }
        visited[u] = true;//标记为找到最短距离,必须放在if判断之后，否则陷入死循环
        
        for(const Edge&e:graph[u]){//取所有节点是相当于把所有目前已经有的距离利用优先队列来找出最小距离的节点
            int v = e.to;//邻接点索引
            int l = e.length;//路径长度

            if (!visited[v] && dist[u] != INF && dist[u] + l < dist[v])
            {
                dist[v] = dist[u] + l;
                pq.push({dist[v], v}); // 更新后的最短距离和节点入队
            }
        }
    }
}

int main(){
    int e;//边数
    char start;//源点
    scanf("%d,%d,%c", &n, &e, &start);
    char u, v;//起始和终点<u,v>
    int l;//路径长度
    //依次读入e条边的数据，存入图数组中
    for (int i = 0; i < e;i++){
        scanf(" <%c,%c,%d>", &u, &v ,&l);
        int uid = u - 'a';
        int vid = v - 'a';
        graph[uid].push_back({vid, l}); // 将邻接表内容依次存入数组
    }
    bool appeared[MAXN] = {false};//记录已经出现过的节点
    appeared[start - 'a'] = true;//源点出现
    for (int i = 0; i < MAXN;i++){//记录出现的每一个节点
        for(const Edge &e:graph[i]){
            appeared[i] = true;//该节点自己出现过
            appeared[e.to] = true;
        }
    }
    for (int i = 0; i < MAXN;i++){//这里如果上界为n会漏掉后面的字母，比如n等于2，有a，z两个节点，这样的话，只能遍历到b就结束了
        if(appeared[i]){
            nodes[nodeCount++] = 'a' + i;//存储字符
        }
    }
    sort(nodes, nodes + nodeCount);//将节点从小到大升序排序方便最后输出结果
    dijkstra(start - 'a');//迪杰斯特拉算法
    for (int i = 0; i < nodeCount;i++){
        cout << nodes[i] << ':' << dist[nodes[i] - 'a'] << endl;
    }
    return 0;
}