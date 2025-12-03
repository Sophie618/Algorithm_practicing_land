#include<bits/stdc++.h>
using namespace std;

int const MAXN = 1005;
int const MAXM = 3005;
int edgeCount = 0; // 现在已连接0条边
int totalCost = 0; // 总成本初值为0

//定义每条边
struct Edge{
    int u, v;//两个顶点
    int cost;//每条路径的成本
};//每次结构体定义都容易忘掉！

int parent[MAXN];//用来存储每个顶点对应的集合根节点
Edge Edges[MAXM];//存储边的相关信息,总共M条

void InitEdges(int N,int M){
    for (int i = 0; i < M;i++){
        cin >> Edges[i].u >> Edges[i].v >> Edges[i].cost;
    }
    for (int i = 1; i <= N;i++){
        parent[i] = i;//让每个顶点的父节点先初始化为它本身
    }
}

//寻找根节点，同时实现路径压缩
int find(int x){
    if(parent[x]!=x){//如果父节点不等于它本身，说明还没找到根节点
        parent[x] = find(parent[x]);//把这个节点的父节点更新到根节点上，进行路径压缩
    }
    return parent[x];//需要返回根节点的值来进行合并的判断依据
}

//合并两个集合同时判断是否是同一个集合下的两个节点
bool Union(int x,int y){
    int rootX = find(x);
    int rootY = find(y);

    if(rootX==rootY){
        return false;//在一个集合内部不需要合并同时要避免选择这条路
    }
    parent[rootX] = rootY;//这里是为了让两个根合并！很容易错！
    return true;//合并且该路径可选
}

//Kruskal算法，贪心选择同时利用并查集实现查找和合并
void Kruskal(int N,int M){
    for (int i = 0; i < M;i++){
        if(edgeCount==N-1){
            break;//已经完全连同了，直接结束后续遍历
        }
        int rootX = find(Edges[i].u);
        int rootY = find(Edges[i].v);

        if(Union(rootX,rootY))
        { // 在合并的同时判断二者是否不是同一个集合内的元素，满足就说明可以修这条路
            totalCost += Edges[i].cost;//Union内部已经把他们合并了，此时只需要求出修这条路需要的成本
            edgeCount++;//记得更新已经连接的边数
        }
    }
    if(edgeCount==N-1){//结束所有的边的遍历以后为什么还有一次判断呢？因为有可能会出现边不够的情况，再次进行分类
        cout << totalCost << endl;
    }
    else{
        cout << -1 << endl;
    }
}

//查找根节点，同时完成路径压缩

int main(){
    int N, M;
    cin >> N >> M;//输入楼的数量（用于检查结束条件），网络线路条数
    InitEdges(N,M);//把边的信息和其根节点先存储了
    //按照成本由低到高排序，便于贪心
    sort(Edges, Edges + M, [](const Edge &a, const Edge &b) { // 接收的是一个Edges数组的首地址和其后面M个元素的地址（左闭右开），其中是一个比较函数即lambda表达式
        return a.cost < b.cost;
    });

    Kruskal(N,M);
    return 0;
}

/*总流程：
1.按照成本对所有路线进行排序
2.使用所谓Kruskal算法-贪心：优先选择不会成环的路径，并且利用并查集找是不是成环（查是不是在同一个集合中，如果不在就可以直接合并到一个根节点下）
3.为了使边数最少和成本最小，要避免成环；排序是为了优先选最便宜的
4.重要的是意识到不必要构造图或者表，重要的是记录所需最低成本最后统一输出即可*/