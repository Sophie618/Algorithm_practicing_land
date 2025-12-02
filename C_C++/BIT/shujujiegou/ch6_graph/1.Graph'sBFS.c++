#include<bits/stdc++.h>
using namespace std;

const int MAXV = 100;

struct ArcNode{
    int adjvex;//存储的邻接点的下标索引
    ArcNode *next;//指向下一个邻接点的指针
};//每次定义完结构体一定要记得打一个分号！！！

struct VNode{
    char data;//顶点数据
    ArcNode *first;//边链表的头指针，指向第一个邻接点
};

struct ALGraph{
    VNode vertices[MAXV];//顶点数组，本身就是结构体，包含顶点及其邻接表
    int vexnum;//顶点数
};

void CreateGraph(ALGraph &G){//加取址符号，为了可以真正修改内部数据；平时也推荐加，因为可以减少开销
    G.vexnum = 0;//顶点数vertex number

    //开始输入，读取顶点
    string vertex;
    while(cin>>vertex&&vertex!="*"){
        G.vertices[G.vexnum].data = vertex[0];//填入顶点数据
        G.vertices[G.vexnum].first = nullptr;//初始化邻接点的头指针
        G.vexnum++;//索引往后挪一位用来读取下一个顶点数据
    }

    //顶点读完开始读邻接表
    string edge;
    while(cin>>edge &&edge!="-1,-1"){
        int pos = edge.find(',');//找到逗号的位置下标
        int v1 = edge[0] - '0';//第一个顶点的下标索引,字符转整数
        int v2 = edge[pos + 1] - '0';//第二个顶点的下标索引

        //添加无向图的两条边
        //v1->v2
        ArcNode *p1 = new ArcNode;
        p1->adjvex = v2;
        p1->next = G.vertices[v1].first;
        G.vertices[v1].first = p1;

        // v2->v1
        ArcNode *p2 = new ArcNode;
        p2->adjvex = v1;
        p2->next = G.vertices[v2].first;
        G.vertices[v2].first = p2;
    }
}

//打印邻接表
void PrintALGraph(ALGraph &G){
    cout << "the ALGraph is" << endl;
    for (int i = 0; i <G.vexnum;i++){//一定要记得vexnum属于G之下，调用需要调用G
        cout << G.vertices[i].data;
        ArcNode *q = G.vertices[i].first;//定义一个指向邻节表头指针的指针来遍历邻接表
        while(q!=nullptr){
            cout << ' ' << q->adjvex;//每次输出一个数据前需要先输出一个空格
            q = q->next;//更新到下一个节点
        }
        cout << endl;//每次打印完一个节点及其邻接表的值需要换行！！
    }
}

//广度优先遍历：使用队列用于记录遍历过的节点，然后依次取出直至队列为空
void BFS(ALGraph &G){
    bool visited[MAXV] = {false};//初始化访问数组，默认均未访问
    queue<int> q;//先进先出
    cout << "the Breadth-First-Seacrh list:";
    for (int i = 0; i<G. vexnum;i++){
        if(!visited[i]){//先判断是否确实没有访问过
            visited[i] = true;//标记为访问
            cout << G.vertices[i].data;//输出该节点数据
            q.push(i);//节点下标入队

            while(!q.empty()){
                int v=q.front();//取出队首元素
                q.pop();//出队
                ArcNode *p = G.vertices[v].first;//取出该节点邻节点头指针
                while(p!=nullptr){
                    if(!visited[p->adjvex]){
                        visited[p->adjvex] = true;//标记为访问
                        cout << G.vertices[p->adjvex].data;
                        q.push(p->adjvex);//把这个下标给入队
                    }//如果q为空，说明当前节点的全部邻接点都被访问过了
                    p = p->next;
                }
            }
        }
    }
    cout << endl;//遍历完了不要忘了输出最后一个换行哦
}

int main()
{
    ALGraph G;
    CreateGraph(G);
    PrintALGraph(G);
    BFS(G);
    return 0;
}