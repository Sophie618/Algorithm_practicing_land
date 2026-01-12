#include<bits/stdc++.h>
using namespace std;

typedef struct HNode{
    int weight;
    HNode *lchild, *rchild;
    HNode(int w) : weight(w), lchild(nullptr), rchild(nullptr){}//构造函数初始化节点权重和子节点指针
} HNode, *HuffmanTree;

struct compare{
    bool operator()(HuffmanTree a,HuffmanTree b){
        return a->weight > b->weight;//a的权重比b大返回true表明b的优先级更高
    }
};//小根堆（优先队列）的比较函数，重新定义以小值优先

HuffmanTree buildHuffman(const vector<int>&weights){
    //定义一个最小根堆变量
    //三个模板参数：存储元素类型，底层容器类型，比较函数
    priority_queue<HuffmanTree, vector<HuffmanTree>, compare> min_heap;//树节点类型的数组，按照自定义的compare排序，因为priority_queue默认是大根堆

    //所有原始权重入堆，且均为叶子节点
    for(int w:weights){
        min_heap.push(new HNode(w));//每个权重入堆
    }

    //总共n个节点要进行n-1次合并
    for (int i = 0; i < weights.size() - 1;i++){
        //取堆中前两个节点即最小权重节点
        HuffmanTree left = min_heap.top();
        min_heap.pop();
        HuffmanTree right = min_heap.top();
        min_heap.pop();

        HuffmanTree parent = new HNode(left->weight + right->weight);//权重是左右节点之和
        parent->lchild = left;
        parent->rchild = right;
        min_heap.push(parent);
    }
        return min_heap.top(); // 堆中最后一个节点就是根节点
}

int calWPL(HuffmanTree root,int depth){
    if(root==nullptr){
        return 0;//空节点就直接返回
    }

    if(root->lchild==nullptr&&root->rchild==nullptr){
        return depth * root->weight;//确认是叶子节点就返回路径深度*权重
    }
    else{
        return calWPL(root->lchild, depth + 1) + calWPL(root->rchild, depth + 1);//递归计算左右子树的WPL之和,这一行非常的精妙啊（其实换成1就是统计叶子节点的数量
    }
}

void freeHuffman(HuffmanTree root){
    if(root==nullptr){
        return;//空节点无需释放
    }
    freeHuffman(root->lchild);//递归释放左子树
    freeHuffman(root->rchild);//递归释放右子树
    delete root;//释放当前节点内存
}

int main()
{
    int n,weight,i=0;
    cin >> n;
    vector<int> weights(n);//用哈希数组来存贮权重，便于求最小的前两个值
    for(i=0;i<n;i++){//依次输入每个节点的权重
        cin >> weights[i];//输入节点权重
        }
            
    //构建Huffman树
    HuffmanTree tree = buildHuffman(weights); // 找到返回的已经挂载哈夫曼树的根节点

    //计算WPL
    int wpl = calWPL(tree, 0);
    cout << "WPL=" << wpl << endl;
    freeHuffman(tree);//释放哈夫曼树内存
    return 0;
}
/*开始前先确定需要哪些函数：
1.构建huffman树
2.计算带权路径长度WPL
3.(可选)释放Huffman树内存
*/