#include<bits/stdc++.h>
using namespace std;

int c, n;//分别是上限和数量
vector<int> w;//用来存储重量的数组
vector<int> bestSolution;//存储最佳解法
vector<int> currentSolution;//存储当前解法
int bestWeight = 0;//最大值，作为全局变量可以传入任何函数中并且修改其值

bool isBestSolution(const vector<int> &sol1, const vector<int> &sol2){
    for (int i = 0; i < n;i++){
        if(sol1[i]>sol2[i]){
            return true;
        }
        if(sol1[i]<sol2[i]){//不能取等因为还不能确定等于的那个下标的后面是否还会出现前者更优的情况,!!!不能写else if，否则相等的时候就直接返回了，但是后面的还没比较呢！
            return false;
        }
        return false;//两个解完全相同的情况
    }
}

int remainingWeight(int index){
    int res = 0;
    for (int i = index; i < n;i++){
        res += w[i];
    }
    return res;
}

// 回溯算法
void backtrack(int index, int currentWeight)//开始回溯的下标和当前总重量
{
    if(index==n){//结束条件是已经遍历到最后一个节点，需要更新最大重量和最佳解法
        //首先判断最大重量
        if(currentWeight>bestWeight){
            bestWeight = currentWeight;
            bestSolution = currentSolution;//容易漏掉！不要只更新重量，最后还要输出最佳解法的！
        }
        else if(currentWeight==bestWeight){
            //若最大重量相等，再比较最优解（因为相等，不需要再更新最大重量了）
            if(isBestSolution(currentSolution,bestSolution)){
                bestSolution = currentSolution;
            }
        }
    }
    
    if(currentWeight+remainingWeight(index)<=bestWeight){
        return;//如果当前重量加上剩余数量仍然不能超过最大重量，直接剪枝，避免多余的复杂度
    }

    //优先选择当前物品，因为最优解要求二进制表示尽量大
    if(currentWeight+w[index]<=c){
        currentSolution[index] = 1;//将对应物品下标标记为1
        backtrack(index + 1, currentWeight + w[index]);//在选择当前位置的情况下继续往后试探
        currentSolution[index] = 0;//回溯，撤销上一步的选择，这个是全局亮点！！！
    }
    // else{//已经大于c【其实不要这个else也一样而且上面已经回溯完了】
    //     currentSolution[index] = 0;
        backtrack(index + 1, currentWeight);//不选择当前位置的物品，继续往下试探
    // }
}

int main()
{
    cin >> c >> n;
    //得到数量后第一时间先修改相关数据容器的大小
    w.resize(n);
    bestSolution.resize(n,0);
    currentSolution.resize(n, 0);
    for (int i = 0; i < n;i++){
        cin >> w[i];//存入相应编号对应的重量
    }
    backtrack(0, 0);

    cout << bestWeight << endl;
    for (int j = 0; j < n;j++){
        if(bestSolution[j]==1){
            cout << j + 1 << " ";
        }
    }
    cout << endl;

    return 0;
}

/*题后总结：
1.漏掉了最佳解法的更新，导致最后输出有问题；
2.把判断更优解的函数写成了两种分支的情况，导致一旦遇到一个两者元素相等的情况就会跳过后面元素的比较，得到的最优解不一定是最优

3.回溯的本质是暴力穷举*/