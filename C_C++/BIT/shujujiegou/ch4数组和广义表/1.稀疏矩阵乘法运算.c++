#include<bits/stdc++.h>
using namespace std;

#define MAXSIZE 125
typedef int ElemType;
struct Triple
{
    int i, j;//非零元的行下标和列下标
    ElemType e;
};
struct TSMatrix
{
    Triple data[MAXSIZE + 1];//非零元三元组表，data[0]未用
    int mu, nu, tu;//行数，列数，非零元个数
};

void CreateSMatrix(TSMatrix &M){
    cin >> M.mu >> M.nu >> M.tu;
    for(int k = 1; k < M.tu+1;k++){
        cin >> M.data[k].i >> M.data[k].j >> M.data[k].e;
    }
}

// data: 1-based 三元组数组 data[1..num]
// 将数组按 (i,j) 排序并合并相同位置的元素，返回压缩后的元素个数
void mergeSort(Triple data[], int &num){
    if (num <= 0) return;
    std::sort(data + 1, data + num + 1, [](const Triple &a, const Triple &b){
        if (a.i != b.i) return a.i < b.i;
        return a.j < b.j;
    });

    int write = 0;
    for (int r = 1; r <= num; ++r){
        if (write == 0 || data[r].i != data[write].i || data[r].j != data[write].j){
            ++write;
            data[write] = data[r];
        } else {
            data[write].e += data[r].e;
        }
    }

    // 去掉合并后值为 0 的元素
    int finalWrite = 0;
    for (int r = 1; r <= write; ++r){
        if (data[r].e != 0){
            ++finalWrite;
            data[finalWrite] = data[r];
        }
    }
    num = finalWrite;
}

TSMatrix multiMatrix(TSMatrix M1,TSMatrix M2){
    TSMatrix M3;
    int k = 1;
    M3.mu = M1.mu;
    M3.nu = M2.nu;
    M3.tu = 0;
    for (int i = 1; i < M1.tu+1;i++){
        for (int j = 1; j < M2.tu+1;j++){
            if(M1.data[i].j==M2.data[j].i){
                M3.data[k].i = M1.data[i].i;
                M3.data[k].j = M2.data[j].j;
                M3.data[k].e = M1.data[i].e * M2.data[j].e;
                M3.tu++;
                k++;
            }
        }
    }
    mergeSort(M3.data, M3.tu);
    return M3;
}

void print(TSMatrix M){
    cout << M.mu << endl
         << M.nu <<endl
         <<M.tu << endl;
    // 数据按 1-based 存储：data[1]..data[tu]
    for (int k = 1; k <= M.tu; k++){
        cout << M.data[k].i << "," << M.data[k].j << "," << M.data[k].e << endl;
    }
}

int main(){
    TSMatrix M1,M2,M3;
    CreateSMatrix(M1);
    CreateSMatrix(M2);
    M3 = multiMatrix(M1,M2);
    print(M3);
    return 0;
}