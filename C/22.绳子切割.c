#include <stdio.h>
#include <stdlib.h>

int binarySearch(int left,int right){
    if(left+1<right){
        int mid=left+(right-left)/2;//防止溢出
        if(check(mid)) return binarySearch(mid,right);
        else return binarySearch(left,mid);
    }
    return check(left)? left:0;
}

int check(int len){
    int count=0;
    for(int i=0;i<N;i++){
        count+=rope[i]/len;
    }
    if(count>=k) return 1;
    else return 0;
}

int main() {
    int N,max=0,k;
    scanf("%d", &N);//绳子数量
    int* rope=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        scanf("%d", &rope[i]);//绳子长度
        if(rope[i]>max) max=rope[i];
    }
    scanf("%d", &k);//需要切割的绳子数量
    printf("%d", binarySearch(1,max+1));
    free(rope);
    return 0;
}