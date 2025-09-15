#include <stdio.h>
#include <stdlib.h>

// 二分搜索函数,递归实现,分治思想
int binarySearch(int arr[], int left, int right, int target) {
    int mid = left + (right - left) / 2;  // 防止溢出
    if (left>right) return -1;
    if(arr[mid]==target) return mid;
    else if(arr[mid]<target) return binarySearch(arr, mid+1, right,target);//目标在右半部分
    else if(arr[mid]>target) return binarySearch(arr, left, mid-1,target);//目标在左半部分
}

int main() {
    int n,target;
    scanf("%d", &n);
    int* arr=(int*)malloc(n*sizeof(int));
    for(int i=0;i<n;i++){
        scanf("%d", &arr[i]);
    }
    scanf("%d", &target);
    
    int result = binarySearch(arr, 0, n-1, target);
    
    if (result !=-1) {
        printf("元素 %d 在数组中的索引是: %d\n", target, result);
    } else {
        printf("元素 %d 不在数组中\n", target);
    }
    free(arr);

    return 0;
}

/*关键点：
1. 二分搜索的数组必须是有序的
2. 每次比较后，将搜索范围缩小一半，因此时间复杂度为 O(log n)
*/
