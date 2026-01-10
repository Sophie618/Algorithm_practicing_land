#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        vector<int> result(nums.size(),0);
        int k=nums.size()-1;
        for(int i=0,j=nums.size()-1;i<=j;){
            if(nums[i]*nums[i] <= nums[j]*nums[j]){
                result[k--]=nums[j]*nums[j];
                j--;
            }
            else{
                result[k--]=nums[i]*nums[i];
                i++;
            }
        }
        return result;
    }
};

/*题后总结：
1.一开始自己写的时候j=nums.size(),导致一直报错又看不出哪里有问题😅
2.经过观察采用快慢指针边遍历边比较，特色是这类方法参数的更新往往是散落的，不会一次性写在for循环条件中
*/