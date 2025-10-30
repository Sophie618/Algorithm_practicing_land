#include <bits/stdc++.h>
using spacename std;

class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int sum=0,i=0,find=0,result=nums.size(),subL=0;
        for(int j=0;j<nums.size();j++){
            sum+=nums[j];
            while(sum>=target){
                find=1;
                subL=j-i+1;
                result=min(result,subL);
                sum-=nums[i];
                i++;
            }
        }
        if(find) return result;
        return 0;
    }
};
/*题后总结：
1.记住要用一个中间变量result来存储滑动过程中的最小值，需要用j把整个数组都遍历完才可以说是所有子数组里面长度最短的
2.用while循环找出每个位置能满足条件的最大i
3.记得最后使用标识符find来判断是否找到*/