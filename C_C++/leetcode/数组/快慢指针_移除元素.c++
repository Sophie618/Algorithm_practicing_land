#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slow=0,fast;
        // int k=0;
        for(fast=0;fast<nums.size();fast++){
            if(nums[fast]!=val){
                nums[slow++]=nums[fast];
                // slow++;
                // k++;
            }
            else continue;
        }
        // return k;
        return slow;
    }
};

/*题后总结：
1. 快慢指针法，相比两层for循环的O(n^2)，快慢指针法的时间复杂度为O(n)，空间复杂度为O(1)。
2. 优化：slow++,可以直接替代k++计数器的作用
3. 更简洁化：直接在取nums[slow]时让slow自加
*/