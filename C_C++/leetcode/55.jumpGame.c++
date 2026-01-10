// https://leetcode.cn/problems/jump-game/
//贪心算法
#include <algorithm>
class Solution
{
public:
    bool canJump(vector<int> &nums)
    {
        int m = nums[0]; // 代表最远的射程下标
        for (int i = 0; i < nums.size(); i++)
        {
            if (m < i)
            { // 剪枝，若连当前位置都无法抵达直接不行
                return false;
            }
            m = max(i + nums[i], m);
            if (m >= nums.size() - 1)
            {
                return true;
            }
        }
        return false;
    }
};