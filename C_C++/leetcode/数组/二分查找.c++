#include<bits/stdc++.h>
using namespace std;

class Solution
{
public://左闭右闭 
int search(vector<int> &nums, int target)
    {
        int left = 0, right = size(nums) - 1;
        while (left <= right)
        {
            int middle = left + (right - left) / 2;
            if (nums[middle] > target)
            {
                right = middle - 1;
            }
            else if (nums[middle] < target)
            {
                left = middle + 1;
            }
            else
                return middle;
        }
        return -1;
    }

//左闭右开
int search(vector<int> &nums, int target)
{
    int left = 0, right = size(nums) ;//右边界本身就不包含nums[size(nums)]
    while (left < right)
    {
        int middle = left + (right - left) / 2;
        if (nums[middle] > target)
        {
            right = middle ;
        }
        else if (nums[middle] < target)
        {
            left = middle+1;
        }
        else
            return middle;
    }
    return -1;
}
}
;

/*题后总结：
1.注意循环不变量，选定一个左闭右开或是左闭右闭后面就一直选择这种形式
2.每种循环不变量对应的边界条件是不一样的，注意区分*/