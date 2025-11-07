#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    vector<int> intersection(vector<int> &nums1, vector<int> &nums2)
    {
        unordered_set<int> result;
        unordered_set<int> nums_set(nums1.begin(), nums1.end());
        for (int num : nums2)
        {
            if (nums_set.find(num) != nums_set.end())
            {
                result.insert(num); // unordered_set类型可以自动去重，只用于记录出现过的元素
            }
        }
        return vector<int>(result.begin(), result.end());
    }
};