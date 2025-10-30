from typing import List, Optional, Dict, Set, Tuple
from collections import deque, defaultdict, Counter
import heapq
import math
from math import inf

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1  # 定义target在左闭右闭的区间里，[left, right]

        while left <= right:
            middle = left + (right - left) // 2
            if nums[middle] > target:
                right = middle - 1  # target在左区间，所以[left, middle - 1]
            elif nums[middle] < target:
                left = middle + 1  # target在右区间，所以[middle + 1, right]
            else:
                return middle  # 数组中找到目标值，直接返回下标
                
        return -1  # 未找到目标值
    
def main():
    # 输入格式: [1,2,3,4,5] 3
    input_str = input("请输入数组和目标值:")
    parts = input_str.split()
    
    # 解析数组: 去掉方括号，按逗号分割，转换为整数列表
    nums_str = parts[0].strip('[]')
    nums = [int(x) for x in nums_str.split(',')] if nums_str else []
    
    # 解析目标值
    target = int(parts[1])
    
    result = Solution().search(nums, target)
    print(result)

if __name__=="__main__":
    main()