# 1. 类型注解（LeetCode必备）
from typing import List, Optional, Dict, Set, Tuple

# 2. 数据结构
from collections import deque, defaultdict, Counter, OrderedDict
import heapq  # 优先队列/堆

# 3. 数学相关
import math
from math import inf, ceil, floor, sqrt

# 4. 其他常用
import sys
from itertools import permutations, combinations
from functools import lru_cache  # 记忆化搜索
import bisect  # 二分查找模块

# 具体场景对应关系
|C++常用|	Python对应导入|	说明|
|--------|----------------|----------|
|vector<int>|	from typing import List	|类型注解|
|queue<int>	|from collections import deque|	队列|
|priority_queue	|import heapq	|优先队列|
|unordered_map	|from collections import defaultdict|	哈希表|
|set<int>	|from typing import Set	|集合类型注解|
|INT_MAX	|from math import inf	|无穷大|
|sqrt()	|from math import sqrt	|数学函数|

# 交换两个变量（C++需要临时变量）
a, b = 5, 10
a, b = b, a  # 现在 a=10, b=5

# 同时赋值多个变量
x, y, z = 1, 2, 3

# 解包列表
nums = [1, 3, 5]
first, second, third = nums  # first=1, second=3, third=5

left, right = 0, len(nums) - 1;
等价于
left = 0
right = len(nums) - 1

# Python中的两种除法
1. / - 浮点除法（总是返回浮点数）
result = 7 / 2
print(result)  # 输出: 3.5（浮点数）

result = 10 / 2
print(result)  # 输出: 5.0（仍然是浮点数！）

2. // - 整除/地板除（返回整数部分）
result = 7 // 2
print(result)  # 输出: 3（整数）

result = 10 // 2
print(result)  # 输出: 5（整数）