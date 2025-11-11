"""
Python 基础编程题 - 面试必备
这些是最常见的Python基础题，务必熟练掌握
"""

# ============= 题目1：列表推导式 =============
print("=" * 50)
print("题目1：使用列表推导式生成1-100的平方数")
squares = [x**2 for x in range(1, 101)]
print(f"前10个: {squares[:10]}")

# 进阶：筛选偶数的平方
even_squares = [x**2 for x in range(1, 101) if x % 2 == 0]
print(f"偶数的平方前10个: {even_squares[:10]}")


# ============= 题目2：字典操作 =============
print("\n" + "=" * 50)
print("题目2：统计单词频率")

def word_frequency(text):
    """统计文本中每个单词出现的次数"""
    words = text.lower().split()
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq

text = "hello world hello python python python"
result = word_frequency(text)
print(f"单词频率: {result}")


# ============= 题目3：Lambda 和 map/filter =============
print("\n" + "=" * 50)
print("题目3：使用lambda和map/filter")

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map: 将所有数字乘以2
doubled = list(map(lambda x: x * 2, numbers))
print(f"乘以2: {doubled}")

# filter: 筛选出偶数
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"偶数: {evens}")


# ============= 题目4：类的定义（面向对象） =============
print("\n" + "=" * 50)
print("题目4：定义一个学生类")

class Student:
    def __init__(self, name, age, scores):
        self.name = name
        self.age = age
        self.scores = scores
    
    def average_score(self):
        """计算平均分"""
        return sum(self.scores) / len(self.scores)
    
    def __str__(self):
        """打印学生信息"""
        return f"Student(name={self.name}, age={self.age}, avg={self.average_score():.2f})"

student = Student("张三", 20, [85, 90, 88, 92])
print(student)
print(f"平均分: {student.average_score():.2f}")


# ============= 题目5：装饰器（进阶） =============
print("\n" + "=" * 50)
print("题目5：实现一个计时装饰器")

import time

def timer(func):
    """计算函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.6f}秒")
        return result
    return wrapper

@timer
def slow_function():
    """模拟耗时操作"""
    time.sleep(0.1)
    return "完成"

result = slow_function()


# ============= 题目6：异常处理 =============
print("\n" + "=" * 50)
print("题目6：安全地除法运算")

def safe_divide(a, b):
    """安全的除法，处理除零错误"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("错误：除数不能为0")
        return None
    except TypeError:
        print("错误：输入必须是数字")
        return None

print(f"10 / 2 = {safe_divide(10, 2)}")
print(f"10 / 0 = {safe_divide(10, 0)}")


# ============= 题目7：文件读写 =============
print("\n" + "=" * 50)
print("题目7：读写文件")

# 写文件
with open('temp_test.txt', 'w', encoding='utf-8') as f:
    f.write("第一行\n")
    f.write("第二行\n")
    f.write("第三行\n")

# 读文件
with open('temp_test.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print("文件内容:")
    print(content)

# 清理
import os
os.remove('temp_test.txt')


# ============= 题目8：数据结构操作 =============
print("\n" + "=" * 50)
print("题目8：找出列表中的重复元素")

def find_duplicates(lst):
    """找出列表中的重复元素"""
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)

numbers = [1, 2, 3, 4, 2, 3, 5, 6, 7, 3]
print(f"重复元素: {find_duplicates(numbers)}")


# ============= 题目9：排序（自定义排序） =============
print("\n" + "=" * 50)
print("题目9：按多个条件排序")

students = [
    {'name': '张三', 'age': 20, 'score': 85},
    {'name': '李四', 'age': 22, 'score': 90},
    {'name': '王五', 'age': 20, 'score': 88},
]

# 按分数降序排序
sorted_students = sorted(students, key=lambda x: x['score'], reverse=True)
print("按分数排序:")
for s in sorted_students:
    print(f"  {s['name']}: {s['score']}")


# ============= 题目10：生成器（进阶） =============
print("\n" + "=" * 50)
print("题目10：使用生成器节省内存")

def fibonacci(n):
    """生成斐波那契数列的前n项"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print("斐波那契数列前10项:")
print(list(fibonacci(10)))


print("\n" + "=" * 50)
print("Python基础题完成！这些都是高频面试题，建议多练习几遍")

