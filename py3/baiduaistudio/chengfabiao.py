for i in range(1,10):
    for j in range(1,i+1):
        print(f"{j}*{i}={i*j:<2d} ",end="")
# /*f-string 的对齐符号是：

# < : 左对齐
# > : 右对齐 (数字类型的默认对齐方式)
# ^ : 居中对齐*/

    print()#print()函数会自动换行
        
for i in range(1,10):
    for j in range(1,i+1):
        print(f"{j}*{i}={i*j}\t",end="")#  使用制表符 \t 来对齐

    print()

multi_line_string = """这是一个
可以跨越
多行的字符串。"""

print(multi_line_string)

my_string = 'He said, "Hello, World!"'
# 如果用双引号，就需要转义："He said, \"Hello, World!\""