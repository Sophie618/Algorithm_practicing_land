# #列表推导式
# squares=[x**2 for x in range(1,101)]
# print(f"前十个：{squares[:10]}")

# even_squares=[x**2 for x in range(1,101) if x%2==0]
# print(f"偶数的平方前10个：{even_squares[:10]}")

#字典操作
def word_frequency(text):
    words=text.lower().split()
    freq={}
    for word in words:
        freq[word]=freq.get(word,0)+1
    return freq

text="hello world hello python python python"
result=word_frequency(text)
print(result)