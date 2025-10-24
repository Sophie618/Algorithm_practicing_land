# fixed 的作用：
- 设置浮点数为定点表示法（固定小数点位数）
- 配合 setprecision(n) 控制小数点后的位数
- 持久生效（不像 setw 只影响一次）
## 记忆口诀：
无 fixed：setprecision = 有效数字总数
有 fixed：setprecision = 小数点后位数
用途：金额、百分比、平均值等需要固定小数位的场景

``` cpp
cout << "total landing requests: " << fixed << setprecision(1) << setw(4) << avg_land_wait;
// 或者更清晰
cout << fixed << setprecision(1);  // 提前设置格式
cout << "total landing requests: " << setw(4) << avg_land_wait << endl;
```

# setfill('0')影响后续输出的问题：
## setfill 的作用机制
- setfill 是一个持久性的格式控制符，它会一直影响后续所有输出，直到你再次改变它。
- 举个例子：
```cpp
cppcout << setfill('0');
cout << setw(4) << 123 << endl;  // 输出：0123
cout << setw(4) << 456 << endl;  // 输出：0456  ← 注意！还是用0填充
cout << setw(4) << 789 << endl;  // 输出：0789  ← 还是用0填充

void freeRunways(int current_time){
    for(int i = 0; i < num; i++){
        if(runways[i].free_time == current_time && runways[i].free_time > 0){
            cout << "runway " << setw(2) << setfill('0') << (i+1) 
                 << setfill(' ') << " is free" << endl;
            //        ^用0填充跑道号  ^立即恢复，防止影响后续输出
        }
    }
}
```
