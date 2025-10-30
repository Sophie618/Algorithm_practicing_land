#include <iostream>
#include <stack>
#include <string>
using namespace std;

// 判断是否为操作数（字母）
bool isOperand(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// 获取运算符优先级
int getPriority(char op) {
    switch(op) {
        case '^': return 4;  // 指数运算优先级最高
        case '*':
        case '/': return 3;
        case '+':
        case '-': return 2;
        case '(': return 1;  // 左括号优先级最低
        default: return 0;
    }
}

// 判断是否为运算符
bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
}

// 中缀转后缀函数
string infixToPostfix(string infix) {
    stack<char> s;
    string postfix = "";
    
    for (int i = 0; i < infix.length(); i++) {
        char c = infix[i];
        
        if (c == '#') {
            // 表达式结束，弹出栈中剩余元素
            while (!s.empty()) {
                postfix += s.top();
                s.pop();
            }
            break;
        }
        else if (isOperand(c)) {
            // 操作数直接输出
            postfix += c;
        }
        else if (c == '(') {
            // 左括号压入栈
            s.push(c);
        }
        else if (c == ')') {
            // 右括号，弹出到左括号
            while (!s.empty() && s.top() != '(') {
                postfix += s.top();
                s.pop();
            }
            s.pop(); // 弹出左括号，但不输出
        }
        else if (isOperator(c)) {
            // 处理运算符
            while (!s.empty() && 
       s.top() != '(' && 
       (getPriority(s.top()) > getPriority(c) ||
        (getPriority(s.top()) == getPriority(c) && c != '^'))) 
        {
        postfix += s.top();
            s.pop();
        }
        s.push(c);
        }
    }
    
    return postfix;
}

int main() {
    int n;
    cin >> n;
    cin.ignore(); // 忽略换行符
    
    for (int i = 0; i < n; i++) {
        string infix;
        getline(cin, infix);
        
        string postfix = infixToPostfix(infix);
        cout << postfix << endl;
    }
    
    return 0;
}
