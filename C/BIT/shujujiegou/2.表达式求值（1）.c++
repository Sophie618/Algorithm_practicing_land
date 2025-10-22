#include <bits/stdc++.h>
using namespace std;

// 判断是否为数字
bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

// 判断是否为运算符
bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == '^';
}

// 获取运算符优先级
int getPriority(char op) {
    switch(op) {
        case '^': return 4;
        case '*':
        case '/':
        case '%': return 3;
        case '+':
        case '-': return 2;
        case '(': return 1;
        default: return 0;
    }
}

// 预处理表达式，处理负数
string preprocess(string expr) {
    string result = "";
    for (int i = 0; i < expr.length(); i++) {
        char c = expr[i];
        if (c == '-' && (i == 0 || expr[i-1] == '(' || isOperator(expr[i-1]))) {
            // 这是一个负号，不是减号
            result += "(0-";
            i++;
            // 读取完整的负数
            while (i < expr.length() && isDigit(expr[i])) {
                result += expr[i];
                i++;
            }
            result += ")";
            i--; // 回退一位，因为for循环会自增
        } else {
            result += c;
        }
    }
    return result;
}

// 检查表达式是否有效
bool isValidExpression(string expr) {
    stack<char> parenStack;
    bool expectOperand = true; // true表示期望操作数，false表示期望运算符
    
    for (int i = 0; i < expr.length(); i++) {
        char c = expr[i];
        
        if (c == '(') {
            parenStack.push(c);
            expectOperand = true;
        } else if (c == ')') {
            if (parenStack.empty()) return false;
            parenStack.pop();
            expectOperand = false;
        } else if (isDigit(c)) {
            if (!expectOperand) return false; // 连续的数字之间缺少运算符
            expectOperand = false;
            // 跳过完整的数字
            while (i + 1 < expr.length() && isDigit(expr[i + 1])) {
                i++;
            }
        } else if (isOperator(c)) {
            if (expectOperand) return false; // 运算符后缺少操作数
            expectOperand = true;
        }
    }
    
    return parenStack.empty() && !expectOperand;
}

// 中缀转后缀
vector<string> infixToPostfix(string infix) {
    stack<char> opStack;
    vector<string> postfix;
    string currentNumber = "";
    
    for (int i = 0; i < infix.length(); i++) {
        char c = infix[i];
        
        if (isDigit(c)) {
            currentNumber += c;
        } else {
            // 如果有累积的数字，先输出
            if (!currentNumber.empty()) {
                postfix.push_back(currentNumber);
                currentNumber = "";
            }
            
            if (c == '(') {
                opStack.push(c);
            } else if (c == ')') {
                while (!opStack.empty() && opStack.top() != '(') {
                    postfix.push_back(string(1, opStack.top()));
                    opStack.pop();
                }
                if (!opStack.empty()) {
                    opStack.pop(); // 弹出左括号
                }
            } else if (isOperator(c)) {
                while (!opStack.empty() && 
                       opStack.top() != '(' && 
                       (getPriority(opStack.top()) > getPriority(c) ||
                        (getPriority(opStack.top()) == getPriority(c) && c != '^'))) {
                    postfix.push_back(string(1, opStack.top()));
                    opStack.pop();
                }
                opStack.push(c);
            }
        }
    }
    
    // 处理最后的数字
    if (!currentNumber.empty()) {
        postfix.push_back(currentNumber);
    }
    
    // 弹出剩余运算符
    while (!opStack.empty()) {
        postfix.push_back(string(1, opStack.top()));
        opStack.pop();
    }
    
    return postfix;
}

// 后缀表达式求值
int evaluatePostfix(vector<string> postfix) {
    stack<int> numStack;
    
    for (string token : postfix) {
        if (isDigit(token[0]) || (token[0] == '-' && token.length() > 1)) {
            // 这是一个数字
            numStack.push(stoi(token));
        } else {
            // 这是一个运算符
            if (numStack.size() < 2) {
                throw runtime_error("Invalid expression");
            }
            
            int b = numStack.top(); numStack.pop();
            int a = numStack.top(); numStack.pop();
            int result;
            
            switch(token[0]) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': 
                    if (b == 0) throw runtime_error("Divide by zero");
                    result = a / b; 
                    break;
                case '%': 
                    if (b == 0) throw runtime_error("Divide by zero");
                    result = a % b; 
                    break;
                case '^': 
                    if (b < 0) throw runtime_error("Negative exponent");
                    result = 1;
                    for (int i = 0; i < b; i++) {
                        result *= a;
                    }
                    break;
            }
            
            numStack.push(result);
        }
    }
    
    if (numStack.size() != 1) {
        throw runtime_error("Invalid expression");
    }
    
    return numStack.top();
}

int operate(string expression){
    try {
        // 预处理
        string processed = preprocess(expression);
        
        // 检查有效性
        if (!isValidExpression(processed)) {
            cout << "error." << endl;
            return 0;
        }
        
        // 转换和求值
        vector<string> postfix = infixToPostfix(processed);
        int result = evaluatePostfix(postfix);
        cout << result << endl;
        
    } catch (const runtime_error& e) {
        if (string(e.what()) == "Divide by zero") {
            cout << "Divide 0." << endl;
        } else {
            cout << "error." << endl;
        }
    }
    return 0;
}

int main(){
    string expression;
    int n;
    cin >> n;
    cin.ignore(); // 忽略换行符
    for (int i = 0; i < n; i++){
        getline(cin, expression);
        operate(expression);
    }
}