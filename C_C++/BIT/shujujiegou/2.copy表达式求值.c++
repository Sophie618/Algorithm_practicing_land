#include <bits/stdc++.h>
using namespace std;

bool isDigit(char c){
    return c>='0'&&c<='9';
}

bool isOperator(char c){
    return c=='+'||c=='-'||c=='*'||c=='/'||c=='%'||c=='^';
}

int getPriority(char c){
    switch(c){
        case '(': return 1;
        case '-':
        case '+': return 2;
        case '/':
        case '*':
        case '%': return 3;
        case '^': return 4;
        default : return 0;
    }
}

string preprocess(string expression){//把原式过滤一遍负号
    string result="";
    for(int i=0;i<expression.length();i++){
        char c=expression[i];
        if(c=='-' && (i==0||expression[i-1]=='('||isOperator(expression[i-1]))){//处理负号
            result+="(0-";
            i++;
            while(i<expression.length()&&isDigit(expression[i])){//读取完整的负数
                result+=expression[i];
                i++;
            }
            result+=')';
            i--;//for循环会自增，回退一位
        }
        else{
            result+=c;
        }
    }
    return result;
}

bool isValidExpression(string expression){
    stack<char> parenStack;
    bool expectOperand=true;//true表示期望操作数，false表示期望运算符

    for(int i=0;i<expression.length();i++){
        char c=expression[i];
        if(c=='('){
            parenStack.push(c);
            expectOperand=true;
        }
        else if(c==')'){
            if(parenStack.empty()) return false;
            parenStack.pop();
            expectOperand=false;
        }
        else if(isDigit(c)){
            if(!expectOperand)  return false;
            expectOperand=false;
            while(i+1<expression.length()&&isDigit(expression[i+1])){
                i++;
            }
        }
        else if(isOperator(c)){
            if(expectOperand) return false;
            expectOperand=true;
        }
    }
    return parenStack.empty()&&!expectOperand;
}

vector<string> infixToPostfix(string infix){
    stack<char> opStack;
    vector<string> postfix;
    string currentNumber = "";
    
    for(int i=0;i<infix.length();i++){
        char c=infix[i];
        
        if(isDigit(c)){
            currentNumber += c;
        }
        else{
            // 如果有累积的数字，先输出
            if(!currentNumber.empty()){
                postfix.push_back(currentNumber);
                currentNumber = "";
            }
            
            if(c=='('){
                opStack.push(c);
            }
            else if(c==')'){
                while(!opStack.empty()&&opStack.top()!='('){
                    postfix.push_back(string(1,opStack.top()));
                    opStack.pop();
                }
                if(!opStack.empty()){
                    opStack.pop(); // 弹出左括号
                }
            }
            else if(isOperator(c)){
                while(!opStack.empty()&& 
                      opStack.top()!='('&& 
                      (getPriority(opStack.top())>getPriority(c)||
                       (getPriority(opStack.top())==getPriority(c)&&c!='^'))){
                    postfix.push_back(string(1,opStack.top()));
                    opStack.pop();
                }
                opStack.push(c);
            }
        }
    }
    
    // 处理最后的数字
    if(!currentNumber.empty()){
        postfix.push_back(currentNumber);
    }
    
    // 弹出剩余运算符
    while(!opStack.empty()){
        postfix.push_back(string(1,opStack.top()));
        opStack.pop();
    }
    
    return postfix;
}

int evaluatePostfix(vector<string> postfix){
    stack<int> numStack;

    for(string token:postfix){
        if(isDigit(token[0])||(token[0]=='-'&&token.length()>1)){
            numStack.push(stoi(token));
        }
        else {
            if(numStack.size()<2){//操作数不足
                throw runtime_error("Invalid expression");
            }
            int b=numStack.top();
            numStack.pop();
            int a=numStack.top();
            numStack.pop();
            int result;
            switch(token[0]){
                case '+': result=a+b; break;
                case '-': result=a-b; break;
                case '*': result=a*b; break;
                case '/': if(b==0) throw runtime_error("Divide by zero");
                result=a/b; break;
                case '%': if(b==0) throw runtime_error("Divide by zero");
                result=a%b; break;
                case '^': if(b<0) throw runtime_error("Negative exponent");
                result=1;
                for(int i=0;i<b;i++){
                    result*=a;
                }
                break;
            }
            numStack.push(result);
        }
    }
    if(numStack.size()!=1){
        throw runtime_error("Invalid expression");
    }
    return numStack.top();
}

int operate(string expression){
    try {
        //预处理
        string processed=preprocess(expression);
        //检查有效性
        if(!isValidExpression(processed)){
            cout<<"error."<<endl;
            return 0;
        }
        //中缀转后缀
        vector<string> postfix=infixToPostfix(processed);
        int result=evaluatePostfix(postfix);
        cout<<result<<endl;}
        catch(const runtime_error& e){
            if(string(e.what())=="Divide by zero"){
                cout<<"Divide 0."<<endl;
            }
            else{
                cout<<"error."<<endl;
            }
        }
    return 0;
}

int main(){
    int n;
    string expression;
    cin>>n;
    cin.ignore();
    for(int i=0;i<n;i++){
        getline(cin,expression);
        operate(expression);
    }
}



/*
知识整理：
postfix.push_back(string(1, opStack.top()));
1. opStack.top()
opStack 是一个 stack<char> 类型的栈
top() 返回栈顶的字符（如 '+', '-', '*' 等）
2. string(1, opStack.top())
string(1, char) 是 string 的构造函数
第一个参数 1 表示要创建的字符串长度
第二个参数 opStack.top() 是要转换为字符串的字符
结果：将单个字符转换为字符串
3. postfix.push_back(...)
postfix 是一个 vector<string> 类型的向量
push_back() 将字符串添加到向量末尾
4.stoi(token)
stoi 是 C++ 标准库函数，全称是 "string to integer"
作用：将字符串转换为整数
参数：token（一个字符串）
返回值：转换后的整数


题后总结：
1.throw会抛出异常，需要用try-catch捕获，try的过程中如果抛出异常，程序会终止，如果捕获异常，程序会继续执行。
2.try-catch可以捕获多种异常，如runtime_error,invalid_argument,out_of_range等。
3.isValidExpression函数算法使用状态机，遇到不同的字符，根据期望的操作数或运算符，进行状态转移。值得复刻。
4.const runtime_error& e中&是引用，引用不会创建新的对象，而是直接操作原对象。e是异常对象的别名。e.g.:int x = 10;
int& ref = x;  // & 是引用符，ref 是 x 的别名
*/