#include <bits/stdc++.h>
using namespace std;

bool isdigit(char c){
    return((c>='A'&&c<='Z')||(c>='a'&&c<='z')) ;
}

bool isOperator(char c){
    return(c=='+'||c=='-'||c=='*'||c=='/'||c=='^');
}

int getPriority(char c){
    switch(c){
        case '+':
        case '-':return 1;
        case '*':
        case '/':return 2;
        case '^':return 3;
        default:return 0;
    }
}

string infixToPostfix(string infix) {
    stack<char> s;
    string postfix="";
    for (char c : infix) {
        if(c=='#'){
            while(!s.empty()){
                postfix+=s.top();
                s.pop();
            }
            break;
        }
        else if (isdigit(c)) {
            postfix += c;
        }
        else if(isOperator(c)){
            if(!s.empty()&&getPriority(s.top())>=getPriority(c)){
                postfix+=s.top();
                s.pop();
            }
            s.push(c);
        }
        else if(c=='('){
            s.push(c);
        }
        else if(c==')'){
            while(!s.empty()&&s.top()!='('){
                postfix +=s.top();
                s.pop();
            }
            s.pop();//弹出左括号，但不输出
        }
    }
    return postfix;
}

int main() {
    int n;
    cin>>n;
    cin.ignore();//忽略换行符

    for(int i=0;i<n;i++){
        string infix;
        getline(cin, infix);
        string postfix=infixToPostfix(infix);
        cout<<postfix<<endl;
    }
}

/*题后总结：
1.所有关于string.的函数，必须加一对括号，因为string是类，不是函数
2.学会使用getline(cin, string)来读取一行字符串
3.学会使用cin.ignore()来忽略换行符
4.stack<char> s;是栈的定义，栈的常用操作是push(x)，pop()，top()，empty()，size()
5.特殊情况：^是右结合运算符，从右往左结合才正确，所以即使运算级别==也不弹出，继续压栈中
*/