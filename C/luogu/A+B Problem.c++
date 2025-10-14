#include <bits/stdc++.h>
using namespace std;

string add(string a,string b){
    //补齐位数，不足位补零
    int maxlen=max(a.length(),b.length());
    while(a.length()<maxlen) a="0"+a;
    while(b.length()<maxlen) b="0"+b;

    string result="";
    int carry=0;

    for(int i=maxlen-1;i>=0;i--){
        int sum=(a[i]-'0')+(b[i]-'0')+carry;
        result=char(sum%10+'0')+result;
        carry=sum/10;
    }
    if(carry>0) result=char(carry+'0')+result;
    return result;
}

int main(){
    string a,b;
    cin>>a>>b;
    cout<<add(a,b)<<endl;
    return 0;
}

/*
加法步骤：
1.补齐位数，不足位前面要补零
2.从低位开始相加，如果相加结果大于10，则进位，结果为结果的个位，进位为结果的十位
3.处理最高位进位
*/