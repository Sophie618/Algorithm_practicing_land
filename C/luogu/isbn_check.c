#include <stdio.h>
#include <math.h>
#include <string.h>



int main() {
    char isbn[15];
    scanf("%s", isbn);
    int nums[9],pos=0;
    for(int i=0;i<strlen(isbn);i++){
        if(isbn[i]>='0'&&isbn[i]<='9'){
            nums[pos++]=isbn[i]-'0';
            if (pos==9) break;
        }
    }
    char check=isbn[strlen(isbn)-1];
    int sum=0;
    for(int i = 0; i < 9; i++) {
        sum += nums[i] * (i + 1);
    }
    int mod=sum%11;
    char correct_code=(mod==10)?'X':mod+'0';
    if(correct_code==check){
        printf("Right\n");
    }
    else{
        printf("%d-%c%c%c-%c%c%c%c%c-%c\n", 
            nums[0], nums[1]+'0', nums[2]+'0', nums[3]+'0',
            nums[4]+'0', nums[5]+'0', nums[6]+'0', nums[7]+'0', nums[8]+'0',
            correct_code);
        }
    return 0;
}