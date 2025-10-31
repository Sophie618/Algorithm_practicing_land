#include <vector>
using namespace std;
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        int offset = 1, startx = 0, starty = 0,count=1;
        vector<vector<int>> nums(n, vector<int>(n, 0));
        int j,i;
        int loop=n/2;
        while (loop--) {
            i = startx;
            j = starty;
            for (; j < (n - offset); j++) {
                nums[i][j] = count++;
            }
            for (; i < (n - offset); i++) {
                nums[i][j] = count++;
            }
            // 上一轮循环结束后i、j达到最大值
            for (; j > starty; j--) {
                nums[i][j] = count++;
            }
            for (; i > startx; i--) {
                nums[i][j] = count++;
            }
            offset++;
            startx++;
            starty++;
        }
        if(n%2){
            nums[n/2][n/2]=count;
        }
        return nums;
    }
};