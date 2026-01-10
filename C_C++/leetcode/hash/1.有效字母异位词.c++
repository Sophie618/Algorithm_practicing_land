#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    bool isAnagram(string s, string t)
    {
        int hash[26];
        int flag = 0;
        for (int i = 0; i < s.size(); i++)
        {
            hash[s[i] - 'a']++;
        }
        for (int i = 0; i < t.size(); i++)
        {
            hash[t[i] - 'a']--;
        }
        for (int k = 0; k < 26; k++)
        {
            if (hash[k] != 0)
            {
                return false;
            }
        }
        return true;
    }
};