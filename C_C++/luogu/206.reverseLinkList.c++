/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution
{
public:
    ListNode *reverseList(ListNode *head)
    {
        if (head == nullptr)
        {
            return nullptr;
        }
        ListNode *pre = nullptr;
        ListNode *cur = head;

        while (cur != nullptr)
        {
            ListNode *temp = cur->next; // 新一轮直接从cur的下一个开始，防止了nullptr->next导致爆炸
            cur->next = pre;            // 反转
            pre = cur;                  // 步进
            cur = temp;                 // 步进
        } // 退出时cur==nullptr，其前一个pre值刚好是反转链表的新头节点
        return pre;
    }
};