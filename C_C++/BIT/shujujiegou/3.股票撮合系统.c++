#include <bits/stdc++.h>
using namespace std;
struct order
{
    int order_id;
    int stock_id;
    float price;
    int quantity;
    char type;
    order*next;
};

struct stockQueue{
    int stock_id;
    order* b_head;
    order* s_head;
    stockQueue* next;
};

stockQueue* stockListHead=nullptr;
int currentOrderId=1;

order* create_order(int stock_id,float price,int quantity,char type){
    order* new_order=new order();
    new_order->order_id=currentOrderId++;
    new_order->stock_id=stock_id;
    new_order->price=price;
    new_order->quantity=quantity;
    new_order->type=type;
    new_order->next=nullptr;
    return new_order;
}

stockQueue* findOrCreateStock(int stock_id){
    stockQueue* curr=stockListHead;
    while(curr!=nullptr){
        if(curr->stock_id==stock_id) return curr;
        curr=curr->next;
    }
    stockQueue* new_stock=new stockQueue();
    new_stock->stock_id=stock_id;
    new_stock->b_head=nullptr;
    new_stock->s_head=nullptr;
    new_stock->next=stockListHead;
    stockListHead=new_stock;
    return new_stock;
}

void insertOrder(order* new_order,order** head,bool isBuyOrder){
    if(*head==nullptr){
        *head=new_order;
        return;
    }
    bool shouldInsertFront=false;
    if(isBuyOrder)
        shouldInsertFront=(new_order->price>(*head)->price);
    else
        shouldInsertFront=(new_order->price<(*head)->price);
    if(shouldInsertFront){
        new_order->next=*head;
        *head=new_order;
        return;
    }

    order* prev=*head;
    order* curr=(*head)->next;
    while(curr!=nullptr){
        bool shouldInsertHere=false;
        if(isBuyOrder){
            if(new_order->price>curr->price) shouldInsertHere=true;
        }
        else{
            if(new_order->price<curr->price) shouldInsertHere=true;
        }
        if(shouldInsertHere){
            break;
        }
        prev=curr;
        curr=curr->next;
    }
    new_order->next=curr;
    prev->next=new_order;
}

void matchAndInsert(order* new_order,stockQueue* stock){
    order** oppositeHead=nullptr;//定义对手方队列
    order** sameHead=nullptr;
    bool isBuyOrder=(new_order->type =='b');
    if(isBuyOrder){
        oppositeHead=&(stock->s_head);
        sameHead=&(stock->b_head);
    }
    else{
        oppositeHead=&(stock->b_head);
        sameHead=&(stock->s_head);
    }

    while(new_order->quantity>0&&*oppositeHead!=nullptr){
        order* topOrder=*oppositeHead;
        bool canMatch=false;
        if(isBuyOrder){
            canMatch=(new_order->price >= topOrder->price);
        }
        else{
            canMatch=(new_order->price <= topOrder->price);
        }
        if(!canMatch) break;//不满足匹配条件，退出循环

        float dealPrice=(new_order->price+topOrder->price)/2.0;
        int dealQuantity=min(new_order->quantity,topOrder->quantity);

        if(isBuyOrder){
            printf("deal--price:%6.1f  quantity:%4d  buyorder:%04d  sellorder:%04d\n"
                  ,dealPrice, dealQuantity, new_order->order_id,topOrder->order_id);
        }
        else{
            printf("deal--price:%6.1f  quantity:%4d  sellorder:%04d  buyorder:%04d\n"
                  ,dealPrice, dealQuantity,new_order->order_id,topOrder->order_id);
        }

        new_order->quantity-=dealQuantity;
        topOrder->quantity-=dealQuantity;

        if(topOrder->quantity==0){
            *oppositeHead=topOrder->next;
            delete topOrder;
        }
    }
    if(new_order->quantity>0){
        insertOrder(new_order,sameHead,isBuyOrder);
    }
    else{
        delete new_order;
    }
}

void process_new_order(int stock_id,float price,int quantity,char type){
    order* new_order=create_order(stock_id,price,quantity,type);
    cout<<"orderid: "<<setw(4)<<setfill('0')<<new_order->order_id<<endl;
    stockQueue* stock=findOrCreateStock(stock_id);
    matchAndInsert(new_order,stock);
}

void queryOrders(int stock_id){
    stockQueue* stock=stockListHead;
    while(stock!=nullptr){
        if(stock->stock_id==stock_id)
            break;
        stock=stock->next;
    }
    cout<<"buy orders:"<<endl;
    if(stock!=nullptr){
    order* curr=stock->b_head;
    while(curr!=nullptr){
        printf("orderid: %04d, stockid:%04d, price: %6.1f, quantity: %4d, b/s: %c\n",curr->order_id,curr->stock_id,curr->price,curr->quantity,curr->type);
        curr=curr->next;
    }}
    cout<<"sell orders:"<<endl;
    if(stock!=nullptr){
    order* curr=stock->s_head;
    while(curr!=nullptr){
        printf("orderid: %04d, stockid:%04d, price: %6.1f, quantity: %4d, b/s: %c\n",curr->order_id,curr->stock_id,curr->price,curr->quantity,curr->type);
        curr=curr->next;
    }}
}

bool removeFromQueue(order** head,int order_id,order** deletedOrder){
    if(*head==nullptr) return false;
    if((*head)->order_id==order_id){
        *deletedOrder=*head;
        *head=(*head)->next;
        return true;
    }
    order* prev=*head;
    order* curr=(*head)->next;
    while(curr!=nullptr){
        if(curr->order_id==order_id){
            *deletedOrder=curr;
            prev->next=curr->next;
            return true;
        }
        prev=curr;
        curr=curr->next;
    }
    return false;
}

void cancelOrder(int order_id){
    stockQueue* stock=stockListHead;
    order* deletedOrder=nullptr;
    bool found=false;
    while(stock!=nullptr){
        if(removeFromQueue(&(stock->b_head),order_id,&deletedOrder)){
            found=true;
            break;
        }
        if(removeFromQueue(&(stock->s_head),order_id,&deletedOrder)){
            found=true;
            break;
        }
        stock=stock->next;
    }
    if(found){
        printf("deleted order:orderid: %04d, stockid:%04d, price: %6.1f, quantity: %4d, b/s: %c\n",deletedOrder->order_id,deletedOrder->stock_id,deletedOrder->price,deletedOrder->quantity,deletedOrder->type);
        delete deletedOrder;
    }
    else{
        printf("not found\n");
    }
}

int main(){
    int command=1;
    while(command){
        cin>>command;
        if(command==0) break;
        else if(command==1){
            int stock_id,quantity;
            float price;
            char type;
            cin>>stock_id>>price>>quantity>>type;
            process_new_order(stock_id,price,quantity,type);
        }
        else if(command==2){
            int stock_id;
            cin>>stock_id;
            queryOrders(stock_id);
        }
        else if(command==3){
            int order_id;
            cin>>order_id;
            cancelOrder(order_id);
        }
    }
    return 0;
}




/*题后总结：
// 双指针变量解析：假设买单，对手方是卖单
oppositeHead = &(stock->s_head);     // oppositeHead 指向 s_head 这个变量

while(...) {
    topOrder = *oppositeHead;         // topOrder = 订单A的地址
    
    // ... 成交匹配 ...
    
    if(topOrder->quantity == 0) {     // 订单A 完全成交
        *oppositeHead = topOrder->next;  // s_head 改成指向订单B
        delete topOrder;                  // 删除订单A
    }
}
stock 结构体：
┌─────────────────────┐
│ stock_id: 100       │
│ b_head: ...         │
│ s_head: 0x1000 ─────┼──┐
└─────────────────────┘  │
                         │
                         ↓
内存地址 0x1000:  [订单A: price=10.5, quantity=100, next=0x2000]
                         │
                         ↓
内存地址 0x2000:  [订单B: price=10.8, quantity=50, next=0x3000]
                         │
                         ↓
内存地址 0x3000:  [订单C: price=11.0, quantity=80, next=nullptr]

oppositeHead = &(stock->s_head);
oppositeHead 指向 s_head 这个变量的地址
         ↓
    ┌────────┐
    │ s_head │ = 0x1000 (指向订单A)
    └────────┘
order* topOrder = *oppositeHead;
*oppositeHead  解引用，得到 s_head 的值 (0x1000)
      ↓
topOrder = 0x1000 (也指向订单A)

*oppositeHead = topOrder->next;
delete topOrder;
stock->s_head (在地址 0xABCD 处) 的值被改成了 0x2000
    ↓
   0x2000 ───→ [订单B] → [订单C] → nullptr
   
[订单A] ← topOrder 还指向这里（但已经从链表断开）

delete topOrder;
stock->s_head 
    ↓
[订单B] → [订单C] → nullptr

[订单A] 被删除释放内存 ✓
*/