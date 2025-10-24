#include <bits/stdc++.h>
using namespace std;

struct Runway{
    int free_time;
    int total_busy_time;

    Runway():free_time(0),total_busy_time(0){}
};

struct Airplane{
    int id;
    int arrive_time;

    Airplane(int _id,int _time):id(_id),arrive_time(_time){}
};

int num,land_time,takeoff_time;
vector<Runway> runways;
queue<Airplane> landing_queue;
queue<Airplane> takeoff_queue;

int land_id=5001;
int takeoff_id=1;

int total_land_wait=0;
int total_takeoff_wait=0;
int land_count=0;
int takeoff_count=0;
int total_busy_time=0;

void init(){//初始化
    cin>>num>>land_time>>takeoff_time;
    runways.resize(num);//动态分配跑道数
}

void freeRunways(int current_time){//释放空闲跑道
    for(int i=0;i<num;i++){
        if(runways[i].free_time==current_time&&runways[i].free_time>0){
            cout<<"runway "<<setw(2)<<setfill('0')<<(i+1)<<" is free"<<endl;
        }
    }
}

bool isRunwayFree(int runway_idx,int current_time){
    return runways[runway_idx].free_time<=current_time;
}

bool allRunwaysFree(int current_time){
    for(int i=0;i<num;i++){
        if(!isRunwayFree(i,current_time)) return false;
    }
    return true;
}

void assignRunways(int current_time){
    for(int i=0;i<num;i++){
        if(!isRunwayFree(i,current_time)) continue;
        //有空闲跑道
        if(!landing_queue.empty()){//优先降落
            Airplane plane=landing_queue.front();
            landing_queue.pop();

            int wait=current_time-plane.arrive_time;
            total_land_wait+=wait;
            land_count++;

            cout<<"airplane "<<setw(4)<<setfill('0')<<plane.id<<" is ready to land on runway "<<setw(2)<<setfill('0')<<(i+1)<<endl;

            runways[i].free_time=current_time+land_time;
            runways[i].total_busy_time+=land_time;
        }
        else if(!takeoff_queue.empty()){//后起飞
            Airplane plane=takeoff_queue.front();
            takeoff_queue.pop();

            int wait=current_time-plane.arrive_time;
            total_takeoff_wait+=wait;
            takeoff_count++;

            cout<<"airplane "<<setw(4)<<setfill('0')<<plane.id<<" is ready to takeoff on runway "<<setw(2)<<setfill('0')<<(i+1)<<endl;

            runways[i].free_time=current_time+takeoff_time;
            runways[i].total_busy_time+=takeoff_time;
        }
    }
}

void printStatistics(int simulation_time){
    cout<<"simulation finished"<<endl;
    cout<<"simulation time: "<<setw(4)<<setfill(' ')<<simulation_time<<endl;
    if(land_count>0){
        double avg_land_wait=(double)total_land_wait/land_count;
        cout<<"average waiting time of landing: "<<setw(4)<<fixed<<setprecision(1)<<setfill(' ')<<avg_land_wait<<endl;
    }
    if(takeoff_count>0){
        double avg_takeoff_wait=(double)total_takeoff_wait/takeoff_count;
        cout<< "average waiting time of takeoff: " <<setw(4)<<fixed<<setprecision(1)<<setfill(' ')<<avg_takeoff_wait<<endl;
    }
    for(int i=0;i<num;i++){
        cout<<"runway "<<setw(2)<<setfill('0')<<(i+1)<<" busy time: "<<setw(4)<<setfill(' ')<<runways[i].total_busy_time<<endl;
        total_busy_time+=runways[i].total_busy_time;
    }
    double avg_percentage=(double)total_busy_time*100/(num*simulation_time);
    cout<<"runway average busy time percentage: "<<setw(4)<<fixed<<setprecision(1)<<avg_percentage<<"%"<<endl;
}

int main(){
    init();
    int current_time=0;
    bool airport_closed=false;
    while(1){
        cout<<"Current Time: "<<setw(4)<<setfill(' ')<<current_time<<endl;
        freeRunways(current_time);
        if(!airport_closed){
            int land_req,takeoff_req;
            cin>>land_req>>takeoff_req;
            if(land_req<0&&takeoff_req<0) airport_closed=true;
            else{
                for(int i=0;i<land_req;i++){
                    landing_queue.push(Airplane(land_id++,current_time));
                }
                for(int i=0;i<takeoff_req;i++){
                    takeoff_queue.push(Airplane(takeoff_id++,current_time));
                }
            }
        }
        assignRunways(current_time);
        if(airport_closed&&landing_queue.empty()&&takeoff_queue.empty()&&allRunwaysFree(current_time)) break;
        current_time++;
    }
    printStatistics(current_time);
    return 0;
}