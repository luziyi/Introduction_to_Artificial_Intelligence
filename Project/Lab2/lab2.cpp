#include <iostream>
#include <queue>
#include <map>
using namespace std;
struct state
{
    int data[3][3];
    int g = 0, h = 0;
    int i, j; // 表示空格的位置
    bool operator>(const state &x) const
    {
        return g + h >= x.g + x.h;
    }
    string path;
};

map<long long, bool> m; // 标记open表中是否存在该情况

priority_queue<state, vector<state>, greater<state>> q; // 用优先队列实现open表

int result[3][3] = {1, 2, 3, 8, 0, 4, 7, 6, 5}; // 题目指定的目标状态
int xx[4] = {1, -1, 0, 0}, yy[4] = {0, 0, 1, -1};

int hn(state x) // 求出从当前状态到目标状态的估计代价，返回-1表示该状态存在过
{
    int h = 0;
    long long vis = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            h += (x.data[i][j] != result[i][j]);
            vis = vis * 10 + x.data[i][j];
        }
    }
    if (m[vis])
        return -1; // 标记vis中是否存在这个情况，也就是探测到的情况
    else
        m[vis] = 1;
    return h;
}

bool check(int result[3][3], int data[3][3]) // 检查输入是否合法
{
    int cnt = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (data[i][j] != 0)
                cnt++;
        }
    }
    if (cnt != 8)
    {
        printf("输入不合法\n");
        exit(0);
    }
    int vis[10] = {0};
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (data[i][j] < 0 || data[i][j] > 8 || vis[data[i][j]])
            {
                printf("输入不合法\n");
                exit(0);
            }
            vis[data[i][j]] = 1;
        }
    }
    return true;
}

int main()
{
    printf("请输入9个数（用空格分开，用0代表空格）：\n");
    state a;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cin >> a.data[i][j];
            if (a.data[i][j] == 0) // 用0表示空格
            {
                a.i = i;
                a.j = j;
            }
        }
    }

    if (check(result, a.data))
    {
        printf("输入合法\n");
    } // 检查输入是否合法
    else
        return 0;

    a.h = hn(a); // 计算估计代价

    q.push(a);

    while (!q.empty())
    {
        state x = q.top();
        q.pop();
        printf("+ - + - + - +\n");
        for (int i = 0; i < 3; i++)
        {
            printf("|");
            for (int j = 0; j < 3; j++)
            {
                printf(" %d |", x.data[i][j]);
            }
            printf("\n");
            printf("+ - + - + - +\n");
        }
        printf("\n");

        if (x.h == 0) // 如果f（n）为0，表明到达了目标状态
        {
            printf("空格移动路径为：%s\n", x.path.c_str());
            printf("最少移动次数为%d次\n", x.g + x.h);
            break;
        }
        for (int i = 0; i < 4; i++)
        {
            if (x.i + xx[i] < 0 || x.i + xx[i] > 2 || x.j + yy[i] < 0 || x.j + yy[i] > 2)
                continue;
            state y = x;
            swap(y.data[x.i][x.j], y.data[x.i + xx[i]][x.j + yy[i]]);
            y.g++;       // 每一步移动的代价是1
            y.h = hn(y); // 计算估计代价
            if (y.h == -1)
                continue; // vis中已经存在这个情况
            y.i = x.i + xx[i];
            y.j = x.j + yy[i];
            y.path += (i == 0 ? "下 " : (i == 1 ? "上 " : (i == 2 ? "右 " : "左 ")));
            q.push(y);
        }
    }
    return 0;
}
