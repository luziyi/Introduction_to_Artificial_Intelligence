// 产生式动物识别系统

#include <iostream>
#include <cstring>
using namespace std;
string fea[25] = {"", "有毛发", "有奶", "有羽毛", "会飞", "会下蛋", "吃肉", "有犬齿", "有爪", "眼町前方", "有蹄", "嚼反刍", "黄褐色", "身上有暗斑点", "身上有黑色条纹", "有长脖子", "有长腿", "不会飞", "会游泳", "有黑白二色", "善飞", "哺乳动物", "鸟", "食肉动物", "蹄类动物"};

string judge(int fea[])
{
    string answer;
    if (fea[1])
        fea[21] = 1;
    if (fea[2])
        fea[21] = 1;
    if (fea[3])
        fea[22] = 1;
    if (fea[4] && fea[5])
        fea[22] = 1;
    if (fea[6])
        fea[23] = 1;
    if (fea[7] && fea[8] && fea[9])
        fea[23] = 1;
    if (fea[21] && fea[10])
        fea[24] = 1;
    if (fea[21] && fea[11])
        fea[24] = 1;

    if (fea[21] && fea[23] && fea[12] && fea[13])
        answer = "金钱豹";
    else if (fea[21] && fea[23] && fea[12] && fea[14])
        answer = "虎";
    else if (fea[24] && fea[15] && fea[16] && fea[13])
        answer = "长颈鹿";
    else if (fea[24] && fea[14])
        answer = "斑马";
    else if (fea[22] && fea[15] && fea[16] && fea[17] && fea[19])
        answer = "鸵鸟";
    else if (fea[22] && fea[18] && fea[17] && fea[19])
        answer = "企鹅";
    else if (fea[22] && fea[20])
        answer = "信天翁";
    else
        answer = "error";

    return answer;
}

int main()
{
    string s;
    cout << "动物特征如下：\n";
    for (int i = 1; i <= 24; i++)
    {
        cout << i << "." << fea[i] << "\t";
        if (i % 4 == 0)
            cout << endl;
    }
    cout << "----------------------------------------------------------\n请输入数字选择动物的特征，结尾处用end(输入stop停止运行程序)：" << endl;
    while (cin >> s && s != "stop")
    {
        int inpfea[25] = {0};
        while (cin >> s && s != "end")
        {
            inpfea[stoi(s)] = 1;
        }
        //输出inpfea

        string answer;
        answer = judge(inpfea);
        if (answer != "error")
            cout << "success:该动物名称为：\n"
                 << answer << endl;
        else
            cout << "error:无法识别该动物\n";
    }
    cout<<"程序成功退出!\n";
    return 0;
}