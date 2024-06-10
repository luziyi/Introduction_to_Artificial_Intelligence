// 产生式动物识别系统

#include <iostream>
#include <cstring>
using namespace std;
string fea[25] = {"", "有毛发", "有奶", "有羽毛", "会飞", "会下蛋", "吃肉", "有犬齿", "有爪", "眼町前方", "有蹄", "嚼反刍", "黄褐色", "身上有暗斑点", "身上有黑色条纹", "有长脖子", "有长腿", "不会飞", "会游泳", "有黑白二色", "善飞", "哺乳动物", "鸟", "食肉动物", "蹄类动物"};

string judge(int inpfea[])
{
    string answer;
    cout << "推理过程：" << endl;
    if (inpfea[1])
    {
        inpfea[21] = 1;
        cout << "特征1：有毛发" << endl;
        cout << "推理结果：属于哺乳动物" << endl;
    }
    if (inpfea[2])
    {
        inpfea[21] = 1;
        cout << "特征2：有奶" << endl;
        cout << "推理结果：属于哺乳动物" << endl;
    }
    if (inpfea[3])
    {
        inpfea[22] = 1;
        cout << "特征3：有羽毛" << endl;
        cout << "推理结果：属于鸟类" << endl;
    }
    if (inpfea[4] && inpfea[5])
    {
        inpfea[22] = 1;
        cout << "特征4：会飞" << endl;
        cout << "特征5：会下蛋" << endl;
        cout << "推理结果：属于鸟类" << endl;
    }
    if (inpfea[6])
    {
        inpfea[23] = 1;
        cout << "特征6：吃肉" << endl;
        cout << "推理结果：属于食肉动物" << endl;
    }
    if (inpfea[7] && inpfea[8] && inpfea[9])
    {
        inpfea[23] = 1;
        cout << "特征7：有犬齿" << endl;
        cout << "特征8：有爪" << endl;
        cout << "特征9：眼町前方" << endl;
        cout << "推理结果：属于食肉动物" << endl;
    }
    if (inpfea[21] && inpfea[10])
    {
        inpfea[24] = 1;
        cout << "特征10：有蹄" << endl;
        cout << "推理结果：属于蹄类动物" << endl;
    }
    if (inpfea[21] && inpfea[11])
    {
        inpfea[24] = 1;
        cout << "特征11：嚼反刍" << endl;
        cout << "推理结果：属于蹄类动物" << endl;
    }

    if (inpfea[21] && inpfea[23] && inpfea[12] && inpfea[13])
    {
        answer = "金钱豹";
        cout << "特征12：黄褐色" << endl;
        cout << "特征13：身上有暗斑点" << endl;
        cout << "推理结果：属于金钱豹" << endl;
    }
    else if (inpfea[21] && inpfea[23] && inpfea[12] && inpfea[14])
    {
        answer = "虎";
        cout << "特征12：黄褐色" << endl;
        cout << "特征14：身上有黑色条纹" << endl;
        cout << "推理结果：属于虎" << endl;
    }
    else if (inpfea[24] && inpfea[15] && inpfea[16] && inpfea[13])
    {
        answer = "长颈鹿";
        cout << "特征15：有长脖子" << endl;
        cout << "特征16：有长腿" << endl;
        cout << "特征13：身上有暗斑点" << endl;
        cout << "推理结果：属于长颈鹿" << endl;
    }
    else if (inpfea[24] && inpfea[15] && inpfea[16] && inpfea[14])
    {
        answer = "斑马";
        cout << "特征15：有长脖子" << endl;
        cout << "特征16：有长腿" << endl;
        cout << "特征14：身上有黑色条纹" << endl;
        cout << "推理结果：属于斑马" << endl;
    }
    else if (inpfea[22] && inpfea[15] && inpfea[16] && inpfea[17] && inpfea[19])
    {
        answer = "鸵鸟";
        cout << "特征15：有长脖子" << endl;
        cout << "特征16：有长腿" << endl;
        cout << "特征17：不会飞" << endl;
        cout << "特征19：会游泳" << endl;
        cout << "推理结果：属于鸵鸟" << endl;
    }
    else if (inpfea[22] && inpfea[18] && inpfea[17] && inpfea[19])
    {
        answer = "企鹅";
        cout << "特征18：有黑白二色" << endl;
        cout << "特征17：不会飞" << endl;
        cout << "特征19：会游泳" << endl;
        cout << "推理结果：属于企鹅" << endl;
    }
    else if (inpfea[22] && inpfea[20])
    {
        answer = "信天翁";
        cout << "特征20：善飞" << endl;
        cout << "推理结果：属于信天翁" << endl;
    }
    else
    {
        answer = "error";
        cout << "推理结果：无法识别该动物" << endl;
    }

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
    cout << "----------------------------------------------------------\n请输入数字选择动物的特征，结尾处用end：" << endl;

    int inpfea[25] = {0};
    while (cin >> s && s != "end")
    {
        inpfea[stoi(s)] = 1;
    }
    // 输出inpfea


    string answer;
    answer = judge(inpfea);
    if (answer != "error")
        cout << "success:该动物名称为："
             << answer << endl;
    else
        cout << "error:无法识别该动物\n";

    cout << "程序成功退出!\n";
    return 0;
}