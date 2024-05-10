#include <iostream>
#include <vector>
#include <queue>
#include <functional>
#include <unordered_map>
#include <memory>
#include <algorithm> // 包含reverse函数所需的头文件

using namespace std;

struct State {
    vector<int> tiles;
    int g;
    int h;
    int f;
    shared_ptr<State> parent;

    State(const vector<int>& t, shared_ptr<State> p = nullptr)
        : tiles(t), g(0), h(0), f(0), parent(p) {}

    size_t hashCode() const {
        size_t code = 0;
        for (int tile : tiles) {
            code = code * 31 + tile; // 使用一个质数减少哈希冲突
        }
        return code;
    }

    bool operator==(const State& other) const {
        return tiles == other.tiles;
    }
};

int heuristic(const State& state) {
    int h = 0;
    int goal[9] = {1, 2, 3, 4, 5, 6, 7, 8, 0};
    for (int i = 0; i < 9; ++i) {
        if (state.tiles[i] != 0) {
            for (int j = 0; j < 9; ++j) {
                if (state.tiles[i] == goal[j]) {
                    h += abs(i % 3 - j % 3) + abs(i / 3 - j / 3);
                    break;
                }
            }
        }
    }
    return h;
}

bool aStar(shared_ptr<State>& initialState, const vector<int>& goal, vector<vector<int>>& solutionPath) {
    priority_queue<pair<int, shared_ptr<State>>, vector<pair<int, shared_ptr<State>>>, greater<pair<int, shared_ptr<State>>>> openQueue;
    unordered_map<size_t, shared_ptr<State>> visited;

    initialState->g = 0;
    initialState->h = heuristic(*initialState);
    initialState->f = initialState->g + initialState->h;
    openQueue.push({initialState->f, initialState});
    visited[initialState->hashCode()] = initialState;

    while (!openQueue.empty()) {
        auto current = openQueue.top().second;
        openQueue.pop();

        if (current->tiles == goal) {
            solutionPath.push_back(current->tiles);
            auto state = current;
            while (state->parent) {
                state = state->parent;
                solutionPath.push_back(state->tiles);
            }
            reverse(solutionPath.begin(), solutionPath.end()); // 反转路径，使其从初始状态到目标状态
            return true;
        }

        int emptyPos = current->tiles.size() - 1 - distance(begin(current->tiles), find(begin(current->tiles), end(current->tiles), 0));
        vector<vector<int>> directions = {
            {0, 0}, {-1, 0}, {0, -1}, {-1, -1}, {0, 1}, {1, -1}, {1, 0}, {0, 1}, {1, 1}
        };

        for (auto& dir : directions) {
            int newX = emptyPos / 3 + dir[0];
            int newY = emptyPos % 3 + dir[1];
            if (newX >= 0 && newX < 3 && newY >= 0 && newY < 3) {
                vector<int> childTiles = current->tiles;
                swap(childTiles[emptyPos], childTiles[3 * newY + newX]);
                auto child = make_shared<State>(childTiles, current);
                size_t childHash = child->hashCode();

                if (visited.find(childHash) == visited.end() || visited[childHash]->g > child->g) {
                    child->g = current->g + 1;
                    child->h = heuristic(*child);
                    child->f = child->g + child->h;
                    openQueue.push({child->f, child});
                    visited[childHash] = child;
                }
            }
        }
    }

    return false; // 没有找到解决方案
}

int main() {
    vector<int> initialState = {1, 2, 3, 4, 5, 6, 7, 8, 0}; // 初始状态
    vector<int> goalState = {1, 2, 3, 4, 6, 5, 7, 8, 0};   // 目标状态
    vector<vector<int>> solutionPath;

    auto initialStatePtr = make_shared<State>(initialState);

    if (aStar(initialStatePtr, goalState, solutionPath)) {
        cout << "Solution found:" << endl;
        for (const auto& path : solutionPath) {
            for (int tile : path) {
                cout << tile << " ";
            }
            cout << endl;
        }
    } else {
        cout << "No solution found." << endl;
    }

    return 0;
}