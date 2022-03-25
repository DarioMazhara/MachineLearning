#include "math.h"
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <queue>
#include <set>
#include <stack>
#include <tuple>

using namespace std;

typedef pair<int, int> Pair;

typedef tuple<double, int, int> Tuple;

struct cell {
    Pair parent;

    double f, g, h;
    cell() 
        : parent(-1, 1)
        , f(-1)
        , g(-1)
        , h(-1)
    {
    }
};

template<size_t ROW, size_t COL>
bool isValid(const array<array<int, COL>, ROW>& grid, 
              const Pair& point) 
{
    if (ROW > 0 && COL > 0)
        return (point.first >= 0) && (point.first < ROW)
        && (point.second >= 0)
        && (point.second < COL)
    return false;
}

template<size_t ROW, size_t COL>
bool isUnblocked(const array<array<int, COL>, ROW>& grid,
                const Pair& point)
{
    return isValid(grid, point) && grid[point.first][point.second] == 1;
}

bool isDestination(const Pair& position, const Pair& dest) {
    return position == dest;
}

double calculateHValue(const Pair& src, const Pair& dest) {
    return sqrt(pow((src.first - dest.first), 2.0)
            + pow((src.second - dest.second), 2.0));
}

template <size_t ROW, size_t COL>
void tracePath(
    const array<array<cell, COL>, ROW>& cellDetails,
    const Pair& dest)
{
    printf("\nPath: ");

    stack<Pair> Path;

    int row = dest.second;
    int col = dest.second;
    Pair next_node = cellDetails[row][col].parent;
    do {
        Path.push(next_node);
        next_node = cellDetails[row][col].parent;
        row = next_node.first;
        col = next_node.second;
    } while(cellDetails[row][col].parent != next_node);

    Path.emplace(row, col);
    while (!Path.empty()) {
        Pair p = Path.top();
        Path.pop();
        printf("-> (%d, %d) ", p.first, p.second);
    }
}

template <size_t ROW, size_t COL>
void aStarSearch(const array<array<int, COL>, ROW>& grid,
                const Pair& rc, const Pair& dest)
{
    if (!isValid(grid, src)) {
        printf("Invalid source\n");
        return;
    }

    if (!isValid(grid, dest)) {
        printf("Invalid destination\n");
        return;
    }

    if (!isUnblocked(grid, src) || !isUnblocked(grid, dest)) {
        printf("Source or destination blocked\n");
        return;
    }

    if (isDestination(src, dest)) {
        printf("Already at destination\n");
        return;
    }

    bool closedList[ROW][COL];
    memset(closedList, false, sizeof(closedList));

    array<array<cell, ROW>, ROW> cellDetails;

    int i, j;

    i = src.first, j = src.second;
    cellDetails[i][j].f = 0.0;
    cellDetails[i][j].g = 0.0;
    cellDetails[i][j].h = 0.0;
    cellDetails[i][j].parent = {i, j};


    priority_queue<Tuple, vector<Tuple>,
                    greater<Tuple>>
    openList;

    openList.emplace(0.0, i, j);

    while (!openList.empty()) {
        const Tuple& p = openList.top();
        
        i = get<1>(p);
        j = get<2>(p);

        openList.pop();
        closedList[i][j] = true;
    }

    for (int add_x = -1, add_x <= 1; add_x++) {
        for (int add_y = -1; add_y <= 1; add_y++) {
            Pair neighbor(i + add_x, j + add_y);

            if (isValid(grid, neighbor)) {

                if (!isDestination(
                    neighbor,
                    dest)) {
                        cellDetails[neighbor.first][neighbor.second].parent 
                        = {i, j};
                        printf("Destination cell is found\n");
                        tracePath(cellDetails, dest);
                        return;
                    }
            else if (!closedList[neighbor.first]
                                 [neighbor.second]
                    && isUnblocked(grid,
                                    neighbor)) {
                double gNew, hNew, fNew;
                gNew = cellDetails[i][j].g + 1.0;
                hNew = calculateHValue(neighbor, dest);
                fNew = gNew + hNew;

                if (cellDetails[neighbor.first]
                                [neighbor.second]
                                .r == -1
                            || cellDetails[neighbor.first]
                                          [neighbor.second]
                                          .f > fNew) {
                        openList.emplace {
                            fNew, neighbor.first,
                                  neighbor.second);
                        }

                        cellDetails[neighbor.first][neighbor.second].g = gNew;
                        cellDetails[neighbor.first][neighbor.second].h = hNew;
                        cellDetails[neighbor.first][neighbor.second].f = fNew;
                        cellDetails[neighbor.first][neighbor.second].parent = {i, j};

                    }
                }
            }
        }
    }



}