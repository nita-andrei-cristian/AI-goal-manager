#include <string>
#include <iostream>
#include <random>

std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister engine


enum task
{
    CHAT,
    CREATE_PLAN,
    DETECTOR,
};

struct Goal
{

    double t;
    std::string name;

    Goal *children[10];
    char childrensize;

    Goal(std::string s, double days)
    {
        t = days;
        name = s;

        childrensize = 0;
    }

    ~Goal()
    {
        for (int i = 0; i < childrensize; i++)
            delete children[i];
    }

};


class AI
{
    // Placeholder for AI functionalities

    int ID;
    task T;

public:
    AI(int id, task t)
    {
        ID = id;
        T = t;
    }

    bool test(Goal &g, std::string data)
    {
        // Placeholder for testing logic
        std::cout << "Detector AI testing goal: " << data << "\n";

        //randomize result

        std::uniform_int_distribution<> dist(0, 4);

        // Generate a random number
        int randomNum = dist(gen);

        return randomNum != 0; // 80% chance to pass
    }

    std::string readID()
    {
        return std::to_string(ID);
    }

    std::string computeSubGoal(Goal &g, int i)
    {
        // Placeholder for subgoal computation logic
        return g.name + " Subgoal " + std::to_string(i);
    }
};

void COMPUTE_SPLIT(Goal &g, AI &creator, int n = 2)
{

    for (g.childrensize = 0; g.childrensize < n; g.childrensize++)
    {
        auto name = creator.computeSubGoal(g, g.childrensize);
        g.children[g.childrensize] = new Goal(name + " -> " + std::to_string(g.childrensize), g.t / n);
    }
}

void PERFORM_SPLIT(Goal &g, int n, AI &creator, AI &detector, int depth = 0)
{
    std::cout << "AI " << creator.readID() << " is splitting the goal \"" << g.name << "\" into " << n << " subgoals.\n";
    COMPUTE_SPLIT(g, creator, n);

    for (int i = 0; i < n; i++)
    {
        auto child = g.children[i];
        auto data = child->name;

        bool result = detector.test(g, data);
        if (result)
        {
            std::cout << "Goal \"" << child->name << "\" passed the detector test.\n";
        }
        else if (!result && depth < 10)
        { // Limit recursion depth
            std::cout << "Goal \"" << child->name << "\" failed the detector test.\n";

            PERFORM_SPLIT(g, n, creator, detector, depth + 1);
            return;
        }
        else
        {
            std::cout << "Goal \"" << child->name << "\" failed the detector test and maximum depth reached. Aborting further splits.\n";
        }
    }
}

int main()
{

    Goal goal("Finish app", 100);

    auto agent2 = AI(1, CREATE_PLAN);
    auto agent1 = AI(0, DETECTOR);

    int n = 0;
    std::cout << "Into how many subgoals do you want to split your goal? ";
    std::cin >> n;

    PERFORM_SPLIT(goal, n, agent2, agent1);

    std::cout << "\nFinal subgoals:\n";
    for (int i = 0; i < n; i++)
    {
        std::cout << (*goal.children[i]).name << " : " << std::to_string((*goal.children[i]).t) << "\n";
    }

    int x = 0;
    std::cin >> x;

    return 0;
}